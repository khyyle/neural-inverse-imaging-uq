import sys
from flax.serialization import to_bytes, from_bytes
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#import ehtim as eh
from typing import Any, Callable
import functools

import jax
import flax
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import ehtim as eh
from skimage import data
from PIL import Image


def load_img(image_type):
    intensity_gt: jnp.array
    xdim: int
    ydim: int
    if image_type == 'black hole':
        image_path = 'datasets/avery_sgra_eofn.txt'
        image_true = eh.image.load_txt(image_path)
        image_true.display()

        intensity_gt = jnp.array(image_true.imarr(), dtype=jnp.float32)
        ydim, xdim = intensity_gt.shape

        assert intensity_gt.size == image_true.imvec.size
    elif image_type == 'camera':
        img = data.camera()
        intensity_gt = jnp.array(img, dtype=jnp.float32) / 255.0
        ydim, xdim = intensity_gt.shape

        assert intensity_gt.size == img.size

        plt.figure()
        plt.title('Ground truth (full)')
        plt.imshow(intensity_gt, cmap='gray')
        plt.axis('off')
        plt.show()
    elif image_type == 'starfish':
        img = Image.open('datasets/starfish.png').convert('RGB')
        print(img.size)
        W, H = img.size
        scale = 256/max(W, H)
        xdim, ydim = (round(W*scale), round(H*scale))
        img = img.resize((xdim, ydim), resample=Image.Resampling.LANCZOS)

        rgb = np.asarray(img, dtype=np.float32)/255.0
        intensity_gt = rgb.mean(axis=-1, dtype=np.float32)
        #gray_pil = Image.fromarray((intensity_gt * 255).astype(np.uint8), mode="L")
        #gray_pil.show()
        
        #plt.figure()
        #plt.title('Ground truth (full)')
        #plt.imshow(intensity_gt, cmap='gray')
        #plt.axis('off')
        #plt.show()

    return xdim, ydim, intensity_gt


# build coords and relevant parameters for later
img_type = 'black hole'
xdim, ydim, intensity_gt = load_img(img_type)

x, y = np.linspace(0, 1, xdim), np.linspace(0, 1, ydim)
coords = np.moveaxis(np.array(np.meshgrid(x, y, indexing='xy')), 0, -1)
coords_flat = coords.reshape(-1,2)

gt_flat = intensity_gt.ravel()
I_max = float(max(gt_flat))
npix = intensity_gt.size


class MLP(nn.Module): 
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu 
    out_channel: int = 1
    do_skip: bool = True
  
    @nn.compact
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform()) 

        if self.do_skip:
            skip_layer = self.net_depth // 2 

        inputs = x
        for i in range(self.net_depth): 
            x = dense_layer(self.net_width)(x)
            x = self.activation(x) 
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)

        return out

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1)) 
    return jnp.concatenate([x] + [four_feat], axis=-1)

class NeuralImage(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    posenc_deg: int, default=3
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    posenc_deg: int = 3
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    
    @nn.compact
    def __call__(self, coords):
        image_MLP = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        def predict_image(coords):
            net_output = image_MLP(posenc(coords, self.posenc_deg))
            image = I_max*nn.sigmoid(net_output[..., 0])

            return image
        return predict_image(coords)
    
safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

def fourier_forward(image_pred, n_crop):
    """
    Predict the image in Fourier space and crop it in the center
    with dimensions n_crop x n_crop.

    Args:
        image_pred (jnp.ndarray(float32)): the predicted image
        n_crop (int): the size of the crop to take from the center of the Fourier transform
    Returns:
        F_c (jnp.ndarray(float32)): the cropped Fourier transform of the predicted image
    """
    F = jnp.fft.fftshift(jnp.fft.fft2(image_pred, norm = "ortho"))
    cx = (image_pred.shape[1] - n_crop) // 2
    cy = (image_pred.shape[0] - n_crop) // 2
    F_c = jax.lax.dynamic_slice(F, start_indices=(cy, cx), slice_sizes=(n_crop, n_crop))
    return F_c.ravel()


def loss_fn(params, predictor_fn, target, sigma_vis, coords, n_crop):
    '''
    Args:
        params: pytree (nested dict) of all of the model's weights and biases. 
        predictor_fn: the model's apply function
        target: measured intensities
        A: measurement matrix
        sigma: Thermal noise per visibility
        coords: array of shape (N_pixels, 2) with all (x,y) grid coords
    
    Returns:
        image: predicted intensities -> image
        loss: 
    '''
    image_pred = predictor_fn({'params': params}, coords)
    vis = fourier_forward(image_pred, n_crop)
    loss = jnp.mean((jnp.abs(vis - target)/sigma_vis)**2)
    return loss, image_pred

@functools.partial(jax.jit, static_argnums=(4,))
def train_step(state, target, sigma_vis, coords, n_crop):
    (loss, image), grads = jax.value_and_grad(loss_fn, argnums=(0), has_aux=True)(state.params, state.apply_fn, target, sigma_vis, coords, n_crop)
    state = state.apply_gradients(grads=grads)
    return loss, state, image

predictor = NeuralImage()

"""
Training loop

lets try on black hole image, camera guy, starfish gpt img from brandon (can average over channel axis, can resize w anti aliasing (pil))
also do ensemble on each and compare uncertainty maps. 
"""
if __name__ == "__main__":
    n_crop = xdim #no crop
    nvis = n_crop**2
    rng_real, rng_im = jax.random.PRNGKey(2), jax.random.PRNGKey(3)
    
    vis_true = fourier_forward(intensity_gt, n_crop)
    
    vis_obs = vis_true
    uu, vv = np.meshgrid(np.arange(-n_crop//2, n_crop//2),
                     np.arange(-n_crop//2, n_crop//2), indexing='ij')
    r = np.sqrt(uu**2 + vv**2) 
    r = r / r.max() # normalize
    sigma_vis = jnp.asarray(1e-4 + 8e-4*(r.ravel()), dtype=jnp.float32)
    vis_obs += sigma_vis * (jax.random.normal(rng_real, (nvis,)) +
                     1j * jax.random.normal(rng_im, (nvis,))) / jnp.sqrt(2)
    
    SEED = int(sys.argv[1])
    hparams = {'num_iters': 20000, 'lr_init': 1e-3, 'lr_final': 9e-4, 'batchsize': 500}
    params = predictor.init(jax.random.PRNGKey(SEED), coords)['params']
    tx = optax.adam(learning_rate=hparams['lr_init'])
    state = train_state.TrainState.create(apply_fn=predictor.apply, params=params, tx=tx)

    for i in tqdm(range(hparams['num_iters']), desc='iteration'):
        #batch = np.random.choice(vis_obs.size, hparams['batchsize'], replace=False)
        loss, state, image = train_step(state, vis_obs, sigma_vis, coords, n_crop)
        if i % 1000 == 0:
            print(f"iteration {i}, loss={loss:.9e}")
    
    fname = f"fourier_ensembling/bh_varnoise_models/params_{SEED}.msgpack"  
    with open(fname, "wb") as fp:  
        fp.write(to_bytes(state.params))  
    print(f"[seed={SEED}] saved params to {fname}")