# Deformation BayesRays for Linear Inverse Imaging

This document derives the two-dimensional deformation BayesRays method used
for neural inverse imaging. It begins with the likelihood used to train an
image model, introduces a deformation-field reparameterization, derives the
Laplace covariance through Fisher information, and propagates deformation
covariance from a coarse grid to every image coordinate.

This derivation adapts Goli et al. 2023, [*Bayes' Rays: Uncertainty Quantification for
Neural Radiance Fields*](https://arxiv.org/abs/2309.03185), to real or complex
linear measurements.

## 1. Trained neural image and measurement likelihood

Let

$$
f_\phi : \mathbb R^2 \rightarrow \mathbb R
$$

be a neural image with parameters $\phi$. Evaluating it at image coordinate
$u_p\in\mathbb R^2$ gives pixel intensity

$$
x_\phi[p] = f_\phi(u_p).
$$

Collecting all $N$ pixels gives

$$
x_\phi \in \mathbb R^N.
$$

A fixed linear measurement operator $\mathcal A$ maps the image to $M$
predicted measurements:

$$
\widehat y_\phi = \mathcal A x_\phi.
$$

For a dense operator, $\mathcal A$ is multiplication by a matrix
$A\in\mathbb C^{M\times N}$. For computed operators such as the Radon
transform, $\mathcal A$ can remain implicit.

The observed data are

$$
y = \widehat y_\phi + \eta,
$$

where the measurements are conditionally independent and measurement $m$ has
known standard deviation $\sigma_m>0$.

For real measurements,

$$
\eta_m \sim \mathcal N(0,\sigma_m^2).
$$

The Gaussian likelihood is

$$
p(y\mid\phi)
\propto
\exp\left[
-\frac12
\sum_{m=1}^{M}
\frac{
\left(\widehat y_{\phi,m}-y_m\right)^2
}{
\sigma_m^2
}
\right].
$$

The negative log-likelihood, up to constants independent of $\phi$, is

$$
\mathcal L_{\mathrm{data}}(\phi)
=
\frac{1}{2M}
\sum_{m=1}^{M}
\frac{
\left(\widehat y_{\phi,m}-y_m\right)^2
}{
\sigma_m^2
}.
$$

The factor $1/M$ makes this the empirical mean of per-measurement losses. It
does not change the maximum-likelihood solution, but it fixes the scale at
which regularization is compared with the data term.

For complex measurements, write

$$
y_m = y_{\mathrm{Re},m} + i y_{\mathrm{Im},m}
$$

and assume that the real and imaginary noise components are independent with
the same per-component standard deviation:

$$
\eta_{\mathrm{Re},m},\eta_{\mathrm{Im},m}
\sim
\mathcal N(0,\sigma_m^2).
$$

Then

$$
\left|
\widehat y_{\phi,m}-y_m
\right|^2
=
\left(
\widehat y_{\mathrm{Re},\phi,m}-y_{\mathrm{Re},m}
\right)^2
+
\left(
\widehat y_{\mathrm{Im},\phi,m}-y_{\mathrm{Im},m}
\right)^2,
$$

and the negative log-likelihood is

$$
\mathcal L_{\mathrm{data}}(\phi)
=
\frac{1}{2M}
\sum_{m=1}^{M}
\frac{
\left|
\widehat y_{\phi,m}-y_m
\right|^2
}{
\sigma_m^2
}.
$$

Thus, taking the squared complex magnitude includes both real and imaginary
residuals. It does not discard the imaginary component.

Training produces optimized neural-image parameters

$$
\phi^\star
\approx
\underset{\phi}{\operatorname{arg\,min}}\,
\mathcal L_{\mathrm{data}}(\phi).
$$

Deformation BayesRays keeps $\phi^\star$ fixed.

## 2. Bayesian posterior and deformation reparameterization

Bayes' rule for the original model parameters is

$$
p(\phi\mid\mathcal I)
=
\frac{
p(\mathcal I\mid\phi)p(\phi)
}{
p(\mathcal I)
}.
$$

Here $\mathcal I$ denotes the training observations; for the linear inverse
problems considered below, those observations are the measurement vector $y$.

Instead of approximating this posterior over every neural-network parameter,
BayesRays introduces a lower-dimensional, spatially meaningful parameter
$\theta$.

Let a deformation grid contain $G_x\times G_y$ nodes. Each node stores a
two-dimensional coordinate offset:

$$
\theta_{i,j}
=
\begin{bmatrix}
\theta_{i,j,x}\\
\theta_{i,j,y}
\end{bmatrix}.
$$

The complete deformation parameter count is

$$
P = 2G_xG_y.
$$

Bilinear interpolation defines a continuous displacement field:

$$
D_\theta(u)
=
\operatorname{Bilinear}(u,\theta).
$$

The neural image is reparameterized by perturbing its input coordinates:

$$
x_\theta[p]
=
f_{\phi^\star}
\left(
u_p+D_\theta(u_p)
\right).
$$

The corresponding predicted measurements are

$$
\widehat y_\theta
=
\mathcal A x_\theta.
$$

Therefore, the deformation enters the likelihood through the following
composition:

$$
\theta
\longmapsto
D_\theta(u)
\longmapsto
f_{\phi^\star}\left(u+D_\theta(u)\right)
\longmapsto
\mathcal A x_\theta
\longmapsto
p(y\mid\theta).
$$

The deformation posterior is

$$
p(\theta\mid\mathcal I)
=
\frac{
p(\mathcal I\mid\theta)p(\theta)
}{
p(\mathcal I)
}.
$$

The evidence $p(\mathcal I)$ is independent of $\theta$, so the negative
log-posterior can be written, up to an additive constant, as

$$
\mathcal L_{\mathrm{post}}(\theta)
=
\mathcal L_{\mathrm{data}}(\theta)
-\log p(\theta).
$$

Choose an independent zero-centered Gaussian prior. If the prior precision is
$\tau$, then

$$
\theta \sim \mathcal N(0,\tau^{-1}I)
$$

and

$$
-\log p(\theta)
=
\frac{\tau}{2}\lVert\theta\rVert^2
+ \text{constant}.
$$

The BayesRays paper writes the penalty as

$$
\lambda\lVert\theta\rVert^2.
$$

This corresponds to

$$
\tau = 2\lambda,
$$

so the prior contributes

$$
2\lambda I
$$

to posterior precision.

At $\theta=0$, every grid offset is zero:

$$
D_0(u)=0.
$$

Consequently,

$$
x_0[p]=f_{\phi^\star}(u_p)
$$

and

$$
\widehat y_0=\mathcal A x_{\phi^\star}.
$$

BayesRays assumes that a spatial deformation cannot improve the already
optimized reconstruction in expectation. Under this assumption, the
deformation posterior mode is

$$
\theta^\star=0.
$$

This is an approximation rather than a theorem: optimization error or model
misspecification can make the deformation gradient at zero nonzero.

## 3. Laplace approximation around zero deformation

Let

$$
h(\theta)
=
-\log p(\theta\mid\mathcal I)
$$

be the negative log-posterior. A second-order Taylor expansion around
$\theta^\star=0$ gives

$$
h(\theta)
\approx
h(0)
+
\nabla h(0)^{\mathsf T}\theta
+
\frac12
\theta^{\mathsf T}
H(0)
\theta,
$$

where

$$
H(0)=\nabla_\theta^2 h(0).
$$

At a posterior mode,

$$
\nabla h(0)=0.
$$

The quadratic approximation is therefore

$$
h(\theta)
\approx
h(0)
+
\frac12
\theta^{\mathsf T}H(0)\theta.
$$

Exponentiating its negative gives a Gaussian approximation:

$$
p(\theta\mid\mathcal I)
\approx
\mathcal N
\left(
0,
\Sigma_\theta
\right),
$$

with

$$
\Sigma_\theta
\approx
H(0)^{-1}.
$$

The covariance is the inverse Hessian of the negative log-posterior. Some
presentations instead define the Hessian of the log-posterior, which is
negative at a mode and introduces an additional minus sign. Using the
negative log-posterior avoids that sign ambiguity.

## 4. Fisher information from the score

For a random observation $Y$ with density $p(Y\mid\theta)$, define the score

$$
s_\theta(Y)
=
\nabla_\theta\log p(Y\mid\theta).
$$

Under standard regularity conditions, the expected score is zero:

$$
\mathbb E[s_\theta(Y)]
=
\int
p(y\mid\theta)
\nabla_\theta\log p(y\mid\theta)
\,dy.
$$

Because

$$
p(y\mid\theta)
\nabla_\theta\log p(y\mid\theta)
=
\nabla_\theta p(y\mid\theta),
$$

it follows that

$$
\mathbb E[s_\theta(Y)]
=
\int
\nabla_\theta p(y\mid\theta)
\,dy
=
\nabla_\theta
\int
p(y\mid\theta)
\,dy
=
\nabla_\theta 1
=
0.
$$

The Fisher information is the covariance of this vector-valued score:

$$
F(\theta)
=
\operatorname{Cov}[s_\theta(Y)].
$$

Since the expected score is zero,

$$
F(\theta)
=
\mathbb E
\left[
s_\theta(Y)s_\theta(Y)^{\mathsf T}
\right].
$$

The word *variance* refers here to a covariance matrix in parameter space, not
to an elementwise scalar variance.

The equivalent expected-Hessian identity is

$$
F(\theta)
=
-\mathbb E
\left[
\nabla_\theta^2\log p(Y\mid\theta)
\right].
$$

For the negative log-likelihood

$$
\ell(\theta)=-\log p(Y\mid\theta),
$$

this becomes

$$
F(\theta)
=
\mathbb E
\left[
\nabla_\theta^2\ell(\theta)
\right].
$$

## 5. Fisher information for a Gaussian measurement

First consider one real scalar measurement:

$$
y\mid\theta
\sim
\mathcal N
\left(
\mu_\theta,
\sigma^2
\right).
$$

Define the residual and measurement Jacobian:

$$
r_\theta=\mu_\theta-y,
$$

$$
j_\theta
=
\frac{\partial\mu_\theta}{\partial\theta}
\in
\mathbb R^{1\times P}.
$$

The negative log-likelihood is

$$
\ell(\theta)
=
\frac{r_\theta^2}{2\sigma^2}
+\text{constant}.
$$

Its gradient is

$$
\nabla_\theta\ell(\theta)
=
\frac{r_\theta}{\sigma^2}
j_\theta^{\mathsf T}.
$$

The outer product of the score is therefore

$$
s_\theta s_\theta^{\mathsf T}
=
\frac{r_\theta^2}{\sigma^4}
j_\theta^{\mathsf T}j_\theta.
$$

The assumed likelihood gives

$$
\mathbb E[r_\theta^2]=\sigma^2.
$$

Taking the expectation gives

$$
F(\theta)
=
\frac{1}{\sigma^2}
j_\theta^{\mathsf T}j_\theta.
$$

The exact negative-log-likelihood Hessian also contains a term involving the
second derivative of $\mu_\theta$:

$$
\nabla_\theta^2\ell
=
\frac{1}{\sigma^2}
j_\theta^{\mathsf T}j_\theta
+
\frac{r_\theta}{\sigma^2}
\nabla_\theta^2\mu_\theta.
$$

The expected residual is zero, so the second term vanishes in expectation.
Keeping only the first term for observed data is the generalized
Gauss--Newton or empirical Fisher approximation used here.

### Direct Hessian, generalized Gauss--Newton, and Fisher

The same result follows more directly by differentiating the complete
Gaussian data term. Let $\mu_\theta\in\mathbb R^M$ contain all predicted
measurements, let

$$
r_\theta=\mu_\theta-y,
\qquad
\Lambda=\operatorname{diag}
\left(
\sigma_1^{-2},\ldots,\sigma_M^{-2}
\right),
$$

and let

$$
J_\theta
=
\frac{\partial\mu_\theta}{\partial\theta}
\in\mathbb R^{M\times P}.
$$

Using the mean-loss convention, the negative log-likelihood is

$$
\mathcal L(\theta)
=
\frac{1}{2M}
r_\theta^{\mathsf T}\Lambda r_\theta.
$$

Direct differentiation gives

$$
\nabla_\theta\mathcal L
=
\frac{1}{M}
J_\theta^{\mathsf T}\Lambda r_\theta,
$$

followed by the exact Hessian

$$
\nabla_\theta^2\mathcal L
=
\frac{1}{M}
J_\theta^{\mathsf T}\Lambda J_\theta
+
\frac{1}{M}
\sum_{m=1}^{M}
\frac{r_{\theta,m}}{\sigma_m^2}
\nabla_\theta^2\mu_{\theta,m}.
$$

The first term is the generalized Gauss--Newton matrix,

$$
G(\theta)
=
\frac{1}{M}
J_\theta^{\mathsf T}\Lambda J_\theta.
$$

It uses only first derivatives of the prediction, is positive semidefinite,
and is independent of the observed residual values. The second term measures
the curvature of the prediction function itself, weighted by the residuals.
It is zero when the prediction is linear in $\theta$, zero at an exact fit,
and zero in expectation under the assumed likelihood because
$\mathbb E[r_\theta]=0$. Consequently,

$$
\mathbb E
\left[
\nabla_\theta^2\mathcal L
\right]
=
G(\theta)
=
F(\theta).
$$

Thus the expected Fisher and generalized Gauss--Newton matrix coincide for
this Gaussian likelihood. BayesRays uses the score/Fisher route because it
immediately produces a first-derivative, positive-semidefinite approximation.
Directly differentiating the likelihood is equally valid and makes clear
which residual-weighted second-order term is being removed.

### The factor $4\epsilon J^{\mathsf T}J$

The BayesRays paper assumes

$$
\sigma^2=\frac12.
$$

Then

$$
\frac{1}{2\sigma^2}=1,
$$

so

$$
\ell(\theta)=r_\theta^2.
$$

Its gradient is

$$
\nabla_\theta\ell
=
2r_\theta j_\theta^{\mathsf T}.
$$

The score outer product is

$$
4r_\theta^2
j_\theta^{\mathsf T}j_\theta.
$$

Defining

$$
\epsilon_\theta=r_\theta^2
$$

produces the paper's expression

$$
4\epsilon_\theta
j_\theta^{\mathsf T}j_\theta.
$$

Finally,

$$
\mathbb E[\epsilon_\theta]
=
\mathbb E[r_\theta^2]
=
\frac12,
$$

so

$$
4\mathbb E[\epsilon_\theta]
j_\theta^{\mathsf T}j_\theta
=
2j_\theta^{\mathsf T}j_\theta.
$$

For vector-valued outputs, the precise score outer product contains the
residual outer product rather than only its squared norm. The scalar notation
in the paper relies on isotropic output noise.

## 6. Heteroscedastic real and complex measurements

For $M$ real measurements with potentially different noise scales, let

$$
j_m
=
\frac{\partial\widehat y_{\theta,m}}{\partial\theta}
\in
\mathbb R^{1\times P}.
$$

The paper defines Fisher as an expectation over measurements. Its
deterministic empirical estimate is

$$
F
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{
j_m^{\mathsf T}j_m
}{
\sigma_m^2
}.
$$

The factor $1/M$ follows from approximating the measurement expectation by a
sample mean. If the full independent-dataset log-likelihood were expressed as
a sum instead, the Fisher would also be a sum. These conventions have the
same maximum-likelihood solution but require correspondingly scaled prior
coefficients to produce the same posterior covariance.

For complex measurements, define

$$
J
=
J_{\mathrm{Re}}
+
iJ_{\mathrm{Im}}
\in
\mathbb C^{M\times P}.
$$

The Fisher for independent real and imaginary components is

$$
F
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{
j_{\mathrm{Re},m}^{\mathsf T}j_{\mathrm{Re},m}
+
j_{\mathrm{Im},m}^{\mathsf T}j_{\mathrm{Im},m}
}{
\sigma_m^2
}.
$$

Define the measurement-noise precision matrix

$$
\Lambda_{\mathrm{noise}}
=
\operatorname{diag}
\left(
\sigma_1^{-2},
\ldots,
\sigma_M^{-2}
\right).
$$

Expanding the complex Gram matrix gives

$$
J^H\Lambda_{\mathrm{noise}}J
=
J_{\mathrm{Re}}^{\mathsf T}
\Lambda_{\mathrm{noise}}
J_{\mathrm{Re}}
+
J_{\mathrm{Im}}^{\mathsf T}
\Lambda_{\mathrm{noise}}
J_{\mathrm{Im}}
$$

$$
\quad
+
i
\left(
J_{\mathrm{Re}}^{\mathsf T}
\Lambda_{\mathrm{noise}}
J_{\mathrm{Im}}
-
J_{\mathrm{Im}}^{\mathsf T}
\Lambda_{\mathrm{noise}}
J_{\mathrm{Re}}
\right).
$$

Therefore,

$$
F
=
\frac{1}{M}
\operatorname{Re}
\left(
J^H
\Lambda_{\mathrm{noise}}
J
\right).
$$

For deformation parameters $k$ and $\ell$, the individual Fisher entry is

$$
F_{k,\ell}
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{
J_{\mathrm{Re},m,k}J_{\mathrm{Re},m,\ell}
+
J_{\mathrm{Im},m,k}J_{\mathrm{Im},m,\ell}
}{
\sigma_m^2
}.
$$

On the diagonal,

$$
F_{k,k}
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{
J_{\mathrm{Re},m,k}^2
+
J_{\mathrm{Im},m,k}^2
}{
\sigma_m^2
}.
$$

In ehtim, the supplied visibility noise scale is used as the standard
deviation of each real and imaginary component. It is therefore used directly;
it is not divided by $\sqrt{2}$.

## 7. Deterministic measurement batching

The full measurement Jacobian has shape

$$
J\in\mathbb C^{M\times P}.
$$

Materializing it can be expensive. For a chunk of $B$ measurements, only

$$
J_{\mathrm{chunk}}
\in
\mathbb C^{B\times P}
$$

is constructed.

For diagonal Fisher, each chunk contributes

$$
\operatorname{diag}
\left(
J_{\mathrm{chunk}}^H
J_{\mathrm{chunk}}
\right)_k
=
\sum_{b=1}^{B}
|J_{\mathrm{chunk},b,k}|^2.
$$

For full Fisher, each chunk contributes

$$
J_{\mathrm{chunk}}^H
J_{\mathrm{chunk}}.
$$

Every measurement is visited exactly once, so this batching is deterministic
and introduces no statistical approximation. It only limits the number of
Jacobian rows held in memory at once.

After all chunks have been accumulated,

$$
F_{\mathrm{empirical}}
=
\frac{1}{M}
\sum_{\mathrm{chunks}}
F_{\mathrm{chunk}}.
$$

## 8. Deformation posterior covariance

Adding the Gaussian deformation prior gives posterior precision

$$
\Lambda_\theta
=
F_{\mathrm{empirical}}
+
2\lambda I.
$$

The Laplace covariance is

$$
\Sigma_\theta
=
\Lambda_\theta^{-1}.
$$

For a diagonal Fisher approximation,

$$
\left[\Sigma_\theta\right]_{k,k}
=
\frac{1}{
F_{k,k}+2\lambda
}.
$$

Only the variance vector is stored:

$$
v
=
\operatorname{diag}(\Sigma_\theta)
\in
\mathbb R^P.
$$

It can be reshaped to the deformation lattice:

$$
v_{\mathrm{grid}}
\in
\mathbb R^{G_x\times G_y\times2}.
$$

For full covariance,

$$
\Sigma_\theta
\in
\mathbb R^{P\times P}
$$

is constructed explicitly. This is practical only for small deformation
grids.

## 9. Bilinear deformation interpolation

For a normalized coordinate

$$
u=(u_x,u_y),
$$

the continuous grid coordinates are

$$
x=u_x(G_x-1),
$$

$$
y=u_y(G_y-1).
$$

The surrounding grid indices are

$$
i_0=\lfloor x\rfloor,
\qquad
i_1=\min(i_0+1,G_x-1),
$$

$$
j_0=\lfloor y\rfloor,
\qquad
j_1=\min(j_0+1,G_y-1).
$$

Define within-cell fractions

$$
w_x=x-i_0,
$$

$$
w_y=y-j_0.
$$

The four bilinear weights are

$$
w_{00}=(1-w_x)(1-w_y),
$$

$$
w_{10}=w_x(1-w_y),
$$

$$
w_{01}=(1-w_x)w_y,
$$

$$
w_{11}=w_xw_y.
$$

The interpolated displacement is

$$
D_\theta(u)
=
w_{00}\theta_{i_0,j_0}
+
w_{10}\theta_{i_1,j_0}
+
w_{01}\theta_{i_0,j_1}
+
w_{11}\theta_{i_1,j_1}.
$$

## 10. Covariance at an image coordinate

Only four deformation nodes influence one image coordinate. Each node has two
scalar displacement components, so the local parameter vector has eight
entries:

$$
\theta_{\mathrm{local}}
=
\begin{bmatrix}
\theta_{00,x}&
\theta_{00,y}&
\theta_{10,x}&
\theta_{10,y}&
\theta_{01,x}&
\theta_{01,y}&
\theta_{11,x}&
\theta_{11,y}
\end{bmatrix}^{\mathsf T}
\in
\mathbb R^8.
$$

The local covariance is the corresponding submatrix of the global deformation
covariance:

$$
\Sigma_{\mathrm{local}}
=
\Sigma_\theta
\left[
\mathcal I(u),
\mathcal I(u)
\right]
\in
\mathbb R^{8\times8},
$$

where $\mathcal I(u)$ contains the eight selected global parameter indices.

The interpolation matrix is

$$
W(u)
=
\begin{bmatrix}
w_{00}&0&w_{10}&0&w_{01}&0&w_{11}&0\\
0&w_{00}&0&w_{10}&0&w_{01}&0&w_{11}
\end{bmatrix}
\in
\mathbb R^{2\times8}.
$$

It maps local deformation parameters to the two-dimensional displacement:

$$
D_\theta(u)
=
W(u)\theta_{\mathrm{local}}.
$$

For a random vector $\theta$ and fixed linear map $W$,

$$
\operatorname{Cov}(W\theta)
=
W\operatorname{Cov}(\theta)W^{\mathsf T}.
$$

Therefore, the displacement covariance at $u$ is

$$
\Sigma_D(u)
=
W(u)
\Sigma_{\mathrm{local}}
W(u)^{\mathsf T}
\in
\mathbb R^{2\times2}.
$$

Its entries are

$$
\Sigma_D(u)
=
\begin{bmatrix}
\operatorname{Var}(D_x) &
\operatorname{Cov}(D_x,D_y)\\
\operatorname{Cov}(D_y,D_x) &
\operatorname{Var}(D_y)
\end{bmatrix}.
$$

The scalar spatial uncertainty is the root sum of displacement variances:

$$
U(u)
=
\sqrt{
\operatorname{tr}
\left(
\Sigma_D(u)
\right)
}
=
\sqrt{
\operatorname{Var}(D_x)
+
\operatorname{Var}(D_y)
}.
$$

For diagonal $\Sigma_\theta$, covariance propagation simplifies to

$$
U(u)^2
=
\sum_{q\in\{00,10,01,11\}}
w_q(u)^2
\left(
v_{q,x}+v_{q,y}
\right).
$$

The squared interpolation weights arise from

$$
\operatorname{Var}(wX)=w^2\operatorname{Var}(X).
$$

For full covariance, the complete expression

$$
W\Sigma_{\mathrm{local}}W^{\mathsf T}
$$

retains correlations between neighboring nodes and between displacement
components.

The BayesRays paper defines an uncertainty field by interpolating scalar node
standard deviations. Direct covariance propagation is more precise for the
random bilinear deformation field and matches the final direct-evaluation path
in the CT notebook.

## 11. Implementation correspondence

The numerical implementation follows this order:

1. `bilinear_interpolate` evaluates $D_\theta(u)$.
2. `DeformationGrid` owns zero-initialized deformation-node offsets.
3. `deformation_fisher_information` differentiates
   $\mathcal A f_{\phi^\star}(u+D_\theta(u))$ with respect to node offsets and
   accumulates Fisher information in deterministic measurement chunks.
4. `deformation_laplace` adds $2\lambda I$ and computes diagonal or full
   posterior covariance.
5. `deformation_uncertainty_map` propagates posterior covariance to every image
   coordinate.

The calculation assumes:

- the trained neural-image parameters $\phi^\star$ remain fixed;
- the deformation posterior is locally approximated around $\theta=0$;
- measurement noise is Gaussian and independent across measurements;
- real and imaginary visibility components share the supplied per-component
  standard deviation;
- the forward operator and neural image are differentiable with respect to
  image coordinates;
- full covariance is used only for deformation grids small enough to store and
  invert a $P\times P$ matrix.
