"""Coordinate-deformation BayesRays (based on Goli et al. 2023) for two-dimensional image models"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tqdm.auto import tqdm

from ..forward import LinearInverseProblem

type ParameterTree = Any


def bilinear_interpolate(
    coordinates: jax.Array,
    grid_values: jax.Array,
) -> jax.Array:
    """
    Bilinearly interpolate a two-dimensional grid at normalized coordinates.

    Parameters:
    -----------
    coordinates: jax.Array
        Coordinates with shape `(..., 2)` and values in `[0, 1]`.
    grid_values: jax.Array
        Scalar values with shape `(grid_size_x, grid_size_y)` or vector values
        with additional trailing dimensions.

    Returns:
    --------
    jax.Array
        Interpolated values with shape
        `coordinates.shape[:-1] + grid_values.shape[2:]`.
    """
    if coordinates.shape[-1] != 2:
        raise ValueError("`coordinates` must have two components.")
    if grid_values.ndim < 2:
        raise ValueError("`grid_values` must have at least two dimensions.")

    # Convert normalized coordinates to continuous deformation-grid indices.
    grid_size_x, grid_size_y = grid_values.shape[:2]
    x = coordinates[..., 0] * (grid_size_x - 1)
    y = coordinates[..., 1] * (grid_size_y - 1)

    # integer pairs 00, 10, 01, and 11 bound each query coordinate
    i0 = jnp.floor(x).astype(jnp.int32)
    j0 = jnp.floor(y).astype(jnp.int32)
    i1 = jnp.clip(i0 + 1, 0, grid_size_x - 1)
    j1 = jnp.clip(j0 + 1, 0, grid_size_y - 1)

    wx = x - i0
    wy = y - j0

    theta_00 = grid_values[i0, j0]
    theta_10 = grid_values[i1, j0]
    theta_01 = grid_values[i0, j1]
    theta_11 = grid_values[i1, j1]

    # D(u) = w00 t00 + w10 t10 + w01 t01 + w11 t11.
    trailing_dimensions = grid_values.ndim - 2
    weight_shape = wx.shape + (1,) * trailing_dimensions
    weight_00 = ((1.0 - wx) * (1.0 - wy)).reshape(weight_shape)
    weight_10 = (wx * (1.0 - wy)).reshape(weight_shape)
    weight_01 = ((1.0 - wx) * wy).reshape(weight_shape)
    weight_11 = (wx * wy).reshape(weight_shape)
    return (
        weight_00 * theta_00
        + weight_10 * theta_10
        + weight_01 * theta_01
        + weight_11 * theta_11
    )


class DeformationGrid(nn.Module):
    """
    Store a zero-centered two-dimensional deformation field.

    Parameters:
    -----------
    resolution: tuple[int, int]
        Number of deformation nodes along both coordinate dimensions.
    """

    resolution: tuple[int, int]

    @nn.compact
    def __call__(self, coordinates: jax.Array) -> jax.Array:
        """
        Interpolate deformation-node offsets at normalized coordinates.

        Parameters:
        -----------
        coordinates: jax.Array
            Coordinates with shape `(..., 2)`.

        Returns:
        --------
        jax.Array
            Two-dimensional offsets with shape `coordinates.shape`.
        """
        if len(self.resolution) != 2 or any(size < 2 for size in self.resolution):
            raise ValueError(
                "`resolution` must contain two values of at least two."
            )
        node_offsets = self.param(
            "node_offsets",
            nn.initializers.zeros, # posterior mode is theta* = 0
            self.resolution + (2,), # shape (grid_x, grid_y, 2)
        )
        return bilinear_interpolate(coordinates, node_offsets)


def deformation_fisher_information(
    forward_model: LinearInverseProblem,
    render_function: Callable[[ParameterTree, jax.Array], jax.Array],
    model_parameters: ParameterTree,
    coordinates: jax.Array,
    deformation_grid: DeformationGrid,
    *,
    diagonal: bool = True,
    measurement_chunk_size: int = 8,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Accumulate deformation Fisher information in measurement chunks.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Measurement operator, observed values, and per-component noise.
    render_function: Callable[[ParameterTree, jax.Array], jax.Array]
        Function rendering image intensities from model parameters and
        coordinates.
    model_parameters: ParameterTree
        Frozen parameters of the reconstructed image.
    coordinates: jax.Array
        Flattened image coordinates with shape `(number_of_pixels, 2)`.
    deformation_grid: DeformationGrid
        Zero-centered deformation field.
    diagonal: bool
        Compute only the Fisher diagonal when true; otherwise compute the full
        parameter-by-parameter Fisher matrix.
    measurement_chunk_size: int
        Number of measurement gradients materialized together.
    show_progress: bool
        Whether to display chunk progress.

    Returns:
    --------
    np.ndarray
        Fisher diagonal with shape `(parameters,)` or full Fisher matrix with
        shape `(parameters, parameters)`.
    """
    if measurement_chunk_size <= 0:
        raise ValueError("`measurement_chunk_size` must be positive.")

    deformation_parameters = deformation_grid.init(
        jax.random.PRNGKey(0),
        coordinates,
    )["params"]
    number_of_parameters = deformation_parameters["node_offsets"].size
    if diagonal:
        fisher = jnp.zeros(number_of_parameters)
    else:
        fisher = jnp.zeros((number_of_parameters, number_of_parameters))

    def predict(
        current_deformation_parameters: ParameterTree,
        measurement_index: jax.Array,
    ) -> jax.Array:
        # First render the deformed image x_theta[p] = f(u_p + D_theta(u_p)).
        # Then return measurement m from y_theta = A x_theta, where A denotes
        # the problem's measurement operator.
        offsets = deformation_grid.apply(
            {"params": current_deformation_parameters},
            coordinates,
        )
        image = render_function(model_parameters, coordinates + offsets)
        prediction = forward_model.predict(
            jnp.asarray(image).reshape(-1),
            measurement_index,
        )
        return jnp.asarray(prediction).reshape(-1)[0]

    def real_prediction(
        current_deformation_parameters: ParameterTree,
        measurement_index: jax.Array,
    ) -> jax.Array:
        return jnp.real(
            predict(current_deformation_parameters, measurement_index)
        )

    def imaginary_prediction(
        current_deformation_parameters: ParameterTree,
        measurement_index: jax.Array,
    ) -> jax.Array:
        return jnp.imag(
            predict(current_deformation_parameters, measurement_index)
        )

    # Each vmapped gradient is one measurement-Jacobian row with shape
    # (grid_x, grid_y, 2).
    real_gradients = jax.jit(
        jax.vmap(jax.grad(real_prediction), in_axes=(None, 0))
    )
    imaginary_gradients = None
    if np.iscomplexobj(forward_model.observed_measurements):
        imaginary_gradients = jax.jit(
            jax.vmap(jax.grad(imaginary_prediction), in_axes=(None, 0))
        )

    # F = (1 / M) sum_m [
    #     (j_re_m.T @ j_re_m + j_im_m.T @ j_im_m) / sigma_m**2
    # ]. Real measurements omit imaginary contribution.
    #
    # note that the fisher information computed here uses a mean. The literal bayesian posterior
    # for M independent observations would be a joint likelihood so F = sum_m F_n. So the resultant
    # uncertainty map is a calibrated spatial uncertainty score rather than absolute physical posterior std.
    noise = jnp.asarray(forward_model.noise_standard_deviation)
    starts = range(0, forward_model.number_of_measurements, measurement_chunk_size)
    for start in tqdm(starts, desc="deformation Fisher", disable=not show_progress):
        stop = min(
            start + measurement_chunk_size,
            forward_model.number_of_measurements,
        )
        indices = jnp.arange(start, stop)

        # Reshape B gradient trees to J_chunk with shape (B, P), then whiten
        # row b by its measurement standard deviation sigma_b.
        chunk_noise = noise[indices, None]
        real_chunk = real_gradients(
            deformation_parameters,
            indices,
        )["node_offsets"].reshape(indices.size, -1)
        fisher = _accumulate_fisher(fisher, real_chunk / chunk_noise, diagonal)

        if imaginary_gradients is not None:
            imaginary_chunk = imaginary_gradients(
                deformation_parameters,
                indices,
            )["node_offsets"].reshape(indices.size, -1)
            fisher = _accumulate_fisher(fisher, imaginary_chunk / chunk_noise, diagonal)

    return np.asarray(fisher / forward_model.number_of_measurements)


def _accumulate_fisher(
    fisher: jax.Array,
    jacobian_chunk: jax.Array,
    diagonal: bool,
) -> jax.Array:
    """Add one Jacobian chunk to diagonal or full Fisher data."""
    if diagonal:
        # diag(J_chunk.T @ J_chunk)[k] = sum_b J_chunk[b, k]**2.
        return fisher + jnp.sum(jacobian_chunk**2, axis=0)
    return fisher + jacobian_chunk.T @ jacobian_chunk


def deformation_uncertainty_map(
    coordinates: jax.Array,
    covariance: np.ndarray,
    grid_resolution: tuple[int, int],
    image_shape: tuple[int, int],
    *,
    chunk_size: int = 8_192,
) -> np.ndarray:
    """
    Propagate deformation covariance to image-coordinate uncertainty.

    Parameters:
    -----------
    coordinates: jax.Array
        Flattened image coordinates with shape `(number_of_pixels, 2)`.
    covariance: np.ndarray
        Posterior variance vector or full parameter covariance matrix.
    grid_resolution: tuple[int, int]
        Number of deformation nodes along both coordinate dimensions.
    image_shape: tuple[int, int]
        Shape of the returned uncertainty map.
    chunk_size: int
        Number of coordinates processed together.

    Returns:
    --------
    np.ndarray
        Scalar displacement standard deviation at each image coordinate.
    """
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    coordinate_array = np.asarray(coordinates).reshape(-1, 2)
    if coordinate_array.shape[0] != int(np.prod(image_shape)):
        raise ValueError(
            "Coordinate count must equal the number of image pixels."
        )

    parameter_count = 2 * int(np.prod(grid_resolution))
    covariance_array = np.asarray(covariance)
    is_diagonal = covariance_array.ndim == 1
    expected_shape = (
        (parameter_count,)
        if is_diagonal
        else (parameter_count, parameter_count)
    )
    if covariance_array.shape != expected_shape:
        raise ValueError(
            f"`covariance` has shape {covariance_array.shape}; expected "
            f"{expected_shape}."
        )

    # Compute U(u)^2 = trace(W(u) Sigma_local W(u).T) at every image coordinate
    uncertainty_chunks = []
    for start in range(0, coordinate_array.shape[0], chunk_size):
        chunk = coordinate_array[start : start + chunk_size]

        # For B image coordinates, gather the four neighboring node indices
        # and their interpolation weights. Both arrays have shape (B, 4).
        node_indices, weights = _bilinear_node_indices_and_weights(
            chunk,
            grid_resolution,
        )
        if is_diagonal:
            # The variance vector is interleaved by node as
            # [node_0_x, node_0_y, node_1_x, node_1_y, ...]. Reshaping to
            # (number_of_nodes, 2) groups each node's x and y variances.
            node_variances = covariance_array.reshape(-1, 2)
            corner_variances = node_variances[node_indices] # (B, 4, 2)

            # U[b]^2 = sum_q sum_a weight[b,q]^2 Var(theta[q,a]).
            # Axis 1 sums the four corners q and Axis 2 sums components a = x,y.
            displacement_variance = np.sum(
                weights[..., None] ** 2 * corner_variances,
                axis=(1, 2),
            )
        else:
            # Convert four node indices into eight scalar offset components
            # [00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y].
            parameter_indices = np.stack(
                (2 * node_indices, 2 * node_indices + 1),
                axis=-1,
            ).reshape(chunk.shape[0], 8)

            # Pair every selected row index with every selected column index.
            # Broadcasting (B, 8, 1) with (B, 1, 8) creates the (B, 8, 8)
            # Cartesian product without mixing different batch coordinates.
            row_indices = parameter_indices[:, :, None]
            column_indices = parameter_indices[:, None, :]
            local_covariance = covariance_array[
                row_indices,
                column_indices,
            ]
            # W maps eight local offset components to one x/y displacement.
            interpolation_matrix = np.zeros(
                (chunk.shape[0], 2, 8),
                dtype=covariance_array.dtype,
            )
            interpolation_matrix[:, 0, 0::2] = weights
            interpolation_matrix[:, 1, 1::2] = weights

            # C[b,a,c] = sum_i,j W[b,a,i] Sigma[b,i,j] W[b,c,j].
            # Batch b and output components a,c remain. Local offset axes i,j
            # are summed, producing B covariance matrices of shape (2, 2).
            displacement_covariance = np.einsum(
                "bai,bij,bcj->bac",
                interpolation_matrix,
                local_covariance,
                interpolation_matrix,
            )
            # U[b]^2 = trace(C[b]) = Var(D_x) + Var(D_y).
            displacement_variance = np.trace(
                displacement_covariance,
                axis1=-2,
                axis2=-1,
            )
        if not np.all(np.isfinite(displacement_variance)):
            raise FloatingPointError(
                "Deformation covariance produced non-finite variance."
            )
        uncertainty_chunks.append(np.sqrt(np.maximum(displacement_variance, 0.0)))
    return np.concatenate(uncertainty_chunks).reshape(image_shape)


def _bilinear_node_indices_and_weights(
    coordinates: np.ndarray,
    grid_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return `00`, `10`, `01`, `11` node indices and weights."""
    grid_size_x, grid_size_y = grid_resolution
    x = coordinates[:, 0] * (grid_size_x - 1)
    y = coordinates[:, 1] * (grid_size_y - 1)
    i0 = np.floor(x).astype(int)
    j0 = np.floor(y).astype(int)
    i1 = np.clip(i0 + 1, 0, grid_size_x - 1)
    j1 = np.clip(j0 + 1, 0, grid_size_y - 1)
    wx = x - i0
    wy = y - j0

    node_indices = np.stack(
        (
            i0 * grid_size_y + j0,
            i1 * grid_size_y + j0,
            i0 * grid_size_y + j1,
            i1 * grid_size_y + j1,
        ),
        axis=1,
    )
    weights = np.stack(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ),
        axis=1,
    )
    return node_indices, weights


@dataclass(frozen=True)
class DeformationLaplaceResult:
    """
    Parameters:
    -----------
    fisher_information: np.ndarray
        Fisher diagonal or full Fisher matrix before adding the prior.
    posterior_covariance: np.ndarray
        Posterior variance vector or full covariance matrix.
    node_variance: np.ndarray
        Marginal variance with shape `(grid_x, grid_y, 2)`.
    uncertainty_map: np.ndarray
        Covariance propagated to every image coordinate.
    diagonal: bool
        Whether diagonal covariance was used.
    """
    fisher_information: np.ndarray
    posterior_covariance: np.ndarray
    node_variance: np.ndarray
    uncertainty_map: np.ndarray
    diagonal: bool


def deformation_laplace(
    forward_model: LinearInverseProblem,
    render_function: Callable[[ParameterTree, jax.Array], jax.Array],
    model_parameters: ParameterTree,
    coordinates: jax.Array,
    grid_resolution: int | tuple[int, int],
    *,
    prior_precision: float,
    diagonal: bool = True,
    measurement_chunk_size: int = 8,
    map_chunk_size: int = 8_192,
    show_progress: bool = True,
) -> DeformationLaplaceResult:
    """
    Compute batched deformation Fisher and spatial uncertainty.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Measurement operator, observations, and per-component noise.
    render_function: Callable[[ParameterTree, jax.Array], jax.Array]
        Function rendering image intensities from parameters and coordinates.
    model_parameters: ParameterTree
        Frozen reconstructed-image parameters.
    coordinates: jax.Array
        Flattened image coordinates with shape `(number_of_pixels, 2)`.
    grid_resolution: int | tuple[int, int]
        Square grid size or explicit two-dimensional resolution.
    prior_precision: float
        Scalar prior coefficient in `Fisher + 2 * prior_precision`.
    diagonal: bool
        Use the BayesRays diagonal approximation when true.
    measurement_chunk_size: int
        Measurement gradients materialized together.
    map_chunk_size: int
        Image coordinates used together for covariance propagation.
    show_progress: bool
        Whether to display Fisher chunk progress.

    Returns:
    --------
    DeformationLaplaceResult
        Fisher information, posterior covariance, node marginals, and map.
    """
    if prior_precision <= 0.0:
        raise ValueError("`prior_precision` must be positive.")
    resolution = (
        (grid_resolution, grid_resolution)
        if isinstance(grid_resolution, int)
        else grid_resolution
    )
    deformation_grid = DeformationGrid(resolution)
    fisher_information = deformation_fisher_information(
        forward_model,
        render_function,
        model_parameters,
        coordinates,
        deformation_grid,
        diagonal=diagonal,
        measurement_chunk_size=measurement_chunk_size,
        show_progress=show_progress,
    )
    parameter_count = 2 * int(np.prod(resolution))
    if diagonal:
        # Diagonal posterior precision and variance:
        # Sigma_k = 1 / F_k + 2*lambda
        posterior_covariance = 1.0 / (fisher_information + 2.0 * prior_precision)
        marginal_variance = posterior_covariance
    else:
        # Full posterior covariance: Sigma = (F + 2 lambda I)^-1.
        posterior_precision = fisher_information + (
            2.0 * prior_precision * np.eye(parameter_count)
        )
        try:
            cholesky_factor = np.linalg.cholesky(posterior_precision)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Posterior precision must be positive definite."
            ) from error
        identity = np.eye(
            parameter_count,
            dtype=posterior_precision.dtype,
        )
        # Two triangular solves recover the inverse without explicitly
        # inverting the posterior precision matrix.
        posterior_covariance = np.linalg.solve(
            cholesky_factor.T,
            np.linalg.solve(cholesky_factor, identity),
        )
        marginal_variance = np.diag(posterior_covariance)

    node_variance = marginal_variance.reshape(*resolution, 2)

    # Propagate the coarse parameter covariance through bilinear deformation
    # to obtain one displacement standard deviation per image coordinate.
    uncertainty_map = deformation_uncertainty_map(
        coordinates,
        posterior_covariance,
        resolution,
        forward_model.image_shape,
        chunk_size=map_chunk_size,
    )
    return DeformationLaplaceResult(
        fisher_information=fisher_information,
        posterior_covariance=posterior_covariance,
        node_variance=node_variance,
        uncertainty_map=uncertainty_map,
        diagonal=diagonal,
    )
