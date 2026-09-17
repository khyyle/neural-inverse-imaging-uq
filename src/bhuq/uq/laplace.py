"""Laplax adapters for parameter-space uncertainty in neural images."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from laplax import laplace
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_var,
    lin_setup,
    set_lin_pushforward,
)

from ..forward import LinearInverseProblem

type ParameterTree = Any
type LaplaceCurvature = Literal["full", "lanczos"]

FLOAT32_BYTES = np.dtype(np.float32).itemsize
DEFAULT_DENSE_MEMORY_WARNING_BYTES = 16 * 1024**3
DENSE_WORKING_ARRAY_COUNT = 3


@dataclass(frozen=True)
class LinearizedLaplaceResult:
    """
    Store image uncertainty from a Laplax parameter posterior.

    Parameters:
    -----------
    pixel_standard_deviation: np.ndarray
        Linearized marginal image standard deviation.
    prior_precision: float
        Isotropic Gaussian parameter-prior precision.
    curvature: LaplaceCurvature
        Laplax curvature representation.
    parameter_count: int
        Number of uncertain scalar parameters.
    rank: int | None
        Lanczos rank, or `None` for full dense curvature.
    """

    pixel_standard_deviation: np.ndarray
    prior_precision: float
    curvature: LaplaceCurvature
    parameter_count: int
    rank: int | None


def linearized_parameter_laplace(
    forward_model: LinearInverseProblem,
    render_function: Callable[[ParameterTree, jax.Array], jax.Array],
    model_parameters: ParameterTree,
    coordinates: jax.Array,
    *,
    prior_precision: float,
    curvature: LaplaceCurvature = "lanczos",
    rank: int = 50,
    random_seed: int = 0,
    dense_memory_warning_bytes: int = DEFAULT_DENSE_MEMORY_WARNING_BYTES,
) -> LinearizedLaplaceResult:
    """
    Fit a Laplax GGN posterior and propagate it to image coordinates.

    The model parameters may be the complete network PyTree or a selected
    parameter subtree. When a subtree is supplied, `render_function` must merge
    it with the frozen parameters before rendering.

    Parameters:
    -----------
    forward_model: LinearInverseProblem
        Measurement operator, observations, and per-measurement noise.
    render_function: Callable[[ParameterTree, jax.Array], jax.Array]
        Function receiving `(parameters, coordinates)` and returning image
        intensities.
    model_parameters: ParameterTree
        MAP parameters represented as a JAX PyTree.
    coordinates: jax.Array
        Flattened image coordinates with shape `(number_of_pixels, dimensions)`.
    prior_precision: float
        Positive isotropic Gaussian prior precision.
    curvature: LaplaceCurvature
        `full` materializes the complete GGN matrix. `lanczos` estimates its
        dominant eigenspace.
    rank: int
        Number of retained Lanczos eigenpairs.
    random_seed: int
        Seed used to initialize the Lanczos iteration.
    dense_memory_warning_bytes: int
        Estimated dense-curvature working memory that emits a warning.

    Returns:
    --------
    LinearizedLaplaceResult
        Pixel standard deviation and posterior configuration.

    Raises:
    -------
    ValueError
        If inputs are invalid.
    """
    if prior_precision <= 0.0:
        raise ValueError("`prior_precision` must be positive.")
    if curvature not in ("full", "lanczos"):
        raise ValueError("`curvature` must be `full` or `lanczos`.")
    if dense_memory_warning_bytes <= 0:
        raise ValueError("`dense_memory_warning_bytes` must be positive.")

    coordinate_array = jnp.asarray(coordinates, dtype=jnp.float32)
    if coordinate_array.ndim != 2:
        raise ValueError("`coordinates` must be a two-dimensional array.")
    if coordinate_array.shape[0] != forward_model.number_of_pixels:
        raise ValueError(
            "Coordinate count must equal the forward model's pixel count."
        )

    parameters = jax.tree.map(_to_float32, model_parameters)
    parameter_count = sum(
        int(parameter.size) for parameter in jax.tree.leaves(parameters)
    )
    if parameter_count == 0:
        raise ValueError("`model_parameters` must not be empty.")

    laplax_arguments: dict[str, Any] = {}
    selected_rank = None
    if curvature == "full":
        estimated_working_memory = (
            DENSE_WORKING_ARRAY_COUNT
            * parameter_count**2
            * FLOAT32_BYTES
        )
        if estimated_working_memory >= dense_memory_warning_bytes:
            gibibytes = estimated_working_memory / 1024**3
            warnings.warn(
                "Full curvature is estimated to require "
                f"{gibibytes:.2f} GiB of working memory. Consider using "
                "`lanczos` if this exceeds available memory.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        if rank <= 0:
            raise ValueError("`rank` must be positive.")
        if rank >= parameter_count:
            raise ValueError(
                "`rank` must be smaller than the parameter count."
            )
        selected_rank = rank
        laplax_arguments = {
            "rank": rank,
            "key": jax.random.key(random_seed),
            "mv_dtype": jnp.float32,
            "calc_dtype": jnp.float32,
            "return_dtype": jnp.float32,
        }

    noise = jnp.asarray(
        forward_model.noise_standard_deviation,
        dtype=jnp.float32,
    )
    observations = jnp.asarray(forward_model.observed_measurements)
    target = _stack_whitened_measurements(observations, noise)

    def measurement_model(**arguments: Any) -> jax.Array:
        params = arguments["params"]
        image = render_function(params, coordinate_array)
        prediction = forward_model.predict(jnp.asarray(image).reshape(-1))
        return _stack_whitened_measurements(prediction, noise)

    def gaussian_data_term(
        prediction: jax.Array,
        measurement_target: jax.Array,
    ) -> jax.Array:
        residual = prediction - measurement_target
        return 0.5 * jnp.sum(residual**2)

    # Data constrain parameters through predicted measurements rather than
    # image pixels, so curvature uses the complete parameter-to-measurement map.
    # Real and imaginary components (if applicable) are considered independent observations.
    posterior_function, _curvature_estimate = laplace(
        model_fn=measurement_model,
        params=parameters,
        data={
            # Laplax requires a supervised input-target pair, but this global
            # inverse problem has no input separate from its fixed operator.
            # so we pass a dummy
            "input": jnp.asarray(0.0, dtype=jnp.float32),
            "target": target,
        },
        loss_fn=gaussian_data_term,
        curv_type=curvature,
        vmap_over_data=False,
        curv_mv_jit=True,
        **laplax_arguments,
    )

    def image_model(**arguments: Any) -> jax.Array:
        return jnp.asarray(
            render_function(arguments["params"], arguments["input"])
        ).reshape(1)

    # The resulting function computes variance at arbitrary image coordinates.
    #
    # Since x_(theta* + delta)(u) is approximately
    # x_theta*(u) + J_x(u) delta, Var[x(u)] is approximately
    # J_x(u) Sigma_theta J_x(u).T.
    predict_at_coordinate = set_lin_pushforward(
        model_fn=image_model,
        mean_params=parameters,
        posterior_fn=posterior_function,
        prior_arguments={"prior_prec": prior_precision},
        pushforward_fns=[lin_setup, lin_pred_mean, lin_pred_var],
    )
    predictive_values = jax.vmap(predict_at_coordinate)(coordinate_array)
    pixel_variance = np.asarray(
        predictive_values["pred_var"],
        dtype=np.float32,
    ).reshape(-1)
    if not np.all(np.isfinite(pixel_variance)):
        raise FloatingPointError(
            "Laplax produced non-finite pixel variances."
        )
    pixel_standard_deviation = np.sqrt(
        np.maximum(pixel_variance, 0.0)
    ).reshape(forward_model.image_shape)

    return LinearizedLaplaceResult(
        pixel_standard_deviation=pixel_standard_deviation,
        prior_precision=prior_precision,
        curvature=curvature,
        parameter_count=parameter_count,
        rank=selected_rank,
    )


def _stack_whitened_measurements(
    measurements: jax.Array,
    noise_standard_deviation: jax.Array,
) -> jax.Array:
    """Represent whitened real or complex measurements as a real vector."""
    whitened = jnp.asarray(measurements).reshape(-1) / noise_standard_deviation
    if jnp.iscomplexobj(whitened):
        return jnp.concatenate(
            (jnp.real(whitened), jnp.imag(whitened))
        ).astype(jnp.float32)
    return whitened.astype(jnp.float32)


def _to_float32(value: Any) -> jax.Array:
    """Convert inexact parameter leaves to float32."""
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return array.astype(jnp.complex64)
    if jnp.issubdtype(array.dtype, jnp.floating):
        return array.astype(jnp.float32)
    return array
