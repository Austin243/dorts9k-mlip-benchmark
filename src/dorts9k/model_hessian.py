"""Model Hessians from force derivatives.

The common benchmark definition is ``H = -dF/dR`` in Cartesian coordinates.
Keeping this numerical layer independent of any MLIP package makes the same
finite-difference audit available for conservative and direct-force models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
ForceFunction = Callable[[Array], Array]
BatchForceFunction = Callable[[Sequence[Array]], Sequence[Array]]


@dataclass(frozen=True)
class HessianEstimate:
    """A Cartesian force-derivative Hessian and its integrability audit."""

    raw: Array
    symmetric: Array
    antisymmetric: Array
    relative_antisymmetry: float
    step_a: float
    force_calls: int


def _as_positions(value: Array) -> Array:
    positions = np.asarray(value, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain non-finite values")
    return positions


def _as_force_vector(value: Array, shape: tuple[int, int]) -> Array:
    forces = np.asarray(value, dtype=np.float64)
    if forces.shape != shape:
        raise ValueError(f"forces must have shape {shape}, got {forces.shape}")
    if not np.all(np.isfinite(forces)):
        raise ValueError("forces contain non-finite values")
    return forces.reshape(-1)


def displaced_position_pairs(positions: Array, step_a: float) -> list[Array]:
    """Return ``[-h,+h]`` pairs in deterministic Cartesian-coordinate order."""

    positions = _as_positions(positions)
    if not np.isfinite(step_a) or step_a <= 0:
        raise ValueError("step_a must be positive and finite")
    flat = positions.reshape(-1)
    displaced: list[Array] = []
    for coordinate in range(flat.size):
        minus = flat.copy()
        plus = flat.copy()
        minus[coordinate] -= step_a
        plus[coordinate] += step_a
        displaced.extend((minus.reshape(positions.shape), plus.reshape(positions.shape)))
    return displaced


def _finish_hessian(raw: Array, step_a: float, force_calls: int) -> HessianEstimate:
    raw = np.asarray(raw, dtype=np.float64)
    symmetric = 0.5 * (raw + raw.T)
    antisymmetric = 0.5 * (raw - raw.T)
    denominator = float(np.linalg.norm(symmetric, ord="fro"))
    # Report the same convention as ``modes.hessian_symmetry_metrics``:
    # ||H-H.T||_F / ||(H+H.T)/2||_F.  Keep the half-difference matrix itself
    # because it is the conventional antisymmetric component.
    numerator = float(np.linalg.norm(raw - raw.T, ord="fro"))
    relative = numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else np.inf)
    return HessianEstimate(
        raw=raw,
        symmetric=symmetric,
        antisymmetric=antisymmetric,
        relative_antisymmetry=relative,
        step_a=float(step_a),
        force_calls=int(force_calls),
    )


def central_force_hessian(
    positions: Array,
    force_function: ForceFunction,
    *,
    step_a: float = 1.0e-3,
) -> HessianEstimate:
    """Estimate ``H = -dF/dR`` with a second-order centered stencil."""

    positions = _as_positions(positions)
    displaced = displaced_position_pairs(positions, step_a)
    dimension = positions.size
    raw = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        f_minus = _as_force_vector(force_function(displaced[2 * coordinate]), positions.shape)
        f_plus = _as_force_vector(force_function(displaced[2 * coordinate + 1]), positions.shape)
        raw[:, coordinate] = -(f_plus - f_minus) / (2.0 * step_a)
    return _finish_hessian(raw, step_a, 2 * dimension)


def batched_central_force_hessian(
    positions: Array,
    batch_force_function: BatchForceFunction,
    *,
    step_a: float = 1.0e-3,
    batch_size: int = 64,
) -> HessianEstimate:
    """Centered force Hessian with displaced geometries evaluated in batches."""

    positions = _as_positions(positions)
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    displaced = displaced_position_pairs(positions, step_a)
    forces: list[Array] = []
    for start in range(0, len(displaced), batch_size):
        geometry_batch = displaced[start : start + batch_size]
        output_batch = list(batch_force_function(geometry_batch))
        if len(output_batch) != len(geometry_batch):
            raise ValueError(
                "batch force output cardinality mismatch: "
                f"expected {len(geometry_batch)}, got {len(output_batch)}"
            )
        forces.extend(
            _as_force_vector(force, positions.shape) for force in output_batch
        )

    dimension = positions.size
    raw = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        f_minus = forces[2 * coordinate]
        f_plus = forces[2 * coordinate + 1]
        raw[:, coordinate] = -(f_plus - f_minus) / (2.0 * step_a)
    return _finish_hessian(raw, step_a, len(displaced))


def relative_frobenius_error(predicted: Array, reference: Array) -> float:
    """Return ``||pred-ref||_F / ||ref||_F`` with explicit zero handling."""

    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} != {reference.shape}")
    numerator = float(np.linalg.norm(predicted - reference, ord="fro"))
    denominator = float(np.linalg.norm(reference, ord="fro"))
    return numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else np.inf)
