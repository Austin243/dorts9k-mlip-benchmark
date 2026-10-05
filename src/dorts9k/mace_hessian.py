"""Analytic Cartesian Hessians from pinned MACE checkpoints.

MACE defines its native Hessian as ``H = -dF/dR``.  The calculator currently
returns that tensor either flattened or with the last two position axes left
separate.  This adapter gives both forms one explicit benchmark convention:
``(3N, 3N)`` float64 in eV / Angstrom**2.

The MACE import is deliberately lazy so the reference-analysis package and its
unit tests do not require a heavyweight model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast

import numpy as np

from .model_hessian import (
    HessianEstimate,
    central_force_hessian,
    relative_frobenius_error,
)


Array = np.ndarray
MACEModelKind = Literal["mace_polar", "mace_mp"]
CalculatorLoader = Callable[..., Any]

ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/angstrom"
HESSIAN_UNIT = "eV/angstrom^2"


@dataclass(frozen=True)
class MACEProvenance:
    """Pinned model identity and runtime settings used by an adapter."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    model_kind: MACEModelKind
    head: str | None
    dtype: str
    device: str
    loader: str
    calculator_class: str
    runtime_versions: tuple[tuple[str, str], ...]
    hessian_execution: str = "native_single_structure_autograd"
    batch_numerical_semantics: str | None = None
    hessian_row_vmap_chunk_size: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for run manifests."""

        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "model_kind": self.model_kind,
            "head": self.head,
            "dtype": self.dtype,
            "device": self.device,
            "loader": self.loader,
            "calculator_class": self.calculator_class,
            "runtime_versions": dict(self.runtime_versions),
            "hessian_execution": self.hessian_execution,
            "batch_numerical_semantics": self.batch_numerical_semantics,
            "hessian_row_vmap_chunk_size": self.hessian_row_vmap_chunk_size,
            "energy_unit": ENERGY_UNIT,
            "force_unit": FORCE_UNIT,
            "hessian_unit": HESSIAN_UNIT,
        }


@dataclass(frozen=True)
class MACEFiniteDifferenceAudit:
    """Centered-force check against a native analytic MACE Hessian."""

    estimate: HessianEstimate
    relative_error: float


@dataclass(frozen=True)
class MACEHessianEvaluation:
    """Energy, forces, analytic Hessian, and an optional numerical audit."""

    energy_ev: float
    forces_ev_per_a: Array
    hessian_ev_per_a2: Array
    finite_difference_audit: MACEFiniteDifferenceAudit | None
    provenance: MACEProvenance


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without reading the whole model into memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_unit_factor(value: object, name: str) -> float:
    try:
        factor = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return factor


def normalize_mace_hessian(
    value: object,
    natoms: int,
    *,
    energy_units_to_ev: float = 1.0,
    length_units_to_a: float = 1.0,
) -> Array:
    """Normalize a native MACE Hessian to ``(3N, 3N)`` float64 eV/A^2.

    ``MACECalculator.get_hessian`` bypasses ASE's usual result conversion, so
    the calculator's energy and length factors are applied explicitly here.
    """

    if isinstance(natoms, bool) or not isinstance(natoms, (int, np.integer)):
        raise TypeError("natoms must be an integer")
    if natoms < 1:
        raise ValueError("natoms must be positive")
    dimension = 3 * int(natoms)
    hessian = np.asarray(value)
    accepted_shapes = ((dimension, dimension), (dimension, int(natoms), 3))
    if hessian.shape not in accepted_shapes:
        raise ValueError(
            "native MACE Hessian must have shape "
            f"{accepted_shapes[0]} or {accepted_shapes[1]}, got {hessian.shape}"
        )
    try:
        normalized = hessian.reshape(dimension, dimension).astype(
            np.float64, copy=True
        )
    except (TypeError, ValueError) as error:
        raise ValueError("native MACE Hessian must be numeric") from error
    if not np.all(np.isfinite(normalized)):
        raise ValueError("native MACE Hessian contains non-finite values")

    energy_factor = _positive_unit_factor(
        energy_units_to_ev, "energy_units_to_ev"
    )
    length_factor = _positive_unit_factor(length_units_to_a, "length_units_to_a")
    normalized *= energy_factor / (length_factor * length_factor)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("converted MACE Hessian contains non-finite values")
    return normalized


def _runtime_versions() -> tuple[tuple[str, str], ...]:
    packages = ("mace-torch", "torch", "ase", "numpy")
    observed: list[tuple[str, str]] = []
    for package in packages:
        try:
            observed.append((package, version(package)))
        except PackageNotFoundError:
            continue
    return tuple(observed)


def _validate_model_options(
    model_kind: str,
    head: str | None,
    dtype: str,
    device: str,
) -> tuple[MACEModelKind, str | None]:
    if model_kind not in ("mace_polar", "mace_mp"):
        raise ValueError("model_kind must be 'mace_polar' or 'mace_mp'")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be an explicit non-empty string")
    if model_kind == "mace_polar":
        if head is not None:
            raise ValueError("mace_polar does not accept a task head")
        normalized_head = None
    else:
        if not isinstance(head, str) or not head.strip():
            raise ValueError("mace_mp requires an explicit non-empty head")
        normalized_head = head.strip()
    return cast(MACEModelKind, model_kind), normalized_head


def _resolve_checkpoint(
    checkpoint: str | Path, expected_sha256: str | None
) -> tuple[Path, str]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MACE checkpoint is not a file: {path}")
    observed = checkpoint_sha256(path)
    if expected_sha256 is not None:
        expected = expected_sha256.lower()
        invalid_character = any(
            character not in "0123456789abcdef" for character in expected
        )
        if len(expected) != 64 or invalid_character:
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        if observed != expected:
            raise RuntimeError(
                "MACE checkpoint hash mismatch: "
                f"expected={expected} observed={observed}"
            )
    return path, observed


def _default_loader(model_kind: MACEModelKind) -> CalculatorLoader:
    try:
        from mace.calculators import mace_mp, mace_polar
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "MACE is required to load a checkpoint; install the pinned MACE runtime"
        ) from error
    return mace_polar if model_kind == "mace_polar" else mace_mp


def _loader_name(loader: CalculatorLoader) -> str:
    module = getattr(loader, "__module__", type(loader).__module__)
    name = getattr(loader, "__qualname__", type(loader).__qualname__)
    return f"{module}.{name}"


class MACEHessianAdapter:
    """One pinned MACE calculator under the benchmark Hessian convention."""

    def __init__(
        self,
        calculator: Any,
        *,
        model_kind: MACEModelKind,
        provenance: MACEProvenance,
    ) -> None:
        if model_kind not in ("mace_polar", "mace_mp"):
            raise ValueError("model_kind must be 'mace_polar' or 'mace_mp'")
        for method_name in ("get_potential_energy", "get_forces", "get_hessian"):
            if not callable(getattr(calculator, method_name, None)):
                raise TypeError(f"calculator lacks callable {method_name}()")
        self.calculator = calculator
        self.model_kind = model_kind
        self.provenance = provenance
        self.energy_units_to_ev = _positive_unit_factor(
            getattr(calculator, "energy_units_to_eV", 1.0),
            "calculator.energy_units_to_eV",
        )
        self.length_units_to_a = _positive_unit_factor(
            getattr(calculator, "length_units_to_A", 1.0),
            "calculator.length_units_to_A",
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        model_kind: MACEModelKind,
        head: str | None = None,
        dtype: str,
        device: str,
        expected_sha256: str | None = None,
        loader: CalculatorLoader | None = None,
        loader_kwargs: Mapping[str, object] | None = None,
    ) -> "MACEHessianAdapter":
        """Load an explicitly selected MACE model and record its identity."""

        model_kind, head = _validate_model_options(model_kind, head, dtype, device)
        path, observed_sha256 = _resolve_checkpoint(checkpoint, expected_sha256)
        selected_loader = loader if loader is not None else _default_loader(model_kind)
        kwargs: dict[str, object] = {
            "model": str(path),
            "device": device,
            "default_dtype": dtype,
        }
        if model_kind == "mace_mp":
            kwargs["head"] = head
        if loader_kwargs:
            overlap = set(kwargs).intersection(loader_kwargs)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"loader_kwargs cannot override pinned options: {names}")
            kwargs.update(loader_kwargs)
        calculator = selected_loader(**kwargs)
        if model_kind == "mace_mp" and getattr(calculator, "head", None) != head:
            raise RuntimeError(
                f"MACE selected head {getattr(calculator, 'head', None)!r} "
                f"instead of requested head {head!r}; "
                f"available heads: {getattr(calculator, 'available_heads', None)!r}"
            )
        provenance = MACEProvenance(
            checkpoint_path=str(path),
            checkpoint_sha256=observed_sha256,
            checkpoint_size_bytes=path.stat().st_size,
            model_kind=model_kind,
            head=head,
            dtype=dtype,
            device=device,
            loader=_loader_name(selected_loader),
            calculator_class=(
                f"{type(calculator).__module__}.{type(calculator).__qualname__}"
            ),
            runtime_versions=_runtime_versions(),
            hessian_execution="batched_equal_natoms_reverse_rows",
            batch_numerical_semantics=(
                "independent molecular graphs; one reverse call per local "
                "Cartesian force component"
            ),
            hessian_row_vmap_chunk_size=None,
        )
        return cls(
            calculator,
            model_kind=model_kind,
            provenance=provenance,
        )

    @staticmethod
    def _validated_positions(atoms: Any) -> Array:
        try:
            natoms = len(atoms)
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("atoms must provide len() and get_positions()") from error
        if natoms < 1 or positions.shape != (natoms, 3):
            raise ValueError(
                f"atom positions must have shape ({natoms}, 3), got {positions.shape}"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("atom positions contain non-finite values")
        return positions

    def _prepare_atoms(self, atoms: Any) -> Any:
        self._validated_positions(atoms)
        try:
            prepared = atoms.copy()
        except AttributeError as error:
            raise TypeError("atoms must provide copy()") from error
        if self.model_kind != "mace_polar":
            return prepared

        info = getattr(prepared, "info", None)
        if not isinstance(info, dict):
            raise TypeError("atoms.info must be a dictionary for MACE-POLAR")
        for key, required in (("charge", 0.0), ("spin", 1.0)):
            if key in info:
                try:
                    compatible = bool(np.isclose(float(info[key]), required))
                except (TypeError, ValueError):
                    compatible = False
                if not compatible:
                    meaning = "neutral" if key == "charge" else "singlet"
                    raise ValueError(
                        f"MACE-POLAR benchmark requires {meaning} {key}={required:g}"
                    )
        if "external_field" in info:
            try:
                field = np.asarray(info["external_field"], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("external_field must be a finite zero 3-vector") from error
            if (
                field.shape != (3,)
                or not np.all(np.isfinite(field))
                or not np.allclose(field, 0.0)
            ):
                raise ValueError("MACE-POLAR benchmark requires external_field=[0, 0, 0]")
        info["charge"] = 0
        info["spin"] = 1
        info["external_field"] = [0.0, 0.0, 0.0]
        return prepared

    @staticmethod
    def _validate_energy(value: object) -> float:
        energy = np.asarray(value, dtype=np.float64)
        if energy.size != 1:
            raise ValueError(f"energy must be scalar, got shape {energy.shape}")
        scalar = float(energy.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError("energy is non-finite")
        return scalar

    @staticmethod
    def _validate_forces(value: object, natoms: int) -> Array:
        try:
            forces = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("forces must be numeric") from error
        if forces.shape != (natoms, 3):
            raise ValueError(
                f"forces must have shape ({natoms}, 3), got {forces.shape}"
            )
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces contain non-finite values")
        return forces.copy()

    def _energy_prepared(self, atoms: Any) -> float:
        return self._validate_energy(self.calculator.get_potential_energy(atoms))

    def _forces_prepared(self, atoms: Any) -> Array:
        return self._validate_forces(self.calculator.get_forces(atoms), len(atoms))

    def _hessian_prepared(self, atoms: Any) -> Array:
        native = self.calculator.get_hessian(atoms)
        return normalize_mace_hessian(
            native,
            len(atoms),
            energy_units_to_ev=self.energy_units_to_ev,
            length_units_to_a=self.length_units_to_a,
        )

    def get_energy(self, atoms: Any) -> float:
        """Return potential energy in eV."""

        return self._energy_prepared(self._prepare_atoms(atoms))

    def get_forces(self, atoms: Any) -> Array:
        """Return Cartesian forces with shape ``(N, 3)`` in eV/A."""

        return self._forces_prepared(self._prepare_atoms(atoms))

    def get_hessian(self, atoms: Any) -> Array:
        """Call native ``calculator.get_hessian`` and return eV/A^2."""

        return self._hessian_prepared(self._prepare_atoms(atoms))

    def finite_difference_audit(
        self,
        atoms: Any,
        *,
        step_a: float = 1.0e-3,
        native_hessian: Array | None = None,
    ) -> MACEFiniteDifferenceAudit:
        """Check the native Hessian with centered differences of MACE forces."""

        prepared = self._prepare_atoms(atoms)
        positions = self._validated_positions(prepared)
        if native_hessian is None:
            native = self._hessian_prepared(prepared)
        else:
            native = np.asarray(native_hessian, dtype=np.float64)
            expected_shape = (positions.size, positions.size)
            if native.shape != expected_shape:
                raise ValueError(
                    f"normalized native Hessian must have shape {expected_shape}, "
                    f"got {native.shape}"
                )
            if not np.all(np.isfinite(native)):
                raise ValueError("normalized native Hessian contains non-finite values")

        def force_function(displaced_positions: Array) -> Array:
            probe = prepared.copy()
            probe.set_positions(displaced_positions)
            return self._forces_prepared(probe)

        estimate = central_force_hessian(
            positions,
            force_function,
            step_a=step_a,
        )
        return MACEFiniteDifferenceAudit(
            estimate=estimate,
            relative_error=relative_frobenius_error(estimate.raw, native),
        )

    def evaluate(
        self,
        atoms: Any,
        *,
        finite_difference_step_a: float | None = None,
    ) -> MACEHessianEvaluation:
        """Evaluate all native outputs and, optionally, a centered-force audit."""

        prepared = self._prepare_atoms(atoms)
        energy = self._energy_prepared(prepared)
        forces = self._forces_prepared(prepared)
        hessian = self._hessian_prepared(prepared)
        audit = None
        if finite_difference_step_a is not None:
            audit = self.finite_difference_audit(
                prepared,
                step_a=finite_difference_step_a,
                native_hessian=hessian,
            )
        return MACEHessianEvaluation(
            energy_ev=energy,
            forces_ev_per_a=forces,
            hessian_ev_per_a2=hessian,
            finite_difference_audit=audit,
            provenance=self.provenance,
        )

    def _graph_batch(self, atoms_batch: list[Any]) -> Any:
        """Build one PyG batch using the pinned calculator's graph settings."""

        if not atoms_batch:
            raise ValueError("MACE batch must contain at least one structure")
        natoms = len(atoms_batch[0])
        if any(len(atoms) != natoms for atoms in atoms_batch):
            raise ValueError("MACE Hessian batches must contain equal-size structures")
        try:
            from mace import data as mace_data
            from mace.tools import torch_geometric
            from mace.tools import torch_tools
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ImportError("MACE's graph batching runtime is unavailable") from error

        calculator = self.calculator
        required_attributes = (
            "arrays_keys",
            "charges_key",
            "info_keys",
            "default_dtype",
            "head",
            "z_table",
            "r_max",
            "available_heads",
        )
        missing = [name for name in required_attributes if not hasattr(calculator, name)]
        if missing:
            raise TypeError(
                "calculator lacks MACE graph settings: " + ", ".join(missing)
            )
        if bool(getattr(calculator, "use_compile", False)):
            raise TypeError("optimized Hessian batching requires uncompiled MACE")

        calculator.arrays_keys.update({calculator.charges_key: "charges"})
        key_specification = mace_data.KeySpecification(
            info_keys=calculator.info_keys,
            arrays_keys=calculator.arrays_keys,
        )
        graphs = []
        with torch_tools.default_dtype(calculator.default_dtype):
            for atoms in atoms_batch:
                config = mace_data.config_from_atoms(
                    self._prepare_atoms(atoms),
                    key_specification=key_specification,
                    head_name=calculator.head,
                )
                graphs.append(
                    mace_data.AtomicData.from_config(
                        config,
                        z_table=calculator.z_table,
                        cutoff=calculator.r_max,
                        heads=calculator.available_heads,
                    )
                )
        device = getattr(calculator, "device", self.provenance.device)
        return torch_geometric.Batch.from_data_list(graphs).to(device)

    def _predict_graph_batch(
        self, atoms_batch: list[Any]
    ) -> tuple[object, object, object]:
        """Run one model graph and recover every diagonal molecular Hessian block."""

        import torch

        models = getattr(self.calculator, "models", None)
        if models is None or len(models) != 1:
            raise TypeError(
                "optimized MACE batching requires a single-model MACECalculator"
            )
        batch = self._graph_batch(atoms_batch)
        model = models[0]
        model_dtype = next(model.parameters()).dtype
        keys = batch.keys if not callable(batch.keys) else batch.keys()
        for key in keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model_dtype)

        batch_dict = batch.to_dict()
        with torch.enable_grad():
            output = model(
                batch_dict,
                compute_hessian=False,
                compute_stress=False,
                training=True,
            )
        energy = output.get("energy")
        forces = output.get("forces")
        if energy is None or forces is None:
            raise RuntimeError("MACE batch prediction did not return energy and forces")

        batch_size = len(atoms_batch)
        natoms = len(atoms_batch[0])
        dimension = 3 * natoms
        if energy.numel() != batch_size:
            raise ValueError(
                f"MACE energy batch has {energy.numel()} entries; expected {batch_size}"
            )
        if tuple(forces.shape) != (batch_size * natoms, 3):
            raise ValueError(
                f"MACE force batch has shape {tuple(forces.shape)}; "
                f"expected ({batch_size * natoms}, 3)"
            )

        positions = batch_dict["positions"]
        force_rows = forces.reshape(batch_size, dimension)
        hessian_rows = []
        for index in range(dimension):
            derivative = torch.autograd.grad(
                force_rows[:, index].sum(),
                positions,
                retain_graph=index + 1 < dimension,
                create_graph=False,
                allow_unused=False,
            )[0]
            hessian_rows.append(
                (-derivative).detach().reshape(batch_size, dimension)
            )
        hessians = torch.stack(hessian_rows, dim=1)
        return energy.detach(), forces.detach().reshape(batch_size, natoms, 3), hessians

    def evaluate_energy_forces_batch(
        self, atoms_batch: list[Any]
    ) -> tuple[np.ndarray, list[Array]]:
        """Evaluate energies and forces without constructing any Hessian graph."""

        import torch

        models = getattr(self.calculator, "models", None)
        if models is None or len(models) != 1:
            raise TypeError(
                "optimized MACE batching requires a single-model MACECalculator"
            )
        batch = self._graph_batch(atoms_batch)
        model = models[0]
        model_dtype = next(model.parameters()).dtype
        keys = batch.keys if not callable(batch.keys) else batch.keys()
        for key in keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model_dtype)

        with torch.enable_grad():
            output = model(
                batch.to_dict(),
                compute_hessian=False,
                compute_stress=False,
                training=False,
            )
        energy = output.get("energy")
        forces = output.get("forces")
        if energy is None or forces is None:
            raise RuntimeError("MACE batch prediction did not return energy and forces")

        batch_size = len(atoms_batch)
        natoms = len(atoms_batch[0])
        if energy.numel() != batch_size:
            raise ValueError(
                f"MACE energy batch has {energy.numel()} entries; expected {batch_size}"
            )
        if tuple(forces.shape) != (batch_size * natoms, 3):
            raise ValueError(
                f"MACE force batch has shape {tuple(forces.shape)}; "
                f"expected ({batch_size * natoms}, 3)"
            )

        energy_values = (
            energy.detach().cpu().numpy().reshape(-1) * self.energy_units_to_ev
        )
        force_values = (
            forces.detach().cpu().numpy().reshape(batch_size, natoms, 3)
            * self.energy_units_to_ev
            / self.length_units_to_a
        )
        validated_energies = np.asarray(
            [self._validate_energy(value) for value in energy_values],
            dtype=np.float64,
        )
        validated_forces = [
            self._validate_forces(force_values[index], len(atoms))
            for index, atoms in enumerate(atoms_batch)
        ]
        return validated_energies, validated_forces

    def evaluate_batch(self, atoms_batch: list[Any]) -> list[MACEHessianEvaluation]:
        """Evaluate equal-size structures with only ``3N`` reverse-mode calls.

        MACE concatenates independent molecular graphs. Summing one matching
        force component over those graphs recovers the corresponding Hessian
        row for every molecule without forming zero cross-molecule blocks.
        """

        energy, forces, hessians = self._predict_graph_batch(atoms_batch)
        energy_values = (
            energy.cpu().numpy().reshape(-1) * self.energy_units_to_ev
        )
        force_values = (
            forces.cpu().numpy()
            * self.energy_units_to_ev
            / self.length_units_to_a
        )
        hessian_values = (
            hessians.cpu().numpy()
            * self.energy_units_to_ev
            / (self.length_units_to_a * self.length_units_to_a)
        )

        evaluations: list[MACEHessianEvaluation] = []
        for index, atoms in enumerate(atoms_batch):
            evaluations.append(
                MACEHessianEvaluation(
                    energy_ev=self._validate_energy(energy_values[index]),
                    forces_ev_per_a=self._validate_forces(
                        force_values[index], len(atoms)
                    ),
                    hessian_ev_per_a2=normalize_mace_hessian(
                        hessian_values[index], len(atoms)
                    ),
                    finite_difference_audit=None,
                    provenance=self.provenance,
                )
            )
        return evaluations
