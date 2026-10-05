"""Hessian adapters for the non-MACE models in the HORM benchmark.

All adapters expose the same convention used by :mod:`mace_hessian`:
Cartesian ``H = -dF/dR`` with shape ``(3N, 3N)`` in eV/angstrom**2.
Heavy model packages are imported lazily so reference-only analysis remains
dependency-light.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .model_hessian import (
    HessianEstimate,
    batched_central_force_hessian,
    central_force_hessian,
    relative_frobenius_error,
)


HARTREE_TO_EV = 27.211386245981


@dataclass(frozen=True)
class ModelProvenance:
    """JSON-ready identity and runtime settings for one model adapter."""

    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class FiniteDifferenceAudit:
    """Centered-force validation of an analytic Hessian."""

    estimate: HessianEstimate
    relative_error: float


@dataclass(frozen=True)
class ModelHessianEvaluation:
    """Common output consumed by the HORM shard runner."""

    energy_ev: float
    forces_ev_per_a: np.ndarray
    hessian_ev_per_a2: np.ndarray
    finite_difference_audit: FiniteDifferenceAudit | None
    provenance: ModelProvenance


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_file(path: str | Path, expected_sha256: str) -> tuple[Path, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = file_sha256(source)
    if observed != expected_sha256.lower():
        raise RuntimeError(
            f"checkpoint hash mismatch: expected={expected_sha256.lower()} observed={observed}"
        )
    return source, observed


def _versions(*packages: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def normalize_hessian(value: object, natoms: int) -> np.ndarray:
    """Normalize any size-compatible native Hessian to a finite matrix."""

    dimension = 3 * int(natoms)
    array = np.asarray(value)
    if array.size != dimension * dimension:
        raise ValueError(
            f"Hessian has {array.size} entries; expected {(dimension * dimension)}"
        )
    matrix = array.reshape(dimension, dimension).astype(np.float64, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Hessian contains non-finite values")
    return matrix


def _validate_outputs(
    energy: object,
    forces: object,
    hessian: object,
    natoms: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    energy_array = np.asarray(energy, dtype=np.float64)
    if energy_array.size != 1 or not np.all(np.isfinite(energy_array)):
        raise ValueError("energy must be one finite scalar")
    force_array = np.asarray(forces, dtype=np.float64)
    if force_array.shape != (natoms, 3) or not np.all(np.isfinite(force_array)):
        raise ValueError(f"forces must be finite with shape ({natoms}, 3)")
    return (
        float(energy_array.reshape(-1)[0]),
        force_array.copy(),
        normalize_hessian(hessian, natoms),
    )


class _BaseAdapter:
    provenance: ModelProvenance

    def get_forces(self, atoms: Any) -> np.ndarray:
        raise NotImplementedError

    def _evaluate_native(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def evaluate(
        self,
        atoms: Any,
        *,
        finite_difference_step_a: float | None = None,
    ) -> ModelHessianEvaluation:
        energy, forces, hessian = self._evaluate_native(atoms)
        audit = None
        if finite_difference_step_a is not None:
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)

            def force_function(displaced: np.ndarray) -> np.ndarray:
                probe = atoms.copy()
                probe.set_positions(displaced)
                return self.get_forces(probe)

            estimate = central_force_hessian(
                positions,
                force_function,
                step_a=finite_difference_step_a,
            )
            audit = FiniteDifferenceAudit(
                estimate=estimate,
                relative_error=relative_frobenius_error(estimate.raw, hessian),
            )
        return ModelHessianEvaluation(
            energy_ev=energy,
            forces_ev_per_a=forces,
            hessian_ev_per_a2=hessian,
            finite_difference_audit=audit,
            provenance=self.provenance,
        )


class UMAHessianAdapter(_BaseAdapter):
    """UMA Hessian from either native autograd or batched force differences."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint: str | Path,
        expected_sha256: str,
        task_name: str,
        charge: int,
        spin: int,
        device: str,
        torch_num_threads: int,
        inference_settings_name: str = "default",
        compile_model: bool = False,
        hessian_method: str = "force_fd",
        finite_difference_step_a: float = 1.0e-3,
        finite_difference_batch_size: int = 4,
        large_natoms_threshold: int | None = None,
        large_finite_difference_batch_size: int | None = None,
    ) -> None:
        if task_name not in {"omol", "omc"}:
            raise ValueError("UMA task_name must be 'omol' or 'omc'")
        required_spin = 1 if task_name == "omol" else 0
        if charge != 0 or spin != required_spin:
            raise ValueError(
                f"UMA {task_name} benchmark requires charge=0 spin={required_spin}"
            )
        if hessian_method not in {"native", "force_fd"}:
            raise ValueError("UMA hessian_method must be 'native' or 'force_fd'")
        if not np.isfinite(finite_difference_step_a) or finite_difference_step_a <= 0:
            raise ValueError("UMA finite-difference step must be positive and finite")
        if finite_difference_batch_size < 1:
            raise ValueError("UMA finite-difference batch size must be at least one")
        if (large_natoms_threshold is None) != (
            large_finite_difference_batch_size is None
        ):
            raise ValueError(
                "UMA large-molecule threshold and batch size must be set together"
            )
        if large_natoms_threshold is not None and large_natoms_threshold < 1:
            raise ValueError("UMA large-molecule atom threshold must be positive")
        if (
            large_finite_difference_batch_size is not None
            and large_finite_difference_batch_size < 1
        ):
            raise ValueError(
                "UMA large-molecule finite-difference batch size must be positive"
            )
        checkpoint_path, checkpoint_hash = _pinned_file(checkpoint, expected_sha256)

        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import NAME_TO_INFERENCE_SETTING

        settings_key = {
            "fast_precise": "turbo",
            "fast_precise_umas": "turbo_umas",
        }.get(inference_settings_name, inference_settings_name)
        settings = copy.deepcopy(NAME_TO_INFERENCE_SETTING[settings_key])
        settings.torch_num_threads = int(torch_num_threads)
        settings.compile = bool(compile_model)
        # UMA's stock turbo profile merges a molecular system into one graph and
        # rejects batches containing multiple systems. Hessian throughput depends
        # on evaluating many equal-size molecules/displacements together, so keep
        # graph merging disabled while retaining turbo's other fast-path settings.
        settings.merge_mole = False
        if inference_settings_name in {"fast_precise", "fast_precise_umas"}:
            # TF32 force noise is strongly amplified by the small finite-
            # difference displacement used for Hessians. Keep turbo's disabled
            # activation checkpointing, but retain full FP32 matmul precision.
            settings.tf32 = False
        settings.predict_untrained_hessian = (
            {task_name} if hessian_method == "native" else set()
        )
        settings.hessian_vmap = False
        actual_checkpoint = pretrained_mlip.pretrained_checkpoint_path_from_name(model_name)
        if file_sha256(actual_checkpoint) != checkpoint_hash:
            raise ValueError("UMA's named pretrained release does not match --checkpoint")
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            inference_settings=settings,
        )
        self.calculator = FAIRChemCalculator(predictor, task_name=task_name)
        self.task_name = task_name
        self.charge = int(charge)
        self.spin = int(spin)
        self.hessian_method = hessian_method
        self.finite_difference_step_a = float(finite_difference_step_a)
        self.finite_difference_batch_size = int(finite_difference_batch_size)
        self.large_natoms_threshold = large_natoms_threshold
        self.large_finite_difference_batch_size = (
            None
            if large_finite_difference_batch_size is None
            else int(large_finite_difference_batch_size)
        )
        self.provenance = ModelProvenance(
            {
                "family": "uma",
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "task_name": task_name,
                "charge": self.charge,
                "spin": self.spin,
                "spin_semantics": (
                    "spin_multiplicity" if task_name == "omol" else "omc_closed_shell_feature"
                ),
                "inference_settings": inference_settings_name,
                "compile": bool(settings.compile),
                "tf32": bool(settings.tf32),
                "activation_checkpointing": bool(settings.activation_checkpointing),
                "merge_mole": bool(settings.merge_mole),
                "execution_mode": settings.execution_mode,
                "predict_untrained_hessian": (
                    [task_name] if hessian_method == "native" else []
                ),
                "hessian_vmap": False,
                "hessian_method": hessian_method,
                "hessian_execution": (
                    "native_autograd_loop"
                    if hessian_method == "native"
                    else "structure_batched_centered_force_difference"
                ),
                "finite_difference_step_a": (
                    self.finite_difference_step_a
                    if hessian_method == "force_fd"
                    else None
                ),
                "finite_difference_batch_size": (
                    self.finite_difference_batch_size
                    if hessian_method == "force_fd"
                    else None
                ),
                "large_natoms_threshold": (
                    self.large_natoms_threshold
                    if hessian_method == "force_fd"
                    else None
                ),
                "large_finite_difference_batch_size": (
                    self.large_finite_difference_batch_size
                    if hessian_method == "force_fd"
                    else None
                ),
                "finite_difference_force_calls": (
                    "6N" if hessian_method == "force_fd" else None
                ),
                "compile": False,
                "device": device,
                "runtime_versions": _versions("fairchem-core", "torch", "ase", "numpy"),
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "hessian_unit": "eV/angstrom^2",
            }
        )

    def _prepare(self, atoms: Any) -> Any:
        prepared = atoms.copy()
        prepared.info["charge"] = self.charge
        prepared.info["spin"] = self.spin
        return prepared

    def _predict_native(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        prepared = self._prepare(atoms)
        hessian = self.calculator.get_property("hessian", prepared)
        results = self.calculator.results
        if "energy" not in results or "forces" not in results:
            raise RuntimeError("UMA Hessian prediction did not return energy and forces")
        return _validate_outputs(results["energy"], results["forces"], hessian, len(prepared))

    def _predict_batch(
        self,
        atoms_batch: list[Any],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Predict energies and forces for an equal-size ASE batch."""

        if not atoms_batch:
            return np.empty(0, dtype=np.float64), []
        from fairchem.core.datasets import data_list_collater

        prepared = [self._prepare(atoms) for atoms in atoms_batch]
        natoms = len(prepared[0])
        if any(len(atoms) != natoms for atoms in prepared):
            raise ValueError("UMA force batches must contain equal-size structures")
        for atoms in prepared:
            self.calculator.predictor.validate_atoms_data(atoms, self.task_name)
        data = data_list_collater(
            [self.calculator.a2g(atoms) for atoms in prepared],
            otf_graph=True,
        )
        prediction = self.calculator.predictor.predict(data)
        energies = np.asarray(
            prediction["energy"].detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)
        force_array = np.asarray(
            prediction["forces"].detach().cpu().numpy(), dtype=np.float64
        )
        expected_force_shape = (len(prepared) * natoms, 3)
        if energies.shape != (len(prepared),):
            raise ValueError(
                f"UMA energy batch has shape {energies.shape}; "
                f"expected ({len(prepared)},)"
            )
        if force_array.shape != expected_force_shape:
            raise ValueError(
                f"UMA force batch has shape {force_array.shape}; "
                f"expected {expected_force_shape}"
            )
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(force_array)):
            raise ValueError("UMA batch prediction contains non-finite values")
        forces = [
            force_array[index * natoms : (index + 1) * natoms].copy()
            for index in range(len(prepared))
        ]
        return energies, forces

    def evaluate_energy_forces_batch(
        self, atoms_batch: list[Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate only energies and forces; no Hessian path is entered."""

        return self._predict_batch(atoms_batch)

    def _predict_force_hessian(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        center_energy, center_forces = self._predict_batch([atoms])
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)

        def force_batch(displaced_batch: list[np.ndarray]) -> list[np.ndarray]:
            probes = []
            for displaced in displaced_batch:
                probe = atoms.copy()
                probe.set_positions(displaced)
                probes.append(probe)
            return self._predict_batch(probes)[1]

        estimate = batched_central_force_hessian(
            positions,
            force_batch,
            step_a=self.finite_difference_step_a,
            batch_size=self._force_batch_size(len(atoms)),
        )
        return _validate_outputs(
            center_energy[0], center_forces[0], estimate.raw, len(atoms)
        )

    def _evaluate_native(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        if self.hessian_method == "native":
            return self._predict_native(atoms)
        return self._predict_force_hessian(atoms)

    def get_forces(self, atoms: Any) -> np.ndarray:
        return self._predict_batch([atoms])[1][0]

    def _force_batch_size(self, natoms: int) -> int:
        if (
            self.large_natoms_threshold is not None
            and natoms >= self.large_natoms_threshold
        ):
            assert self.large_finite_difference_batch_size is not None
            return self.large_finite_difference_batch_size
        return self.finite_difference_batch_size

    def evaluate_batch(self, atoms_batch: list[Any]) -> list[ModelHessianEvaluation]:
        """Evaluate same-size UMA structures with batched displaced geometries.

        UMA's benchmark Hessian uses centered force differences.  Displacements
        from every molecule and Cartesian row are pooled into inference batches,
        allowing one force prediction to advance multiple Hessians at once.
        """

        if not atoms_batch:
            return []
        if self.hessian_method != "force_fd":
            return [self.evaluate(atoms) for atoms in atoms_batch]
        natoms = len(atoms_batch[0])
        if any(len(atoms) != natoms for atoms in atoms_batch):
            raise ValueError("UMA Hessian batches must contain equal-size structures")

        batch_size = len(atoms_batch)
        dimension = 3 * natoms
        step = self.finite_difference_step_a
        force_batch_size = self._force_batch_size(natoms)
        energies, center_forces = self._predict_batch(atoms_batch)
        displaced_forces = np.empty(
            (dimension, batch_size, 2, dimension), dtype=np.float64
        )
        pending_atoms: list[Any] = []
        pending_slots: list[tuple[int, int, int]] = []

        def flush_pending() -> None:
            if not pending_atoms:
                return
            _, predicted_forces = self._predict_batch(pending_atoms)
            if len(predicted_forces) != len(pending_slots):
                raise RuntimeError("UMA displaced-force batch cardinality mismatch")
            for slot, forces in zip(pending_slots, predicted_forces, strict=True):
                coordinate, molecule, sign_index = slot
                displaced_forces[coordinate, molecule, sign_index] = np.asarray(
                    forces, dtype=np.float64
                ).reshape(-1)
            pending_atoms.clear()
            pending_slots.clear()

        for coordinate in range(dimension):
            atom_index, axis = divmod(coordinate, 3)
            for molecule, atoms in enumerate(atoms_batch):
                for sign_index, direction in enumerate((-1.0, 1.0)):
                    probe = atoms.copy()
                    positions = np.asarray(
                        atoms.get_positions(), dtype=np.float64
                    ).copy()
                    positions[atom_index, axis] += direction * step
                    probe.set_positions(positions)
                    pending_atoms.append(probe)
                    pending_slots.append((coordinate, molecule, sign_index))
                    if len(pending_atoms) >= force_batch_size:
                        flush_pending()
        flush_pending()

        hessians = np.empty((batch_size, dimension, dimension), dtype=np.float64)
        for coordinate in range(dimension):
            force_minus = displaced_forces[coordinate, :, 0, :]
            force_plus = displaced_forces[coordinate, :, 1, :]
            hessians[:, :, coordinate] = -(force_plus - force_minus) / (2.0 * step)

        evaluations: list[ModelHessianEvaluation] = []
        for index, atoms in enumerate(atoms_batch):
            energy, forces, hessian = _validate_outputs(
                energies[index], center_forces[index], hessians[index], len(atoms)
            )
            evaluations.append(
                ModelHessianEvaluation(
                    energy_ev=energy,
                    forces_ev_per_a=forces,
                    hessian_ev_per_a2=hessian,
                    finite_difference_audit=None,
                    provenance=self.provenance,
                )
            )
        return evaluations


class ANI1xnrHessianAdapter(_BaseAdapter):
    """Exact autograd Hessian of the eight-member ANI-1xnr ensemble mean."""

    def __init__(
        self,
        *,
        model_info: str | Path,
        expected_sha256: str,
        device: str,
        dtype: str = "float64",
    ) -> None:
        import torch
        import torchani
        from torchani.neurochem import load_model_from_info_file

        if dtype not in {"float32", "float64"}:
            raise ValueError("ANI dtype must be float32 or float64")
        info_path, info_hash = _pinned_file(model_info, expected_sha256)
        torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        model = load_model_from_info_file(
            str(info_path),
            strategy="pyaev",
            periodic_table_index=True,
        )
        self.model = model.to(device=device, dtype=torch_dtype).eval()
        self.device = device
        self.torch_dtype = torch_dtype
        self.provenance = ModelProvenance(
            {
                "family": "ani1xnr",
                "model_info_path": str(info_path),
                "model_info_sha256": info_hash,
                "ensemble_members": len(self.model),
                "aev_strategy": "pyaev",
                "dtype": dtype,
                "device": device,
                "charge": 0,
                "spin_multiplicity": 1,
                "state_metadata_consumed": False,
                "torchani_version": getattr(torchani, "__version__", "unknown"),
                "runtime_versions": _versions("torchani", "torch", "ase", "numpy"),
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "hessian_unit": "eV/angstrom^2",
            }
        )

    def evaluate_energy_forces_batch(self, atoms_batch):
        import torch
        species, positions = self._torch_batch_inputs(atoms_batch)
        with torch.enable_grad():
            energies = self.model((species, positions)).energies
            gradient = torch.autograd.grad(energies.sum(), positions, create_graph=False)[0]
        energy_values = energies.detach().cpu().numpy().astype(np.float64) * HARTREE_TO_EV
        force_values = -gradient.detach().cpu().numpy().astype(np.float64) * HARTREE_TO_EV
        if not np.isfinite(energy_values).all() or not np.isfinite(force_values).all():
            raise ValueError("ANI returned non-finite energies or forces")
        return energy_values, list(force_values)

    def _torch_inputs(self, atoms: Any):
        return self._torch_batch_inputs([atoms])

    def _torch_batch_inputs(self, atoms_batch: list[Any]):
        import torch

        if not atoms_batch:
            raise ValueError("ANI batch must contain at least one structure")
        natoms = len(atoms_batch[0])
        if any(len(atoms) != natoms for atoms in atoms_batch):
            raise ValueError("ANI Hessian batches must contain equal-size structures")
        species = torch.as_tensor(
            np.stack(
                [np.asarray(atoms.numbers, dtype=np.int64) for atoms in atoms_batch]
            ),
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.as_tensor(
            np.stack(
                [np.asarray(atoms.positions, dtype=np.float64) for atoms in atoms_batch]
            ),
            dtype=self.torch_dtype,
            device=self.device,
        ).requires_grad_(True)
        return species, positions

    def _energy_gradient(self, atoms: Any, *, create_graph: bool):
        import torch

        species, positions = self._torch_inputs(atoms)
        output = self.model((species, positions))
        energy_hartree = output.energies.sum()
        gradient = torch.autograd.grad(
            energy_hartree,
            positions,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return energy_hartree, gradient, positions

    def get_forces(self, atoms: Any) -> np.ndarray:
        energy, gradient, positions = self._energy_gradient(atoms, create_graph=False)
        del energy, positions
        return (-gradient.detach().cpu().numpy()[0] * HARTREE_TO_EV).astype(np.float64)

    def _evaluate_native(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        evaluation = self.evaluate_batch([atoms])[0]
        return (
            evaluation.energy_ev,
            evaluation.forces_ev_per_a,
            evaluation.hessian_ev_per_a2,
        )

    def evaluate_batch(self, atoms_batch: list[Any]) -> list[ModelHessianEvaluation]:
        """Evaluate exact analytic Hessians for an equal-atom-count batch.

        ANI energies are independent along the leading molecule dimension.  Each
        reverse-mode call therefore produces the same Cartesian Hessian row for
        every molecule in the batch, amortizing the Python/autograd launch cost.
        """

        import torch

        species, positions = self._torch_batch_inputs(atoms_batch)
        energy_hartree = self.model((species, positions)).energies
        expected_energy_shape = (len(atoms_batch),)
        if energy_hartree.shape != expected_energy_shape:
            raise ValueError(
                f"ANI energy batch has shape {tuple(energy_hartree.shape)}; "
                f"expected {expected_energy_shape}"
            )
        gradient = torch.autograd.grad(
            energy_hartree.sum(),
            positions,
            create_graph=True,
            retain_graph=True,
        )[0]
        batch_size = len(atoms_batch)
        dimension = gradient.shape[1] * gradient.shape[2]
        flat_gradient = gradient.reshape(batch_size, dimension)
        rows = []
        for index in range(dimension):
            row = torch.autograd.grad(
                flat_gradient[:, index].sum(),
                positions,
                retain_graph=index + 1 < dimension,
            )[0]
            rows.append(row.reshape(batch_size, dimension))
        hessians = torch.stack(rows, dim=1).detach().cpu().numpy() * HARTREE_TO_EV
        forces = -gradient.detach().cpu().numpy() * HARTREE_TO_EV
        energies = energy_hartree.detach().cpu().numpy() * HARTREE_TO_EV

        evaluations: list[ModelHessianEvaluation] = []
        for index, atoms in enumerate(atoms_batch):
            energy, force, hessian = _validate_outputs(
                energies[index],
                forces[index],
                hessians[index],
                len(atoms),
            )
            evaluations.append(
                ModelHessianEvaluation(
                    energy_ev=energy,
                    forces_ev_per_a=force,
                    hessian_ev_per_a2=hessian,
                    finite_difference_audit=None,
                    provenance=self.provenance,
                )
            )
        return evaluations


class OrbMolV2HessianAdapter(_BaseAdapter):
    """Analytic local Hessian of conservative OrbMol-v2."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        expected_sha256: str,
        device: str,
    ) -> None:
        import torch
        import orb_models
        from orb_models.forcefield import pretrained

        checkpoint_path, checkpoint_hash = _pinned_file(checkpoint, expected_sha256)
        self.model, self.graph_adapter = pretrained.orbmol_v2(
            weights_path=str(checkpoint_path),
            device=device,
            precision="float32-high",
            compile=False,
        )
        self.model.eval()
        self.device = device
        self.torch = torch
        self.provenance = ModelProvenance(
            {
                "family": "orbmol_v2",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "precision": "float32-high",
                "fp64_energy": True,
                "compile": False,
                "edge_method": "knn_alchemi",
                "hessian_graph_policy": "fixed_neighbor_graph_at_reference_geometry",
                "hessian_execution": "batched_equal_natoms_reverse_rows",
                "batch_numerical_semantics": (
                    "float32 graph batching can change reduction order relative to scalar"
                ),
                "charge": 0,
                "spin": 1,
                "spin_semantics": "spin_multiplicity",
                "device": device,
                "orb_models_version": getattr(orb_models, "__version__", "unknown"),
                "runtime_versions": _versions("orb-models", "torch", "ase", "numpy"),
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "hessian_unit": "eV/angstrom^2",
            }
        )

    def _prepare(self, atoms: Any) -> Any:
        prepared = atoms.copy()
        prepared.info["charge"] = 0
        prepared.info["spin"] = 1
        return prepared

    def _graph_batch(self, atoms_batch: list[Any]):
        if not atoms_batch:
            raise ValueError("OrbMol-v2 batch must contain at least one structure")
        natoms = len(atoms_batch[0])
        if any(len(atoms) != natoms for atoms in atoms_batch):
            raise ValueError(
                "OrbMol-v2 Hessian batches must contain equal-size structures"
            )
        return self.graph_adapter.from_ase_atoms_list(
            [self._prepare(atoms) for atoms in atoms_batch],
            device=self.device,
            edge_method="knn_alchemi",
            output_dtype=self.torch.float32,
            graph_construction_dtype=self.torch.float32,
        )

    def _predict_batch(self, atoms_batch: list[Any], *, hessian: bool):
        torch = self.torch
        graph = self._graph_batch(atoms_batch)
        batch_size = len(atoms_batch)
        natoms = len(atoms_batch[0])
        dimension = 3 * natoms
        previous_training = self.model.training
        # Only the top-level flag controls create_graph in the conservative
        # regressor. Children remain in eval mode, avoiding training behavior.
        self.model.training = bool(hessian)
        try:
            with torch.enable_grad():
                output = self.model.predict(graph, split=False, fp64_energy=True)
            energy = output[self.model.energy_name]
            forces = output[self.model.grad_forces_name]
            if energy.numel() != batch_size:
                raise ValueError(
                    f"OrbMol-v2 energy batch has {energy.numel()} entries; "
                    f"expected {batch_size}"
                )
            if forces.shape != (batch_size * natoms, 3):
                raise ValueError(
                    f"OrbMol-v2 force batch has shape {tuple(forces.shape)}; "
                    f"expected ({batch_size * natoms}, 3)"
                )
            energy = energy.reshape(batch_size)
            forces = forces.reshape(batch_size, natoms, 3)
            if not hessian:
                return energy, forces, None
            positions = graph.node_features["positions"]
            flat_forces = forces.reshape(batch_size, dimension)
            rows = []
            for index in range(dimension):
                derivative = torch.autograd.grad(
                    flat_forces[:, index].sum(),
                    positions,
                    retain_graph=index + 1 < dimension,
                )[0]
                rows.append(-derivative.reshape(batch_size, dimension))
            matrix = torch.stack(rows, dim=1)
            return energy, forces, matrix
        finally:
            self.model.training = previous_training

    def get_forces(self, atoms: Any) -> np.ndarray:
        _, forces, _ = self._predict_batch([atoms], hessian=False)
        return np.asarray(forces[0].detach().cpu().numpy(), dtype=np.float64)

    def evaluate_energy_forces_batch(
        self, atoms_batch: list[Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate an equal-size Orb batch without constructing Hessians."""

        if not atoms_batch:
            return np.empty((0,), dtype=np.float64), []
        energy, forces, hessian = self._predict_batch(atoms_batch, hessian=False)
        if hessian is not None:
            raise RuntimeError("OrbMol-v2 entered the Hessian path during force-only inference")
        energies = np.asarray(energy.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        force_array = np.asarray(forces.detach().cpu().numpy(), dtype=np.float64)
        if energies.shape != (len(atoms_batch),):
            raise ValueError(
                f"OrbMol-v2 energy batch has shape {energies.shape}; "
                f"expected ({len(atoms_batch)},)"
            )
        result_forces = [force_array[index].copy() for index in range(len(atoms_batch))]
        if not np.all(np.isfinite(energies)):
            raise ValueError("OrbMol-v2 batch returned non-finite energies")
        for atoms, force in zip(atoms_batch, result_forces, strict=True):
            if force.shape != (len(atoms), 3) or not np.all(np.isfinite(force)):
                raise ValueError("OrbMol-v2 batch returned invalid forces")
        return energies, result_forces

    def _evaluate_native(self, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
        evaluation = self.evaluate_batch([atoms])[0]
        return (
            evaluation.energy_ev,
            evaluation.forces_ev_per_a,
            evaluation.hessian_ev_per_a2,
        )

    def evaluate_batch(self, atoms_batch: list[Any]) -> list[ModelHessianEvaluation]:
        """Evaluate exact fixed-graph Hessians for an equal-size Orb batch.

        Orb concatenates independent molecular graphs. Summing the same force
        component across graphs therefore lets each reverse-mode call recover
        that Cartesian Hessian row for every molecule simultaneously.
        """

        energy, forces, hessian = self._predict_batch(atoms_batch, hessian=True)
        assert hessian is not None
        energies = energy.detach().cpu().numpy()
        force_arrays = forces.detach().cpu().numpy()
        hessians = hessian.detach().cpu().numpy()
        evaluations: list[ModelHessianEvaluation] = []
        for index, atoms in enumerate(atoms_batch):
            validated_energy, validated_forces, validated_hessian = _validate_outputs(
                energies[index],
                force_arrays[index],
                hessians[index],
                len(atoms),
            )
            evaluations.append(
                ModelHessianEvaluation(
                    energy_ev=validated_energy,
                    forces_ev_per_a=validated_forces,
                    hessian_ev_per_a2=validated_hessian,
                    finite_difference_audit=None,
                    provenance=self.provenance,
                )
            )
        return evaluations


class SevenNetEnergyForceAdapter:
    """Batched fixed-geometry energy/force inference for SevenNet-Omni."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        expected_sha256: str,
        modal: str,
        device: str,
        torch_num_threads: int,
        enable_flash: bool = False,
    ) -> None:
        if modal not in {"omol25_low", "spice"}:
            raise ValueError("SevenNet DORTS modal must be 'omol25_low' or 'spice'")
        checkpoint_path, checkpoint_hash = _pinned_file(checkpoint, expected_sha256)

        import torch
        import sevenn._keys as sevenn_keys
        from sevenn.util import load_checkpoint

        torch.set_num_threads(torch_num_threads)
        loaded = load_checkpoint(str(checkpoint_path))
        model = loaded.build_model(enable_flash=enable_flash)
        if model.modal_map is None or modal not in model.modal_map:
            available = [] if model.modal_map is None else sorted(model.modal_map)
            raise ValueError(
                f"SevenNet modal {modal!r} is unavailable; choices={available}"
            )
        model.to(device)
        model.set_is_batch_data(True)
        model.eval()

        self.torch = torch
        self.keys = sevenn_keys
        self.model = model
        self.device = torch.device(device)
        self.modal = modal
        self.cutoff = float(model.cutoff)
        self.type_map = dict(loaded.config[sevenn_keys.TYPE_MAP])
        self.provenance = ModelProvenance(
            {
                "model_kind": "sevennet",
                "model_name": "SevenNet-Omni-i12",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "head": modal,
                "modal": modal,
                "device": str(self.device),
                "dtype": "float32",
                "cutoff_a": self.cutoff,
                "tensor_product_accelerator": "FlashTP" if enable_flash else "none",
                "packages": _versions(
                    "sevenn", "torch", "torch-geometric", "e3nn", "flashTP-e3nn"
                ),
            }
        )

    def evaluate_energy_forces_batch(
        self, atoms_batch: list[Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate one heterogeneous molecular batch on a single GPU."""

        if not atoms_batch:
            return np.empty((0,), dtype=np.float64), []

        from torch_geometric.loader import DataLoader
        from sevenn.atom_graph_data import AtomGraphData
        from sevenn.train.dataload import unlabeled_atoms_to_graph
        from sevenn.train.modal_dataset import SevenNetMultiModalDataset
        from sevenn.util import to_atom_graph_list

        graphs = [
            AtomGraphData.from_numpy_dict(
                unlabeled_atoms_to_graph(atoms, self.cutoff)
            )
            for atoms in atoms_batch
        ]
        dataset = SevenNetMultiModalDataset({self.modal: graphs})
        loader = DataLoader(dataset, batch_size=len(graphs), shuffle=False)
        batch = next(iter(loader)).to(self.device)
        output = self.model(batch).detach().cpu()
        separated = to_atom_graph_list(output)
        energies = np.asarray(
            [
                float(
                    np.asarray(item[self.keys.PRED_TOTAL_ENERGY]).reshape(-1)[0]
                )
                for item in separated
            ],
            dtype=np.float64,
        )
        forces = [
            np.asarray(item[self.keys.PRED_FORCE], dtype=np.float64).copy()
            for item in separated
        ]
        if len(energies) != len(atoms_batch) or len(forces) != len(atoms_batch):
            raise RuntimeError("SevenNet batch output cardinality mismatch")
        for atoms, force in zip(atoms_batch, forces, strict=True):
            if force.shape != (len(atoms), 3) or not np.all(np.isfinite(force)):
                raise ValueError("SevenNet returned invalid forces")
        if not np.all(np.isfinite(energies)):
            raise ValueError("SevenNet returned invalid energies")
        return energies, forces


def load_hessian_adapter(
    *,
    model_kind: str,
    checkpoint: str,
    expected_sha256: str,
    head: str | None,
    dtype: str,
    device: str,
    model_name: str | None,
    charge: int,
    spin: int,
    torch_num_threads: int,
    uma_inference_settings: str = "default",
    uma_compile: bool = False,
    uma_hessian_method: str = "force_fd",
    uma_finite_difference_step_a: float = 1.0e-3,
    uma_finite_difference_batch_size: int = 4,
    uma_large_natoms_threshold: int | None = None,
    uma_large_finite_difference_batch_size: int | None = None,
    sevennet_enable_flash: bool = False,
):
    """Create one adapter from the common shard-runner arguments."""

    if model_kind in {"mace_polar", "mace_mp"}:
        from .mace_hessian import MACEHessianAdapter

        return MACEHessianAdapter.from_checkpoint(
            checkpoint,
            model_kind=model_kind,
            head=head,
            dtype=dtype,
            device=device,
            expected_sha256=expected_sha256,
        )
    if model_kind == "uma":
        if model_name is None or head is None:
            raise ValueError("UMA requires --model-name and --head")
        return UMAHessianAdapter(
            model_name=model_name,
            checkpoint=checkpoint,
            expected_sha256=expected_sha256,
            task_name=head,
            charge=charge,
            spin=spin,
            device=device,
            torch_num_threads=torch_num_threads,
            inference_settings_name=uma_inference_settings,
            compile_model=uma_compile,
            hessian_method=uma_hessian_method,
            finite_difference_step_a=uma_finite_difference_step_a,
            finite_difference_batch_size=uma_finite_difference_batch_size,
            large_natoms_threshold=uma_large_natoms_threshold,
            large_finite_difference_batch_size=(
                uma_large_finite_difference_batch_size
            ),
        )
    if model_kind == "orbmol":
        if charge != 0 or spin != 1:
            raise ValueError("OrbMol-v2 benchmark requires charge=0 spin=1")
        return OrbMolV2HessianAdapter(
            checkpoint=checkpoint,
            expected_sha256=expected_sha256,
            device=device,
        )
    if model_kind == "ani1xnr":
        import torch
        torch.set_num_threads(torch_num_threads)
        if charge != 0 or spin != 1:
            raise ValueError("ANI-1xnr benchmark is restricted to neutral singlets")
        return ANI1xnrHessianAdapter(
            model_info=checkpoint,
            expected_sha256=expected_sha256,
            device=device,
            dtype=dtype,
        )
    if model_kind == "sevennet":
        if head is None:
            raise ValueError("SevenNet requires --head")
        if charge != 0 or spin != 1:
            raise ValueError("SevenNet DORTS benchmark requires charge=0 spin=1")
        return SevenNetEnergyForceAdapter(
            checkpoint=checkpoint,
            expected_sha256=expected_sha256,
            modal=head,
            device=device,
            torch_num_threads=torch_num_threads,
            enable_flash=sevennet_enable_flash,
        )
    raise ValueError(f"unsupported model_kind={model_kind!r}")
