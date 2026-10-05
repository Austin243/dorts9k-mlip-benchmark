"""Lazy, manifest-backed access to the public DORTS-9K HDF5 release."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
from ase.data import atomic_numbers, chemical_symbols


DATASET_DOI = "10.5281/zenodo.17141108"
REFERENCE_LEVEL = "omegaB97M-V/def2-TZVP"
UNKNOWN_NEGATIVE_MODE_COUNT = -1


@dataclass(frozen=True)
class DORTSRecord:
    """One DORTS configuration with its DFT energy and forces."""

    key_index: int
    reaction_number: int
    reaction_id: str
    role: str
    expected_negative_mode_count: int
    atomic_numbers: np.ndarray
    positions: np.ndarray
    formula: str
    source_path: str
    reference_energy_ev: float
    reference_forces_ev_per_a: np.ndarray
    barrier_kcal_mol: float = float("nan")
    bond_changes: str = ""

    @property
    def natoms(self) -> int:
        return int(self.atomic_numbers.size)

    @property
    def stationarity_class_evaluable(self) -> bool:
        return self.expected_negative_mode_count >= 0


def _decode_symbols(values: np.ndarray) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            symbols.append(value.decode("utf-8"))
        else:
            symbols.append(str(value))
    return tuple(symbols)


def symbols_to_numbers(values: np.ndarray) -> np.ndarray:
    """Convert a DORTS symbol dataset to atomic numbers."""

    try:
        return np.asarray([atomic_numbers[s] for s in _decode_symbols(values)], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown element in DORTS symbols: {exc.args[0]!r}") from exc


def dorts_formula(numbers: np.ndarray) -> str:
    """Return a Hill-system formula for the broader DORTS element set."""

    counts = Counter(chemical_symbols[int(number)] for number in numbers)
    order: list[str] = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
        order.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    else:
        order.extend(sorted(counts))
    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] != 1 else "")
        for symbol in order
    )


def _configuration(array: h5py.Dataset, index: int, *, trailing_ndim: int) -> np.ndarray:
    """Read one scalar/vector configuration from scalar or stacked storage."""

    if array.ndim == trailing_ndim:
        if index != 0:
            raise IndexError(f"single-configuration dataset cannot provide index {index}")
        value = array[()]
    elif array.ndim == trailing_ndim + 1:
        value = array[index]
    else:
        raise ValueError(f"unexpected DORTS dataset shape {array.shape}")
    result = np.asarray(value)
    if not np.all(np.isfinite(result)):
        raise ValueError("DORTS configuration contains non-finite values")
    return result


class DORTS9KDataset:
    """Compact-manifest access to DORTS-9K without loading coordinates eagerly."""

    FORMAT_VERSION = 1

    def __init__(self, manifest: str | Path, source_h5: str | Path) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.source_path = Path(source_h5).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.source_path.is_file():
            raise FileNotFoundError(self.source_path)

        with h5py.File(self.manifest_path, "r") as handle:
            version = int(handle.attrs.get("format_version", -1))
            if version != self.FORMAT_VERSION:
                raise ValueError(f"unsupported DORTS manifest format {version}")
            if handle.attrs.get("dataset_id") != "DORTS-9K":
                raise ValueError("manifest is not for DORTS-9K")
            self.reaction_names = tuple(handle["reaction_names"].asstr()[:])
            self.reaction_formulas = tuple(handle["reaction_formulas"].asstr()[:])
            self.reaction_numbers = np.asarray(
                handle["reaction_numbers"], dtype=np.int64
            )
            self.reaction_atomic_numbers = tuple(
                np.asarray(values, dtype=np.int64)
                for values in handle["reaction_atomic_numbers"]
            )
            records = handle["records"]
            self.record_reaction_index = np.asarray(
                records["reaction_index"], dtype=np.int32
            )
            self.record_subset_code = np.asarray(records["subset_code"], dtype=np.uint8)
            self.record_source_kind = np.asarray(records["source_kind"], dtype=np.uint8)
            self.record_source_names = np.asarray(records["source_name"])
            self.record_natoms = np.asarray(records["natoms"], dtype=np.uint8)
            self.subset_names = {
                int(key): value
                for key, value in json.loads(handle.attrs["subset_names_json"]).items()
            }
            self.source_md5 = str(handle.attrs["source_md5"])

        size = len(self.record_reaction_index)
        if not (
            len(self.record_subset_code)
            == len(self.record_source_kind)
            == len(self.record_source_names)
            == len(self.record_natoms)
            == size
        ):
            raise ValueError("DORTS manifest record columns have different lengths")
        if size == 0:
            raise ValueError("DORTS manifest is empty")
        if len(self.reaction_names) != len(self.reaction_atomic_numbers):
            raise ValueError("DORTS reaction metadata columns have different lengths")
        self._source: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.record_reaction_index)

    def __enter__(self) -> "DORTS9KDataset":
        self._open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        del exc_type, exc, traceback
        self.close()

    def _open(self) -> h5py.File:
        if self._source is None:
            self._source = h5py.File(self.source_path, "r")
        return self._source

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def _group(self, source_name: str, subset_name: str, source_kind: int) -> h5py.Group:
        source = self._open()
        if source_kind == 0:
            return source["NMS"][source_name]
        if source_kind == 1:
            return source["IRC"][subset_name][source_name]
        raise ValueError(f"unknown DORTS source kind {source_kind}")

    def record(self, index: int) -> DORTSRecord:
        if not 0 <= index < len(self):
            raise IndexError(f"DORTS key index out of range: {index}")
        reaction_index = int(self.record_reaction_index[index])
        reaction_name = self.reaction_names[reaction_index]
        subset_name = self.subset_names[int(self.record_subset_code[index])]
        source_kind = int(self.record_source_kind[index])
        source_name = self.record_source_names[index]
        if isinstance(source_name, bytes):
            source_name = source_name.decode("utf-8")
        else:
            source_name = str(source_name)
        group = self._group(source_name, subset_name, source_kind)
        positions = _configuration(group["coords"], 0, trailing_ndim=2)
        forces = _configuration(group["forces"], 0, trailing_ndim=2)
        energy = _configuration(group["energy"], 0, trailing_ndim=0)
        numbers = self.reaction_atomic_numbers[reaction_index].copy()
        expected_natoms = int(self.record_natoms[index])
        if positions.shape != (expected_natoms, 3):
            raise ValueError(
                f"DORTS coordinate shape mismatch for key {index}: {positions.shape}"
            )
        if forces.shape != (expected_natoms, 3):
            raise ValueError(f"DORTS force shape mismatch for key {index}: {forces.shape}")
        if numbers.shape != (expected_natoms,):
            raise ValueError(f"DORTS symbol count mismatch for key {index}")
        formula = self.reaction_formulas[reaction_index]
        if dorts_formula(numbers) != formula:
            raise ValueError(f"DORTS manifest formula mismatch for key {index}")

        normalized = subset_name.lower()
        role = {
            "ts": "TS",
            "reactant": "RS",
            "product": "PS",
            "f_rev": "F_rev",
            "f_forw": "F_forw",
            "nms": "NMS",
        }.get(normalized, subset_name)
        if source_kind == 0:
            role = f"NMS_{role}"
            expected_negative = UNKNOWN_NEGATIVE_MODE_COUNT
            source_path = f"/NMS/{source_name}"
        else:
            expected_negative = {"TS": 1, "RS": 0, "PS": 0}.get(
                role, UNKNOWN_NEGATIVE_MODE_COUNT
            )
            source_path = f"/IRC/{subset_name}/{source_name}"
        return DORTSRecord(
            key_index=index,
            reaction_number=int(self.reaction_numbers[reaction_index]),
            reaction_id=reaction_name,
            role=role,
            expected_negative_mode_count=expected_negative,
            atomic_numbers=numbers,
            positions=np.asarray(positions, dtype=np.float64),
            formula=formula,
            source_path=source_path,
            reference_energy_ev=float(np.asarray(energy).reshape(-1)[0]),
            reference_forces_ev_per_a=np.asarray(forces, dtype=np.float64),
        )

    def iter_records(self, indices: Iterable[int]) -> Iterator[DORTSRecord]:
        try:
            for index in indices:
                yield self.record(index)
        finally:
            self.close()
