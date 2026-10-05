#!/usr/bin/env python3
"""Validate DORTS-9K and build a compact, deterministic benchmark manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from dorts9k.dorts import (
    DATASET_DOI,
    REFERENCE_LEVEL,
    dorts_formula,
    symbols_to_numbers,
)


EXPECTED_MD5 = "f59c2b2b7fb8869f2e734d0bbacdfd08"
ANI_ATOMIC_NUMBERS = frozenset((1, 6, 7, 8))
PREFERRED_IRC_ORDER = ("reactant", "ts", "product", "F_rev", "F_forw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--reaction-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--ani-indices", required=True)
    parser.add_argument("--all-indices", required=True)
    parser.add_argument("--smoke-indices", required=True)
    parser.add_argument("--ani-smoke-indices", required=True)
    parser.add_argument("--expected-md5", default=EXPECTED_MD5)
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify Zenodo checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    )


def base_reaction_name(source_name: str) -> str:
    if "_opt-irc_" not in source_name:
        raise ValueError(f"DORTS source key has no _opt-irc_ marker: {source_name}")
    return source_name.split("_opt-irc_", 1)[0]


def nms_subset_name(source_name: str) -> str:
    suffix = source_name.split("_opt-irc_", 1)[1]
    if "_qnms" not in suffix:
        raise ValueError(f"DORTS NMS key has no _qnms marker: {source_name}")
    return suffix.rsplit("_qnms", 1)[0]


def configuration_count(group: h5py.Group) -> int:
    coords = group["coords"]
    forces = group["forces"]
    energy = group["energy"]
    if coords.ndim == 2:
        count = 1
        expected_coords = (coords.shape[0], 3)
        if coords.shape != expected_coords or forces.shape != expected_coords:
            raise ValueError(f"invalid single-configuration shapes in {group.name}")
        if energy.ndim not in (0, 1) or energy.size != 1:
            raise ValueError(f"invalid single energy shape in {group.name}: {energy.shape}")
    elif coords.ndim == 3:
        count = int(coords.shape[0])
        if coords.shape[2] != 3 or forces.shape != coords.shape:
            raise ValueError(f"invalid stacked coordinate/force shapes in {group.name}")
        if energy.ndim != 1 or energy.shape != (count,):
            raise ValueError(f"invalid stacked energy shape in {group.name}: {energy.shape}")
    else:
        raise ValueError(f"invalid coordinate shape in {group.name}: {coords.shape}")
    if count < 1:
        raise ValueError(f"empty configuration group {group.name}")
    return count


def validate_samples(group: h5py.Group, count: int) -> None:
    indices = sorted({0, count // 2, count - 1})
    for name in ("coords", "forces", "energy"):
        dataset = group[name]
        for index in indices:
            value = dataset[()] if dataset.ndim in (0, 2) else dataset[index]
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite {name} sample in {group.name} at {index}")


def ordered_subsets(irc: h5py.Group) -> list[str]:
    available = list(irc.keys())
    by_lower = {name.lower(): name for name in available}
    ordered: list[str] = []
    for preferred in PREFERRED_IRC_ORDER:
        actual = by_lower.get(preferred.lower())
        if actual is not None:
            ordered.append(actual)
    ordered.extend(sorted(set(available) - set(ordered), key=natural_key))
    return ordered


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_h5).resolve()
    output_path = Path(args.output_manifest).resolve()
    observed_md5 = file_md5(source_path)
    if observed_md5 != args.expected_md5.lower():
        raise RuntimeError(
            f"DORTS source checksum mismatch: expected={args.expected_md5.lower()} "
            f"observed={observed_md5}"
        )

    with h5py.File(source_path, "r") as source:
        if not {"IRC", "NMS"}.issubset(source.keys()):
            raise ValueError(f"DORTS root groups are {list(source.keys())}, expected IRC/NMS")
        subsets = ordered_subsets(source["IRC"])
        subset_by_lower = {name.lower(): name for name in subsets}
        subset_codes = {name: index for index, name in enumerate(subsets)}
        first_nms: dict[str, str] = {}
        nms_counts: Counter[str] = Counter()
        nms_role_counts: Counter[str] = Counter()
        reaction_names_set: set[str] = set()
        max_source_name_length = 1
        for source_name in source["NMS"]:
            base = base_reaction_name(source_name)
            role_raw = nms_subset_name(source_name)
            try:
                role = subset_by_lower[role_raw.lower()]
            except KeyError as exc:
                raise ValueError(f"unknown NMS source role {role_raw!r}") from exc
            reaction_names_set.add(base)
            first_nms.setdefault(base, source_name)
            nms_counts[base] += 1
            nms_role_counts[role] += 1
            max_source_name_length = max(max_source_name_length, len(source_name.encode()))

        irc_sources: dict[str, dict[str, str]] = defaultdict(dict)
        for subset in subsets:
            for source_name in source["IRC"][subset]:
                base = base_reaction_name(source_name)
                reaction_names_set.add(base)
                if subset in irc_sources[base]:
                    raise ValueError(f"duplicate IRC {subset} entry for {base}")
                irc_sources[base][subset] = source_name
                max_source_name_length = max(
                    max_source_name_length, len(source_name.encode())
                )

        names = sorted(reaction_names_set, key=natural_key)
        if not names:
            raise ValueError("DORTS source contains no reactions")
        reaction_index_by_name = {name: index for index, name in enumerate(names)}

        reaction_rows: list[dict[str, object]] = []
        reaction_numbers = list(range(1, len(names) + 1))
        formulas: list[str] = []
        atomic_number_rows: list[np.ndarray] = []
        total_records = len(source["NMS"]) + sum(
            len(source["IRC"][subset]) for subset in subsets
        )
        for reaction_index, name in enumerate(names):
            candidates: list[h5py.Group] = []
            for subset in subsets:
                source_name = irc_sources.get(name, {}).get(subset)
                if source_name is not None:
                    group = source["IRC"][subset][source_name]
                    if configuration_count(group) != 1:
                        raise ValueError(f"IRC group {group.name} is not scalar")
                    validate_samples(group, 1)
                    candidates.append(group)
            if name in first_nms:
                group = source["NMS"][first_nms[name]]
                if configuration_count(group) != 1:
                    raise ValueError(f"NMS group {group.name} is not scalar")
                validate_samples(group, 1)
                candidates.append(group)
            if not candidates:
                raise AssertionError(name)
            numbers = symbols_to_numbers(candidates[0]["symbols"][()])
            for group in candidates[1:]:
                observed = symbols_to_numbers(group["symbols"][()])
                if not np.array_equal(observed, numbers):
                    raise ValueError(f"inconsistent symbols for reaction {name}")
            natoms = int(numbers.size)
            for group in candidates:
                coords = group["coords"]
                group_natoms = int(coords.shape[-2])
                if group_natoms != natoms:
                    raise ValueError(f"atom-count mismatch in {group.name}")
            formula = dorts_formula(numbers)
            element_set = sorted(set(map(int, numbers)))
            ani_supported = set(element_set).issubset(ANI_ATOMIC_NUMBERS)
            nms_count = int(nms_counts[name])
            irc_count = len(irc_sources.get(name, {}))
            record_count = nms_count + irc_count
            formulas.append(formula)
            atomic_number_rows.append(numbers.astype(np.int16))
            reaction_rows.append(
                {
                    "reaction_index": reaction_index,
                    "reaction_number": reaction_numbers[reaction_index],
                    "reaction_id": name,
                    "natoms": natoms,
                    "formula": formula,
                    "atomic_numbers": ",".join(map(str, element_set)),
                    "ani_chno_supported": int(ani_supported),
                    "nms_configuration_count": nms_count,
                    "irc_configuration_count": irc_count,
                    "total_configuration_count": record_count,
                }
            )

        record_reaction = np.empty(total_records, dtype=np.int32)
        record_subset = np.empty(total_records, dtype=np.uint8)
        record_source_kind = np.empty(total_records, dtype=np.uint8)
        record_natoms = np.empty(total_records, dtype=np.uint8)
        ani_supported_records = np.empty(total_records, dtype=np.bool_)
        record_source_name = np.empty(total_records, dtype=f"S{max_source_name_length}")
        cursor = 0
        subset_names = {index: name for index, name in enumerate(subsets)}
        for source_name in source["NMS"]:
            base = base_reaction_name(source_name)
            role = subset_by_lower[nms_subset_name(source_name).lower()]
            reaction_index = reaction_index_by_name[base]
            row = reaction_rows[reaction_index]
            record_reaction[cursor] = reaction_index
            record_subset[cursor] = subset_codes[role]
            record_source_kind[cursor] = 0
            record_natoms[cursor] = int(row["natoms"])
            ani_supported_records[cursor] = bool(row["ani_chno_supported"])
            record_source_name[cursor] = source_name.encode()
            cursor += 1
        for subset in subsets:
            for source_name in source["IRC"][subset]:
                base = base_reaction_name(source_name)
                reaction_index = reaction_index_by_name[base]
                row = reaction_rows[reaction_index]
                record_reaction[cursor] = reaction_index
                record_subset[cursor] = subset_codes[subset]
                record_source_kind[cursor] = 1
                record_natoms[cursor] = int(row["natoms"])
                ani_supported_records[cursor] = bool(row["ani_chno_supported"])
                record_source_name[cursor] = source_name.encode()
                cursor += 1
        if cursor != total_records:
            raise AssertionError((cursor, total_records))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    string_type = h5py.string_dtype("utf-8")
    vlen_int16 = h5py.vlen_dtype(np.dtype("int16"))
    with h5py.File(temporary, "w") as manifest:
        manifest.attrs["format_version"] = 1
        manifest.attrs["dataset_id"] = "DORTS-9K"
        manifest.attrs["dataset_doi"] = DATASET_DOI
        manifest.attrs["reference_level"] = REFERENCE_LEVEL
        manifest.attrs["source_path"] = str(source_path)
        manifest.attrs["source_md5"] = observed_md5
        manifest.attrs["record_count"] = total_records
        manifest.attrs["reaction_count"] = len(reaction_rows)
        manifest.attrs["subset_names_json"] = json.dumps(subset_names, sort_keys=True)
        manifest.create_dataset("reaction_names", data=np.asarray(names, dtype=object), dtype=string_type)
        manifest.create_dataset("reaction_numbers", data=np.asarray(reaction_numbers, dtype=np.int64))
        manifest.create_dataset("reaction_formulas", data=np.asarray(formulas, dtype=object), dtype=string_type)
        atomic_dataset = manifest.create_dataset(
            "reaction_atomic_numbers", (len(atomic_number_rows),), dtype=vlen_int16
        )
        for index, numbers in enumerate(atomic_number_rows):
            atomic_dataset[index] = numbers
        records = manifest.create_group("records")
        records.create_dataset("reaction_index", data=record_reaction, compression="gzip", shuffle=True)
        records.create_dataset("subset_code", data=record_subset, compression="gzip", shuffle=True)
        records.create_dataset("source_kind", data=record_source_kind, compression="gzip", shuffle=True)
        records.create_dataset("source_name", data=record_source_name, compression="gzip", shuffle=True)
        records.create_dataset("natoms", data=record_natoms, compression="gzip", shuffle=True)
        records.create_dataset("ani_chno_supported", data=ani_supported_records, compression="gzip", shuffle=True)
    temporary.replace(output_path)

    reaction_csv = Path(args.reaction_csv).resolve()
    reaction_csv.parent.mkdir(parents=True, exist_ok=True)
    with reaction_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reaction_rows[0]))
        writer.writeheader()
        writer.writerows(reaction_rows)

    all_indices = np.arange(total_records, dtype=np.int64)
    ani_indices = np.flatnonzero(ani_supported_records).astype(np.int64)
    Path(args.all_indices).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.all_indices, all_indices)
    np.save(args.ani_indices, ani_indices)

    def representative_indices(candidates: np.ndarray, count: int = 64) -> np.ndarray:
        if candidates.size <= count:
            return candidates.copy()
        positions = np.linspace(0, candidates.size - 1, count, dtype=np.int64)
        selected = set(map(int, candidates[positions]))
        candidate_natoms = record_natoms[candidates]
        selected.add(int(candidates[int(np.argmin(candidate_natoms))]))
        selected.add(int(candidates[int(np.argmax(candidate_natoms))]))
        # Include one structure for every element represented by these reactions.
        selected_elements = {
            int(value)
            for key in selected
            for value in atomic_number_rows[int(record_reaction[key])]
        }
        for reaction_index, numbers in enumerate(atomic_number_rows):
            if set(map(int, numbers)).issubset(selected_elements):
                continue
            matching = candidates[record_reaction[candidates] == reaction_index]
            if matching.size:
                selected.add(int(matching[0]))
                selected_elements.update(map(int, numbers))
        return np.asarray(sorted(selected), dtype=np.int64)

    smoke_indices = representative_indices(all_indices)
    ani_smoke_indices = representative_indices(ani_indices)
    np.save(args.smoke_indices, smoke_indices)
    np.save(args.ani_smoke_indices, ani_smoke_indices)

    element_reaction_counts: dict[str, int] = {}
    for numbers in atomic_number_rows:
        for number in sorted(set(map(int, numbers))):
            key = str(number)
            element_reaction_counts[key] = element_reaction_counts.get(key, 0) + 1
    summary = {
        "dataset_id": "DORTS-9K",
        "dataset_doi": DATASET_DOI,
        "reference_level": REFERENCE_LEVEL,
        "source_path": str(source_path),
        "source_md5": observed_md5,
        "reaction_count": len(reaction_rows),
        "record_count": total_records,
        "nms_record_count": len(record_source_kind) - int(np.count_nonzero(record_source_kind)),
        "irc_record_count": int(np.count_nonzero(record_source_kind)),
        "nms_role_counts": dict(nms_role_counts),
        "irc_subsets": subsets,
        "ani_chno_reaction_count": int(sum(bool(row["ani_chno_supported"]) for row in reaction_rows)),
        "ani_chno_record_count": int(ani_indices.size),
        "all_model_common_record_count": int(ani_indices.size),
        "natoms_min": min(int(row["natoms"]) for row in reaction_rows),
        "natoms_max": max(int(row["natoms"]) for row in reaction_rows),
        "element_reaction_counts_by_atomic_number": element_reaction_counts,
        "manifest_path": str(output_path),
    }
    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
