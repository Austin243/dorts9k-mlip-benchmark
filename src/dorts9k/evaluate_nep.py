#!/usr/bin/env python3
"""Run one resumable DORTS-9K energy/force shard with official GPUMD NEP89."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

from ase.data import chemical_symbols
import h5py
import numpy as np

from dorts9k.dorts import DORTS9KDataset, dorts_formula


STOP_REQUESTED = False
DATASET_ID = "DORTS-9K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--model-id", default="nep89_20250409")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--nep-binary", required=True)
    parser.add_argument("--expected-binary-sha256")
    parser.add_argument("--gpumd-git-commit", default="unknown")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--max-batch-atoms", type=int, default=160000)
    parser.add_argument("--vacuum-margin-a", type=float, default=8.0)
    parser.add_argument("--flush-every", type=int, default=4096)
    parser.add_argument("--max-errors", type=int, default=1)
    parser.add_argument("--work-parent")
    return parser.parse_args()


def handle_stop(signum, frame) -> None:  # noqa: ANN001
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_indices(path: str | None) -> list[int] | None:
    if path is None:
        return None
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer NumPy array")
    indices = sorted(int(value) for value in values)
    if len(indices) != len(set(indices)):
        raise ValueError("indices contain duplicates")
    return indices


def selected_indices(
    dataset_length: int,
    requested: list[int] | None,
    *,
    num_shards: int,
    shard_index: int,
) -> list[int]:
    source = range(dataset_length) if requested is None else requested
    selected = [index for index in source if index % num_shards == shard_index]
    if any(not 0 <= index < dataset_length for index in selected):
        raise IndexError("selected dataset index is outside the manifest")
    return selected


def checkpoint_prediction_input(checkpoint: Path) -> tuple[tuple[str, ...], str]:
    with checkpoint.open(encoding="utf-8") as handle:
        header = [handle.readline().split() for _ in range(7)]
    if any(not line for line in header):
        raise ValueError("incomplete NEP checkpoint header")
    first_line = header[0]
    if len(first_line) < 3 or first_line[0] != "nep4_zbl":
        raise ValueError("only nep4_zbl checkpoints are supported")
    count = int(first_line[1])
    species = tuple(first_line[2:])
    if (
        count < 1
        or len(species) != count
        or len(set(species)) != count
        or any(symbol not in chemical_symbols[1:] for symbol in species)
    ):
        raise ValueError("invalid NEP checkpoint species list")
    expected = [("zbl", 3), ("cutoff", 5), ("n_max", 3),
                ("basis_size", 3), ("l_max", 4), ("ANN", 3)]
    for line, (keyword, length) in zip(header[1:], expected, strict=True):
        if line[0] != keyword or len(line) != length:
            raise ValueError(f"unsupported NEP checkpoint {keyword} header: {line}")
    inner, outer = map(float, header[1][1:])
    radial, angular = map(float, header[2][1:3])
    n_radial, n_angular = map(int, header[3][1:])
    basis_radial, basis_angular = map(int, header[4][1:])
    l_three, l_four, l_five = map(int, header[5][1:])
    neurons, second_layer = map(int, header[6][1:])
    if not (1.0 <= outer <= 4.0 and inner == outer / 2.0):
        raise ValueError("checkpoint ZBL cutoffs cannot be represented by GPUMD zbl input")
    if not (
        3.0 <= angular <= radial <= 100.0
        and 0 <= n_radial <= 12 and 0 <= n_angular <= 8
        and 0 <= basis_radial <= 16 and 0 <= basis_angular <= 12
        and 0 <= l_three <= 8 and l_four in (0, 2) and l_five in (0, 1)
        and l_four <= l_three and not (l_four == 0 and l_five == 1)
        and 1 <= neurons <= 120 and second_layer == 0
    ):
        raise ValueError("unsupported NEP checkpoint architecture")
    prediction_input = (
        f"type {count} {' '.join(species)}\nprediction 1\nversion 4\n"
        f"zbl {outer:.17g}\ncutoff {radial:.17g} {angular:.17g}\n"
        f"n_max {n_radial} {n_angular}\nbasis_size {basis_radial} {basis_angular}\n"
        f"l_max {l_three} {l_four} {l_five}\nneuron {neurons}\n"
    )
    return species, prediction_input


def validate_output_attributes(handle: h5py.File, expected: dict[str, object]) -> None:
    for key, value in expected.items():
        if handle.attrs.get(key) != value:
            raise RuntimeError(
                f"existing output attribute {key}={handle.attrs.get(key)!r} "
                f"does not match {value!r}"
            )


def force_vector_metrics(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(np.asarray(forces, dtype=np.float64), axis=1)
    return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))


def write_prediction_xyz(path: Path, records: list[object], vacuum_margin_a: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            positions = np.asarray(record.positions, dtype=np.float64)
            lower = np.min(positions, axis=0)
            upper = np.max(positions, axis=0)
            lengths = upper - lower + 2.0 * vacuum_margin_a
            shifted = positions - lower + vacuum_margin_a
            handle.write(f"{record.natoms}\n")
            handle.write(
                "Lattice=\""
                f"{lengths[0]:.17g} 0 0 0 {lengths[1]:.17g} 0 0 0 {lengths[2]:.17g}"
                "\" energy=0 Properties=species:S:1:pos:R:3:force:R:3\n"
            )
            for atomic_number, position in zip(record.atomic_numbers, shifted, strict=True):
                symbol = chemical_symbols[int(atomic_number)]
                handle.write(
                    f"{symbol} {position[0]:.17g} {position[1]:.17g} "
                    f"{position[2]:.17g} 0 0 0\n"
                )


def parse_prediction_outputs(
    workdir: Path,
    records: list[object],
) -> tuple[np.ndarray, list[np.ndarray]]:
    energies = np.loadtxt(workdir / "energy_train.out", dtype=np.float64, ndmin=2)
    if energies.shape != (len(records), 2):
        raise ValueError(f"unexpected energy_train.out shape {energies.shape}")
    if not np.all(np.isfinite(energies)) or not np.allclose(energies[:, 1], 0.0):
        raise ValueError("NEP energy output is non-finite or dummy targets changed")
    total_atoms = sum(record.natoms for record in records)
    force_table = np.loadtxt(workdir / "force_train.out", dtype=np.float64, ndmin=2)
    if force_table.shape != (total_atoms, 6):
        raise ValueError(f"unexpected force_train.out shape {force_table.shape}")
    if not np.all(np.isfinite(force_table)) or not np.allclose(force_table[:, 3:], 0.0):
        raise ValueError("NEP force output is non-finite or dummy targets changed")

    model_energies = energies[:, 0] * np.asarray(
        [record.natoms for record in records], dtype=np.float64
    )
    model_forces: list[np.ndarray] = []
    offset = 0
    for record in records:
        stop = offset + record.natoms
        model_forces.append(force_table[offset:stop, :3].copy())
        offset = stop
    return model_energies, model_forces


def initialize_output(
    handle: h5py.File,
    args: argparse.Namespace,
    *,
    manifest_sha256: str,
    expected_record_count: int,
    source_md5: str,
    checkpoint_sha256: str,
    binary_sha256: str,
    species: tuple[str, ...],
    prediction_input: str,
) -> None:
    expected = {
        "format_version": 1,
        "dataset_id": DATASET_ID,
        "model_id": args.model_id,
        "benchmark_kind": "energy_forces",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "manifest_sha256": manifest_sha256,
        "expected_record_count": expected_record_count,
        "prediction_batch_size": args.prediction_batch_size,
        "max_batch_atoms": args.max_batch_atoms,
        "vacuum_margin_a": args.vacuum_margin_a,
        "source_md5": source_md5,
        "hessian_calculated": 0,
        "nep_prediction_input": prediction_input,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_binary_sha256": binary_sha256,
    }
    if "format_version" in handle.attrs:
        validate_output_attributes(handle, expected)
        return
    for key, value in expected.items():
        handle.attrs[key] = value
    provenance = {
        "model_id": args.model_id,
        "model_kind": "nep89",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "runtime": "GPUMD NEP",
        "runtime_binary": str(Path(args.nep_binary).resolve()),
        "runtime_binary_sha256": binary_sha256,
        "gpumd_git_commit": args.gpumd_git_commit,
        "gpumd_release": "v5.0",
        "species": list(species),
        "nep_prediction_input": prediction_input,
        "benchmark_kind": "energy_forces",
        "properties": ["energy", "forces"],
        "hessian_calculated": False,
    }
    handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    handle.attrs["manifest_path"] = str(Path(args.manifest).resolve())
    handle.attrs["source_h5"] = str(Path(args.source_h5).resolve())
    handle.attrs["model_provenance_json"] = json.dumps(provenance, sort_keys=True)
    handle.attrs["properties_json"] = json.dumps(["energy", "forces"])
    handle.attrs["benchmark_semantics"] = (
        "fixed-geometry model energies and forces compared with DORTS DFT references; "
        "no model or reference Hessians or frequencies are calculated"
    )
    handle.attrs["complete"] = 0
    handle.require_group("records")
    handle.require_group("pending")
    handle.require_group("errors")
    handle.flush()


def record_result(
    handle: h5py.File,
    record,
    *,
    model_energy_ev: float,
    model_forces_ev_per_a: np.ndarray,
    batch_size: int,
    batch_wall_s: float,
) -> None:  # noqa: ANN001
    key = f"{record.key_index:06d}"
    if key in handle["pending"]:
        del handle["pending"][key]
    group = handle["pending"].create_group(key)
    model_forces = np.asarray(model_forces_ev_per_a, dtype=np.float64)
    reference_forces = np.asarray(record.reference_forces_ev_per_a, dtype=np.float64)
    if model_forces.shape != reference_forces.shape or not np.all(np.isfinite(model_forces)):
        raise ValueError("model forces have invalid shape or non-finite values")
    if not np.isfinite(model_energy_ev):
        raise ValueError("model energy is non-finite")
    force_error = model_forces - reference_forces
    model_force_max, model_force_rms = force_vector_metrics(model_forces)
    reference_force_max, reference_force_rms = force_vector_metrics(reference_forces)
    if dorts_formula(record.atomic_numbers) != record.formula:
        raise ValueError(f"manifest formula mismatch for key {record.key_index}")
    scalar = {
        "key_index": record.key_index,
        "reaction_number": record.reaction_number,
        "reaction_id": record.reaction_id,
        "role": record.role,
        "natoms": record.natoms,
        "formula": record.formula,
        "model_energy_ev": float(model_energy_ev),
        "reference_energy_ev": float(record.reference_energy_ev),
        "raw_model_minus_reference_energy_ev": float(model_energy_ev)
        - float(record.reference_energy_ev),
        "model_force_max_vector_ev_per_a": model_force_max,
        "model_force_vector_rms_ev_per_a": model_force_rms,
        "reference_force_max_vector_ev_per_a": reference_force_max,
        "reference_force_vector_rms_ev_per_a": reference_force_rms,
        "model_reference_force_component_mae_ev_per_a": float(
            np.mean(np.abs(force_error))
        ),
        "model_reference_force_component_rmse_ev_per_a": float(
            np.sqrt(np.mean(force_error**2))
        ),
        "model_batch_size": batch_size,
        "model_batch_wall_time_s": batch_wall_s,
        "calc_time_s": batch_wall_s / batch_size,
        "source_path": record.source_path,
        "hessian_calculated": 0,
    }
    for name, value in scalar.items():
        group.attrs[name] = value
    group.create_dataset("atomic_numbers", data=record.atomic_numbers.astype(np.int16))
    group.create_dataset("positions_a", data=record.positions.astype(np.float64))
    group.create_dataset("model_forces_ev_per_a", data=model_forces.astype(np.float32))
    group.create_dataset("reference_forces_ev_per_a", data=reference_forces.astype(np.float32))
    group.attrs["complete"] = 1
    if key in handle["records"]:
        raise RuntimeError(f"record {key} appeared twice")
    handle.move(f"pending/{key}", f"records/{key}")
    if key in handle["errors"]:
        del handle["errors"][key]


def main() -> int:
    args = parse_args()
    if (
        args.num_shards < 1
        or not 0 <= args.shard_index < args.num_shards
        or args.prediction_batch_size < 1
        or args.max_batch_atoms < 1
        or args.vacuum_margin_a <= 6.0
        or args.flush_every < 1
        or args.max_errors < 1
    ):
        raise ValueError("invalid shard, batching, vacuum, flush, or error-limit option")

    checkpoint = Path(args.checkpoint).resolve()
    binary = Path(args.nep_binary).resolve()
    checkpoint_sha256 = file_sha256(checkpoint)
    binary_sha256 = file_sha256(binary)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("NEP89 checkpoint SHA256 mismatch")
    if args.expected_binary_sha256 and binary_sha256 != args.expected_binary_sha256:
        raise RuntimeError("GPUMD nep binary SHA256 mismatch")
    species, prediction_input = checkpoint_prediction_input(checkpoint)
    supported_atomic_numbers = {chemical_symbols.index(symbol) for symbol in species}

    dataset = DORTS9KDataset(args.manifest, args.source_h5)
    selection = selected_indices(
        len(dataset),
        load_indices(args.indices),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    dataset_atomic_numbers = {
        int(number)
        for numbers in dataset.reaction_atomic_numbers
        for number in np.asarray(numbers).reshape(-1)
    }
    unsupported = dataset_atomic_numbers - supported_atomic_numbers
    if unsupported:
        raise ValueError(f"NEP89 does not support DORTS atomic numbers {sorted(unsupported)}")

    expected_keys = {f"{index:06d}" for index in selection}
    manifest_sha256 = file_sha256(args.manifest)
    output_root = Path(args.outdir) / args.model_id
    output_root.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    output_path = output_root / f"{tag}.h5"
    progress_path = output_root / f"{tag}.progress.json"
    done_path = output_root / f"{tag}.done.json"
    if done_path.exists() and output_path.exists():
        with h5py.File(output_path, "r") as existing_output:
            validate_output_attributes(existing_output, {
                "nep_prediction_input": prediction_input,
                "checkpoint_sha256": checkpoint_sha256,
                "runtime_binary_sha256": binary_sha256,
                "manifest_sha256": manifest_sha256,
            })
            complete = int(existing_output.attrs.get("complete", 0)) == 1
            records = set(existing_output.get("records", {}))
            errors = len(existing_output.get("errors", {}))
            if complete and records == expected_keys and errors == 0:
                print(f"{done_path} exists; verified shard is already complete", flush=True)
                return 0
        raise RuntimeError(f"stale or inconsistent done marker: {done_path}")

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    processed = 0
    errors_this_session = 0
    started = time.time()
    work_parent = Path(args.work_parent).resolve() if args.work_parent else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "a") as output:
        initialize_output(
            output,
            args,
            manifest_sha256=manifest_sha256,
            expected_record_count=len(selection),
            source_md5=dataset.source_md5,
            checkpoint_sha256=checkpoint_sha256,
            binary_sha256=binary_sha256,
            species=species,
            prediction_input=prediction_input,
        )
        existing = set(output["records"].keys())

        def write_progress(*, force: bool = False) -> None:
            if not force and processed % args.flush_every:
                return
            output.flush()
            elapsed = time.time() - started
            atomic_json(
                progress_path,
                {
                    "status": "stopping" if STOP_REQUESTED else "running",
                    "dataset_id": DATASET_ID,
                    "model_id": args.model_id,
                    "benchmark_kind": "energy_forces",
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "expected_total": len(selection),
                    "processed_this_session": processed,
                    "completed_total": len(output["records"]),
                    "errors_total": len(output["errors"]),
                    "elapsed_s": elapsed,
                    "throughput_structures_per_s": processed / elapsed if elapsed else 0.0,
                    "prediction_batch_size": args.prediction_batch_size,
                    "max_batch_atoms": args.max_batch_atoms,
                    "hessian_calculated": False,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "host": os.uname().nodename,
                },
            )

        def register_error(record, error: Exception) -> None:  # noqa: ANN001
            nonlocal errors_this_session
            errors_this_session += 1
            key = f"{record.key_index:06d}"
            if key in output["pending"]:
                del output["pending"][key]
            error_group = output["errors"].require_group(key)
            error_group.attrs["error_repr"] = repr(error)
            error_group.attrs["traceback"] = traceback.format_exc(limit=20)
            error_group.attrs["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            output.flush()
            write_progress(force=True)
            print(f"ERROR key={record.key_index}: {error!r}", flush=True)
            if errors_this_session >= args.max_errors:
                raise RuntimeError(f"reached max errors ({args.max_errors})") from error

        def process_batch(records: list[object]) -> None:
            nonlocal processed
            if not records:
                return
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"nep89_{args.shard_index:03d}_",
                    dir=work_parent,
                ) as temporary:
                    workdir = Path(temporary)
                    (workdir / "nep.txt").symlink_to(checkpoint)
                    (workdir / "nep.in").write_text(prediction_input, encoding="utf-8")
                    write_prediction_xyz(workdir / "train.xyz", records, args.vacuum_margin_a)
                    evaluation_started = time.time()
                    completed = subprocess.run(
                        [str(binary)],
                        cwd=workdir,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    batch_wall_s = time.time() - evaluation_started
                    if completed.returncode != 0:
                        tail = completed.stdout[-12000:]
                        raise RuntimeError(
                            f"GPUMD nep exited {completed.returncode}; output tail:\n{tail}"
                        )
                    energies, forces = parse_prediction_outputs(workdir, records)
            except Exception as error:
                if len(records) == 1:
                    register_error(records[0], error)
                    return
                midpoint = len(records) // 2
                print(
                    f"BATCH_ERROR size={len(records)}: {error!r}; splitting",
                    flush=True,
                )
                process_batch(records[:midpoint])
                process_batch(records[midpoint:])
                return

            for index, record in enumerate(records):
                try:
                    record_result(
                        output,
                        record,
                        model_energy_ev=float(energies[index]),
                        model_forces_ev_per_a=forces[index],
                        batch_size=len(records),
                        batch_wall_s=batch_wall_s,
                    )
                except Exception as error:
                    register_error(record, error)
                    continue
                processed += 1
                existing.add(f"{record.key_index:06d}")
            write_progress(force=True)
            print(
                f"BATCH_COMPLETE size={len(records)} atoms={sum(r.natoms for r in records)} "
                f"wall_s={batch_wall_s:.3f} completed_total={len(output['records'])}",
                flush=True,
            )

        pending: list[object] = []
        pending_atoms = 0
        for record in dataset.iter_records(selection):
            if STOP_REQUESTED:
                break
            key = f"{record.key_index:06d}"
            if key in existing:
                continue
            if pending and (
                len(pending) >= args.prediction_batch_size
                or pending_atoms + record.natoms > args.max_batch_atoms
            ):
                process_batch(pending)
                pending = []
                pending_atoms = 0
                if STOP_REQUESTED:
                    break
            pending.append(record)
            pending_atoms += record.natoms
        if pending and not STOP_REQUESTED:
            process_batch(pending)

        natural_completion = (
            not STOP_REQUESTED
            and set(output["records"].keys()) == expected_keys
            and len(output["errors"]) == 0
        )
        if natural_completion:
            output.attrs["complete"] = 1
            output.attrs["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output.flush()
        write_progress(force=True)

    final = {
        "status": "complete" if natural_completion else "stopped",
        "dataset_id": DATASET_ID,
        "model_id": args.model_id,
        "benchmark_kind": "energy_forces",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "expected_total": len(selection),
        "processed_this_session": processed,
        "errors_this_session": errors_this_session,
        "elapsed_s": time.time() - started,
        "hessian_calculated": False,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(done_path if natural_completion else progress_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if natural_completion else 130


if __name__ == "__main__":
    raise SystemExit(main())
