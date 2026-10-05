#!/usr/bin/env python3
"""Run one resumable, batched DORTS-9K energy/force-only MLIP shard."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from ase import Atoms
import h5py
import numpy as np

from dorts9k.dorts import DORTS9KDataset, dorts_formula
from dorts9k.universal_hessian import load_hessian_adapter


STOP_REQUESTED = False
DATASET_ID = "DORTS-9K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--model-kind",
        choices=("mace_polar", "mace_mp", "uma", "orbmol", "sevennet", "ani1xnr"),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--head")
    parser.add_argument("--model-name")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--uma-inference-settings", default="fast_precise")
    parser.add_argument("--sevennet-enable-flash", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--structure-batch-size", type=int, default=128)
    parser.add_argument("--large-structure-natoms-threshold", type=int, default=24)
    parser.add_argument("--large-structure-batch-size", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=256)
    parser.add_argument("--max-errors", type=int, default=1)
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


def force_vector_metrics(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(np.asarray(forces, dtype=np.float64), axis=1)
    return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))


def provenance_dict(adapter) -> dict[str, object]:  # noqa: ANN001
    provenance = adapter.provenance
    payload = provenance.as_dict() if hasattr(provenance, "as_dict") else dict(provenance)
    payload.update(
        {
            "benchmark_kind": "energy_forces",
            "properties": ["energy", "forces"],
            "hessian_calculated": False,
        }
    )
    return payload


def initialize_output(
    handle: h5py.File,
    args: argparse.Namespace,
    adapter,
    *,
    manifest_sha256: str,
    expected_record_count: int,
    source_md5: str,
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
        "structure_batch_size": args.structure_batch_size,
        "large_structure_natoms_threshold": args.large_structure_natoms_threshold,
        "large_structure_batch_size": args.large_structure_batch_size,
        "source_md5": source_md5,
        "hessian_calculated": 0,
    }
    if "format_version" in handle.attrs:
        for key, value in expected.items():
            if handle.attrs.get(key) != value:
                raise RuntimeError(
                    f"existing output attribute {key}={handle.attrs.get(key)!r} "
                    f"does not match {value!r}"
                )
        return
    for key, value in expected.items():
        handle.attrs[key] = value
    handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    handle.attrs["manifest_path"] = str(Path(args.manifest).resolve())
    handle.attrs["source_h5"] = str(Path(args.source_h5).resolve())
    handle.attrs["model_provenance_json"] = json.dumps(
        provenance_dict(adapter), sort_keys=True
    )
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
) -> None:
    key = f"{record.key_index:06d}"
    if key in handle["pending"]:
        del handle["pending"][key]
    group = handle["pending"].create_group(key)
    model_forces = np.asarray(model_forces_ev_per_a, dtype=np.float64)
    reference_forces = np.asarray(record.reference_forces_ev_per_a, dtype=np.float64)
    if model_forces.shape != reference_forces.shape or not np.all(np.isfinite(model_forces)):
        raise ValueError("model forces have invalid shape or non-finite values")
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
        "raw_model_minus_reference_energy_ev": float(model_energy_ev) - float(record.reference_energy_ev),
        "model_force_max_vector_ev_per_a": model_force_max,
        "model_force_vector_rms_ev_per_a": model_force_rms,
        "reference_force_max_vector_ev_per_a": reference_force_max,
        "reference_force_vector_rms_ev_per_a": reference_force_rms,
        "model_reference_force_component_mae_ev_per_a": float(np.mean(np.abs(force_error))),
        "model_reference_force_component_rmse_ev_per_a": float(np.sqrt(np.mean(force_error**2))),
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
    args.expected_sha256 = args.expected_sha256 or file_sha256(args.checkpoint)
    positive = (
        args.num_shards,
        args.structure_batch_size,
        args.large_structure_natoms_threshold,
        args.large_structure_batch_size,
        args.flush_every,
        args.max_errors,
    )
    if any(value < 1 for value in positive) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard, batch, flush, or error-limit option")

    dataset = DORTS9KDataset(args.manifest, args.source_h5)
    selection = selected_indices(
        len(dataset),
        load_indices(args.indices),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if args.model_kind == "ani1xnr":
        with h5py.File(args.manifest, "r") as manifest:
            supported = np.asarray(manifest["records/ani_chno_supported"], dtype=bool)
        if not np.all(supported[selection]):
            raise ValueError("ANI-1xnr requires CHNO-only indices")
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
            complete = int(existing_output.attrs.get("complete", 0)) == 1
            records = set(existing_output.get("records", {}))
            errors = len(existing_output.get("errors", {}))
            if complete and records == expected_keys and errors == 0:
                print(f"{done_path} exists; verified shard is already complete", flush=True)
                return 0
        raise RuntimeError(f"stale or inconsistent done marker: {done_path}")

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    adapter = load_hessian_adapter(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        head=args.head,
        dtype=args.dtype,
        device=args.device,
        model_name=args.model_name,
        charge=args.charge,
        spin=args.spin,
        torch_num_threads=args.torch_num_threads,
        uma_inference_settings=args.uma_inference_settings,
        uma_compile=False,
        uma_hessian_method="force_fd",
        sevennet_enable_flash=args.sevennet_enable_flash,
    )
    evaluator = getattr(adapter, "evaluate_energy_forces_batch", None)
    if not callable(evaluator):
        raise TypeError("adapter lacks energy/force-only batch evaluation")

    processed = 0
    errors_this_session = 0
    started = time.time()
    with h5py.File(output_path, "a") as output:
        initialize_output(
            output,
            args,
            adapter,
            manifest_sha256=manifest_sha256,
            expected_record_count=len(selection),
            source_md5=dataset.source_md5,
        )
        existing = set(output["records"].keys())

        def write_progress(*, force: bool = False) -> None:
            if not force and (processed + errors_this_session) % args.flush_every:
                return
            output.flush()
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
                    "elapsed_s": time.time() - started,
                    "structure_batch_size": args.structure_batch_size,
                    "large_structure_natoms_threshold": args.large_structure_natoms_threshold,
                    "large_structure_batch_size": args.large_structure_batch_size,
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
            atoms_batch = [
                Atoms(numbers=record.atomic_numbers, positions=record.positions, pbc=False)
                for record in records
            ]
            evaluation_started = time.time()
            batch_error_repr = None
            try:
                energies, forces = evaluator(atoms_batch)
                if len(energies) != len(records) or len(forces) != len(records):
                    raise RuntimeError("model batch output cardinality mismatch")
            except Exception as error:
                if len(records) == 1:
                    register_error(records[0], error)
                    return
                batch_error_repr = repr(error)
            if batch_error_repr is not None:
                # Leave the exception scope before retrying so its traceback no
                # longer pins the failed CUDA graph and allocator blocks.
                gc.collect()
                if args.device.startswith("cuda"):
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except ImportError:
                        pass
                midpoint = len(records) // 2
                print(
                    f"BATCH_ERROR size={len(records)}: {batch_error_repr}; splitting",
                    flush=True,
                )
                process_batch(records[:midpoint])
                process_batch(records[midpoint:])
                return
            batch_wall_s = time.time() - evaluation_started
            for index, record in enumerate(records):
                try:
                    record_result(
                        output,
                        record,
                        model_energy_ev=energies[index],
                        model_forces_ev_per_a=forces[index],
                        batch_size=len(records),
                        batch_wall_s=batch_wall_s,
                    )
                except Exception as error:
                    register_error(record, error)
                    continue
                processed += 1
                existing.add(f"{record.key_index:06d}")
                write_progress()

        def batch_limit(natoms: int) -> int:
            if natoms >= args.large_structure_natoms_threshold:
                return args.large_structure_batch_size
            return args.structure_batch_size

        pending_by_natoms: dict[int, list[object]] = defaultdict(list)
        for record in dataset.iter_records(selection):
            if STOP_REQUESTED:
                break
            key = f"{record.key_index:06d}"
            if key in existing:
                continue
            pending = pending_by_natoms[record.natoms]
            pending.append(record)
            if len(pending) >= batch_limit(record.natoms):
                process_batch(pending)
                pending.clear()
        if not STOP_REQUESTED:
            for natoms in sorted(pending_by_natoms):
                process_batch(pending_by_natoms[natoms])
                pending_by_natoms[natoms].clear()

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
