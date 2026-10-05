"""Run one resumable, batched AIMNet2 energy/force shard on DORTS."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import h5py
import numpy as np
from dorts9k.dorts import DORTS9KDataset, dorts_formula
STOP_REQUESTED = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', choices=('aimnet2_wb97m_d3', 'aimnet2_rxn'), default='aimnet2_wb97m_d3')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source-h5', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--indices')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--structure-batch-size', type=int, default=256)
    parser.add_argument('--large-structure-natoms-threshold', type=int, default=32)
    parser.add_argument('--large-structure-batch-size', type=int, default=128)
    parser.add_argument('--torch-num-threads', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flush-every', type=int, default=256)
    parser.add_argument('--max-errors', type=int, default=1)
    return parser.parse_args()

def handle_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True

def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_indices(path: str | None) -> list[int] | None:
    if path is None:
        return None
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('indices must be a one-dimensional integer NumPy array')
    indices = sorted((int(value) for value in values))
    if len(indices) != len(set(indices)):
        raise ValueError('indices contain duplicates')
    return indices

def selected_indices(dataset_length: int, requested: list[int] | None, *, num_shards: int, shard_index: int) -> list[int]:
    source = range(dataset_length) if requested is None else requested
    selected = [index for index in source if index % num_shards == shard_index]
    if any((not 0 <= index < dataset_length for index in selected)):
        raise IndexError('selected dataset index is outside the manifest')
    return selected

def force_vector_metrics(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(np.asarray(forces, dtype=np.float64), axis=1)
    return (float(np.max(norms)), float(np.sqrt(np.mean(norms ** 2))))

class AIMNet2EnergyForceAdapter:
    """Batched energy, force, and partial-charge inference for equal-size molecules."""

    def __init__(self, *, checkpoint: str | Path, expected_sha256: str, device: str, torch_num_threads: int) -> None:
        import aimnet
        import torch
        from aimnet.calculators import AIMNet2Calculator
        source = Path(checkpoint).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = file_sha256(source)
        if observed != expected_sha256.lower():
            raise RuntimeError(f'checkpoint hash mismatch: expected={expected_sha256.lower()} observed={observed}')
        torch.set_num_threads(torch_num_threads)
        self.torch = torch
        self.device = device
        self.calculator = AIMNet2Calculator(str(source), device=device, compile_model=False)
        self.provenance = {'family': 'aimnet2', 'model_name': source.stem, 'ensemble_member': 0, 'checkpoint_path': str(source), 'checkpoint_sha256': observed, 'checkpoint_size_bytes': source.stat().st_size, 'aimnet_version': getattr(aimnet, '__version__', 'unknown'), 'torch_version': torch.__version__, 'device': device, 'compile_model': False, 'charge': 0, 'spin_multiplicity': 1, 'properties': ['energy', 'forces', 'partial_charges'], 'hessian_calculated': False, 'energy_unit': 'eV', 'force_unit': 'eV/angstrom', 'charge_unit': 'e'}

    def evaluate_batch(self, records: list[object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not records:
            raise ValueError('AIMNet2 batch must contain at least one structure')
        natoms = records[0].natoms
        if any((record.natoms != natoms for record in records)):
            raise ValueError('AIMNet2 batches must contain equal-size structures')
        data = {'coord': np.stack([np.asarray(record.positions, dtype=np.float32) for record in records]), 'numbers': np.stack([np.asarray(record.atomic_numbers, dtype=np.int64) for record in records]), 'charge': np.zeros(len(records), dtype=np.float32), 'mult': np.ones(len(records), dtype=np.float32)}
        result = self.calculator.eval(data, forces=True, hessian=False, validate_species=True)
        energies = result['energy'].detach().cpu().numpy().astype(np.float64)
        forces = result['forces'].detach().cpu().numpy().astype(np.float64)
        charges = result['charges'].detach().cpu().numpy().astype(np.float64)
        expected_force_shape = (len(records), natoms, 3)
        expected_charge_shape = (len(records), natoms)
        if energies.shape != (len(records),) or not np.all(np.isfinite(energies)):
            raise ValueError('AIMNet2 energies have invalid shape or non-finite values')
        if forces.shape != expected_force_shape or not np.all(np.isfinite(forces)):
            raise ValueError(f'AIMNet2 forces must be finite with shape {expected_force_shape}')
        if charges.shape != expected_charge_shape or not np.all(np.isfinite(charges)):
            raise ValueError(f'AIMNet2 charges must be finite with shape {expected_charge_shape}')
        return (energies, forces, charges)

    def recover_after_batch_error(self) -> None:
        if self.device.startswith('cuda') and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

def initialize_output(handle: h5py.File, args: argparse.Namespace, adapter: AIMNet2EnergyForceAdapter, *, manifest_sha256: str, expected_record_count: int, source_fingerprint: str, dataset_id: str) -> None:
    expected = {'format_version': 1, 'dataset_id': dataset_id, 'model_id': args.model_id, 'benchmark_kind': 'energy_forces', 'num_shards': args.num_shards, 'shard_index': args.shard_index, 'manifest_sha256': manifest_sha256, 'expected_record_count': expected_record_count, 'structure_batch_size': args.structure_batch_size, 'large_structure_natoms_threshold': args.large_structure_natoms_threshold, 'large_structure_batch_size': args.large_structure_batch_size, 'source_fingerprint': source_fingerprint, 'hessian_calculated': 0}
    if 'format_version' in handle.attrs:
        for key, value in expected.items():
            if handle.attrs.get(key) != value:
                raise RuntimeError(f'existing output attribute {key}={handle.attrs.get(key)!r} does not match {value!r}')
        return
    for key, value in expected.items():
        handle.attrs[key] = value
    handle.attrs['created'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    handle.attrs['manifest_path'] = str(Path(args.manifest).resolve())
    handle.attrs['source_h5'] = str(Path(args.source_h5).resolve())
    handle.attrs['reference_data_available'] = 1
    handle.attrs['model_provenance_json'] = json.dumps(adapter.provenance, sort_keys=True)
    handle.attrs['properties_json'] = json.dumps(['energy', 'forces', 'partial_charges'])
    handle.attrs['complete'] = 0
    handle.require_group('records')
    handle.require_group('pending')
    handle.require_group('errors')
    handle.flush()

def record_result(handle: h5py.File, record, *, model_energy_ev: float, model_forces_ev_per_a: np.ndarray, model_charges_e: np.ndarray, batch_size: int, batch_wall_s: float) -> None:
    key = f'{record.key_index:06d}'
    pending = handle['pending']
    records = handle['records']
    if key in pending:
        del pending[key]
    group = pending.create_group(key)
    reference_forces_raw = getattr(record, 'reference_forces_ev_per_a', None)
    reference_forces = None if reference_forces_raw is None else np.asarray(reference_forces_raw, dtype=np.float64)
    model_force_max, model_force_rms = force_vector_metrics(model_forces_ev_per_a)
    observed_formula = dorts_formula(record.atomic_numbers)
    if observed_formula != record.formula:
        raise ValueError(f'manifest formula mismatch for key {record.key_index}')
    scalar = {'key_index': record.key_index, 'source_key_index': getattr(record, 'source_key_index', record.key_index), 'reaction_number': record.reaction_number, 'reaction_id': record.reaction_id, 'role': record.role, 'natoms': record.natoms, 'formula': record.formula, 'model_energy_ev': float(model_energy_ev), 'model_force_max_vector_ev_per_a': model_force_max, 'model_force_vector_rms_ev_per_a': model_force_rms, 'model_charge_sum_e': float(np.sum(model_charges_e)), 'model_charge_sum_abs_residual_e': float(abs(np.sum(model_charges_e))), 'model_batch_size': batch_size, 'model_batch_wall_time_s': batch_wall_s, 'calc_time_s': batch_wall_s / batch_size, 'barrier_kcal_mol': record.barrier_kcal_mol, 'bond_changes': record.bond_changes, 'source_path': record.source_path, 'hessian_calculated': 0}
    if reference_forces is not None:
        force_error = np.asarray(model_forces_ev_per_a, dtype=np.float64) - reference_forces
        reference_force_max, reference_force_rms = force_vector_metrics(reference_forces)
        scalar.update({'reference_energy_ev': float(record.reference_energy_ev), 'raw_model_minus_reference_energy_ev': float(model_energy_ev) - float(record.reference_energy_ev), 'reference_force_max_vector_ev_per_a': reference_force_max, 'reference_force_vector_rms_ev_per_a': reference_force_rms, 'model_reference_force_component_mae_ev_per_a': float(np.mean(np.abs(force_error))), 'model_reference_force_component_rmse_ev_per_a': float(np.sqrt(np.mean(force_error ** 2)))})
    for name, value in scalar.items():
        group.attrs[name] = value
    group.create_dataset('atomic_numbers', data=record.atomic_numbers.astype(np.int16))
    group.create_dataset('positions_a', data=record.positions.astype(np.float64))
    group.create_dataset('model_forces_ev_per_a', data=np.asarray(model_forces_ev_per_a, dtype=np.float32))
    if reference_forces is not None:
        group.create_dataset('reference_forces_ev_per_a', data=reference_forces.astype(np.float32))
    group.create_dataset('model_atomic_charges_e', data=np.asarray(model_charges_e, dtype=np.float32))
    group.attrs['complete'] = 1
    if key in records:
        raise RuntimeError(f'record {key} appeared twice')
    handle.move(f'pending/{key}', f'records/{key}')
    if key in handle['errors']:
        del handle['errors'][key]

def main() -> int:
    args = parse_args()
    args.expected_sha256 = args.expected_sha256 or file_sha256(args.checkpoint)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('invalid shard count/index')
    positive = (args.structure_batch_size, args.large_structure_natoms_threshold, args.large_structure_batch_size, args.flush_every, args.max_errors)
    if any((value < 1 for value in positive)):
        raise ValueError('batch, threshold, flush, and error limits must be positive')
    if not args.source_h5:
        raise ValueError('DORTS-9K requires --source-h5 and no --xyz-root')
    dataset = DORTS9KDataset(args.manifest, args.source_h5)
    dataset_id = 'DORTS-9K'
    source_fingerprint = dataset.source_md5
    requested = load_indices(args.indices)
    if args.model_id == 'aimnet2_rxn':
        with h5py.File(args.manifest, 'r') as manifest:
            supported = np.asarray(manifest['records/ani_chno_supported'], dtype=bool)
        if requested is None or not np.all(supported[requested]):
            raise ValueError('AIMNet2-RXN requires CHNO-only indices for this benchmark')
    selection = selected_indices(len(dataset), requested, num_shards=args.num_shards, shard_index=args.shard_index)
    expected_keys = {f'{index:06d}' for index in selection}
    manifest_sha256 = file_sha256(args.manifest)
    output_root = Path(args.outdir) / args.model_id
    output_root.mkdir(parents=True, exist_ok=True)
    tag = f'shard_{args.shard_index:03d}_of_{args.num_shards:03d}'
    output_path = output_root / f'{tag}.h5'
    progress_path = output_root / f'{tag}.progress.json'
    done_path = output_root / f'{tag}.done.json'
    if done_path.exists() and output_path.exists():
        with h5py.File(output_path, 'r') as existing_output:
            complete = int(existing_output.attrs.get('complete', 0)) == 1
            records = set(existing_output.get('records', {}))
            errors = len(existing_output.get('errors', {}))
            if complete and records == expected_keys and (errors == 0):
                print(f'{done_path} exists; verified shard is already complete', flush=True)
                return 0
        raise RuntimeError(f'stale or inconsistent done marker: {done_path}')
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    adapter = AIMNet2EnergyForceAdapter(checkpoint=args.checkpoint, expected_sha256=args.expected_sha256, device=args.device, torch_num_threads=args.torch_num_threads)
    processed = 0
    errors_this_session = 0
    started = time.time()
    with h5py.File(output_path, 'a') as output:
        initialize_output(output, args, adapter, manifest_sha256=manifest_sha256, expected_record_count=len(selection), source_fingerprint=source_fingerprint, dataset_id=dataset_id)
        existing = set(output['records'].keys())

        def write_progress(*, force: bool=False) -> None:
            if not force and (processed + errors_this_session) % args.flush_every:
                return
            output.flush()
            atomic_json(progress_path, {'status': 'stopping' if STOP_REQUESTED else 'running', 'dataset_id': dataset_id, 'model_id': args.model_id, 'benchmark_kind': 'energy_forces', 'shard_index': args.shard_index, 'num_shards': args.num_shards, 'expected_total': len(selection), 'processed_this_session': processed, 'completed_total': len(output['records']), 'errors_total': len(output['errors']), 'elapsed_s': time.time() - started, 'structure_batch_size': args.structure_batch_size, 'large_structure_natoms_threshold': args.large_structure_natoms_threshold, 'large_structure_batch_size': args.large_structure_batch_size, 'hessian_calculated': False, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': os.uname().nodename})

        def register_error(record, error: Exception) -> None:
            nonlocal errors_this_session
            errors_this_session += 1
            key = f'{record.key_index:06d}'
            if key in output['pending']:
                del output['pending'][key]
            error_group = output['errors'].require_group(key)
            error_group.attrs['error_repr'] = repr(error)
            error_group.attrs['traceback'] = traceback.format_exc(limit=20)
            error_group.attrs['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            output.flush()
            write_progress(force=True)
            print(f'ERROR key={record.key_index}: {error!r}', flush=True)
            if errors_this_session >= args.max_errors:
                raise RuntimeError(f'reached max errors ({args.max_errors})') from error

        def process_batch(records: list[object]) -> None:
            nonlocal processed
            if not records:
                return
            evaluation_started = time.time()
            try:
                energies, forces, charges = adapter.evaluate_batch(records)
            except Exception as error:
                adapter.recover_after_batch_error()
                if len(records) > 1:
                    midpoint = len(records) // 2
                    print(f'BATCH_ERROR size={len(records)}: {error!r}; splitting', flush=True)
                    process_batch(records[:midpoint])
                    process_batch(records[midpoint:])
                    return
                register_error(records[0], error)
                return
            batch_wall_s = time.time() - evaluation_started
            for index, record in enumerate(records):
                key = f'{record.key_index:06d}'
                try:
                    record_result(output, record, model_energy_ev=energies[index], model_forces_ev_per_a=forces[index], model_charges_e=charges[index], batch_size=len(records), batch_wall_s=batch_wall_s)
                except Exception as error:
                    register_error(record, error)
                    continue
                processed += 1
                existing.add(key)
                write_progress()

        def batch_limit(natoms: int) -> int:
            if natoms >= args.large_structure_natoms_threshold:
                return args.large_structure_batch_size
            return args.structure_batch_size
        pending_by_natoms: dict[int, list[object]] = defaultdict(list)
        for record in dataset.iter_records(selection):
            if STOP_REQUESTED:
                break
            key = f'{record.key_index:06d}'
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
        record_keys = set(output['records'].keys())
        natural_completion = not STOP_REQUESTED and record_keys == expected_keys and (len(output['errors']) == 0)
        if natural_completion:
            output.attrs['complete'] = 1
            output.attrs['completed'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        output.flush()
        write_progress(force=True)
    final = {'status': 'complete' if natural_completion else 'stopped', 'dataset_id': dataset_id, 'model_id': args.model_id, 'benchmark_kind': 'energy_forces', 'shard_index': args.shard_index, 'num_shards': args.num_shards, 'expected_total': len(selection), 'processed_this_session': processed, 'errors_this_session': errors_this_session, 'elapsed_s': time.time() - started, 'hessian_calculated': False, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    atomic_json(done_path if natural_completion else progress_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if natural_completion else 130
if __name__ == '__main__':
    raise SystemExit(main())
