#!/usr/bin/env python3
"""ANI's E/F-only timing protocol: disjoint whole batches, one GPU per worker.

Use a fresh output directory. Timed calls are not warmed up; the first call is
retained. No Hessian is requested. See docs/timing.md for accounting.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import math
import time

import h5py
import numpy as np
from ase import Atoms
from dorts9k.dorts import DORTS9KDataset
from dorts9k.universal_hessian import ANI1xnrHessianAdapter, file_sha256


def build_batch_plan(indices,natoms,batch_size):
    pending=defaultdict(list);batches=[]
    for index in indices:
        group=pending[int(natoms[index])];group.append(int(index))
        if len(group)==batch_size:
            batches.append(np.asarray(group,dtype=np.int32));group.clear()
    for n in sorted(pending):
        if pending[n]: batches.append(np.asarray(pending[n],dtype=np.int32))
    return batches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-h5',type=Path,required=True)
    p.add_argument('--model-info',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--worker-count',type=int,default=8)
    p.add_argument('--worker-id',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--torch-threads',type=int,default=1)
    a=p.parse_args()
    if a.worker_count<1 or not 0<=a.worker_id<a.worker_count or a.batch_size<1 or a.torch_threads<1:
        p.error('Invalid worker, batch, or thread setting')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Expose exactly one GPU per worker')
    torch.set_num_threads(a.torch_threads)
    with h5py.File(a.manifest,'r') as h:
        flags=np.asarray(h['records/ani_chno_supported'],dtype=bool)
        natoms=np.asarray(h['records/natoms'])
        reaction=np.asarray(h['records/reaction_index'])
        atomic_flags=np.asarray([np.isin(z,[1,6,7,8]).all() for z in h['reaction_atomic_numbers']])
    if not np.array_equal(flags,atomic_flags[reaction]): raise ValueError('Incorrect CHNO flags')
    indices=np.flatnonzero(flags)
    if len(indices)!=399750 or len(np.unique(reaction[indices]))!=3995:
        raise ValueError('This protocol requires the complete published CHNO frame set')
    batches=build_batch_plan(indices,natoms,a.batch_size)
    assigned=[(b,ids) for b,ids in enumerate(batches) if b%a.worker_count==a.worker_id]
    if not assigned: raise ValueError('Worker has no batches')
    a.outdir.mkdir(parents=True,exist_ok=True)
    csv_path=a.outdir/f'worker_{a.worker_id:02d}_batches.csv'
    if csv_path.exists(): raise FileExistsError('Use a new output directory for a new timing trial')
    adapter=ANI1xnrHessianAdapter(model_info=a.model_info,expected_sha256=file_sha256(a.model_info),device='cuda',dtype='float64')
    if len(adapter.model)!=8: raise ValueError('Expected the eight-member ANI-1xnr ensemble')
    rows=[]
    with DORTS9KDataset(a.manifest,a.source_h5) as dataset, csv_path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['batch_id','natoms','n_configurations','batch_wall_time_s'])
        writer.writeheader()
        for batch_id,ids in assigned:
            records=[dataset.record(int(i)) for i in ids]
            atoms=[Atoms(numbers=r.atomic_numbers,positions=r.positions,pbc=False) for r in records]
            torch.cuda.synchronize();start=time.perf_counter()
            energies,forces=adapter.evaluate_energy_forces_batch(atoms)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            del energies,forces
            row=dict(batch_id=batch_id,natoms=len(atoms[0]),n_configurations=len(ids),batch_wall_time_s=seconds)
            rows.append(row);writer.writerow(row);handle.flush()
    total=math.fsum(r['batch_wall_time_s'] for r in rows)
    count=sum(r['n_configurations'] for r in rows)
    info=dict(worker_id=a.worker_id,worker_count=a.worker_count,n_batches=len(rows),n_configurations=count,
        model_calc_time_s=total,milliseconds_per_configuration=1000*total/count,
        batch_size=a.batch_size,torch_num_threads=a.torch_threads,gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,hessian_calculated=False,complete=True)
    (a.outdir/f'worker_{a.worker_id:02d}.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__': main()
