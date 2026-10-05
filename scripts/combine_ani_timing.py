#!/usr/bin/env python3
"""Check all workers, then SUM single-GPU costs; do not divide by GPU count."""
from pathlib import Path
import argparse
import csv
import json
import math


def combine(folder):
    workers=[json.loads(p.read_text()) for p in sorted(folder.glob('worker_??.json'))]
    if not workers: raise ValueError('No completed worker summaries')
    count=workers[0]['worker_count']
    if sorted(w['worker_id'] for w in workers)!=list(range(count)):
        raise ValueError('Missing or duplicate workers')
    common=['worker_count','batch_size','torch_num_threads','gpu','torch_version']
    batches=[]
    for w in workers:
        if not w['complete'] or w['hessian_calculated'] or any(w[k]!=workers[0][k] for k in common):
            raise ValueError('Worker settings differ or worker is incomplete')
        with (folder/f'worker_{w["worker_id"]:02d}_batches.csv').open() as h: rows=list(csv.DictReader(h))
        if len(rows)!=w['n_batches'] or any(int(r['batch_id'])%count!=w['worker_id'] for r in rows):
            raise ValueError('Worker batch assignment mismatch')
        seconds=math.fsum(float(r['batch_wall_time_s']) for r in rows)
        if not math.isclose(seconds,w['model_calc_time_s'],rel_tol=1e-12): raise ValueError('Worker timing sum mismatch')
        if sum(int(r['n_configurations']) for r in rows)!=w['n_configurations']: raise ValueError('Worker frame count mismatch')
        batches.extend(rows)
    ids=sorted(int(r['batch_id']) for r in batches)
    if ids!=list(range(len(ids))): raise ValueError('Missing or duplicated global batches')
    n=sum(w['n_configurations'] for w in workers)
    if n!=399750: raise ValueError('Timing does not cover all 399,750 CHNO frames')
    seconds=math.fsum(w['model_calc_time_s'] for w in workers)
    result=dict(n_configurations=n,n_batches=len(batches),workers=count,model_calc_time_s=seconds,
        milliseconds_per_configuration=1000*seconds/n,hessian_calculated=False)
    (folder/'combined.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    print(json.dumps(combine(p.parse_args().directory),indent=2))
