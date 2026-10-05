"""Small command surface over the production runners and saved-data analysis."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

from .models import MODELS, CHNO_ONLY


def evaluation_command(args):
    m = MODELS[args.model]
    if args.model in CHNO_ONLY and args.cohort != 'chno':
        raise ValueError(f'{args.model} must use --cohort chno')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Require num-shards > 0 and 0 <= shard-index < num-shards')
    module = {'aimnet2': 'evaluate_aimnet', 'nep89': 'evaluate_nep'}.get(m['kind'], 'evaluate')
    cmd = [sys.executable, '-m', 'dorts9k.'+module,
           '--manifest', str(args.prepared/'manifest.h5'), '--source-h5', args.source,
           '--model-id', args.model, '--checkpoint', args.checkpoint,
           '--outdir', str(args.outdir), '--num-shards', str(args.num_shards),
           '--shard-index', str(args.shard_index)]
    if args.cohort == 'chno' or args.smoke:
        indices = ('chno_' if args.cohort == 'chno' else '') + ('smoke_indices.npy' if args.smoke else 'indices.npy')
        cmd += ['--indices', str(args.prepared/indices)]
    if m['kind'] == 'nep89':
        if not args.nep_binary:
            raise ValueError('NEP89 requires --nep-binary /path/to/GPUMD/src/nep')
        cmd += ['--nep-binary', args.nep_binary, '--prediction-batch-size', '4096', '--max-batch-atoms', '160000']
    else:
        cmd += ['--structure-batch-size', str(m['batch']), '--large-structure-natoms-threshold', str(m['threshold']),
                '--large-structure-batch-size', str(m['large_batch']), '--device', args.device,
                '--torch-num-threads', '1' if args.model=='ani1xnr' else '8']
        if m['kind'] != 'aimnet2':
            cmd += ['--model-kind', m['kind'], '--dtype', m['dtype'], '--spin', str(m['spin'])]
            if m['head']: cmd += ['--head', m['head']]
            if m['model_name']: cmd += ['--model-name', m['model_name']]
            if m['kind'] == 'uma': cmd += ['--uma-inference-settings', 'fast_precise']
            if m['kind'] == 'sevennet': cmd += ['--sevennet-enable-flash']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('models', help='Print the 17 evaluated models and settings')
    p = sub.add_parser('prepare', help='Index the original DORTS-9K HDF5 file')
    p.add_argument('--source', required=True)
    p.add_argument('--outdir', type=Path, default=Path('prepared'))
    p = sub.add_parser('evaluate', help='Run energy/force evaluation in one model environment')
    p.add_argument('--model', choices=list(MODELS), required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prepared', type=Path, default=Path('prepared'))
    p.add_argument('--outdir', type=Path, default=Path('runs'))
    p.add_argument('--cohort', choices=['full','chno'], default='full')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--nep-binary')
    p.add_argument('--smoke', action='store_true', help='Use the prepared small smoke-test selection')
    p.add_argument('--dry-run', action='store_true', help='Print the command without loading a model')
    p = sub.add_parser('summarize', help='CPU-only reduction of saved evaluation shards to the processed tables')
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reactions', type=Path, default=Path('data/reaction_manifest.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/data'))
    args = parser.parse_args()
    if args.command == 'models':
        print(json.dumps(MODELS, indent=2)); return
    if args.command == 'prepare':
        args.outdir.mkdir(parents=True, exist_ok=True)
        outputs = {'output-manifest':'manifest.h5','reaction-csv':'reactions.csv','summary-json':'summary.json',
                   'ani-indices':'chno_indices.npy','all-indices':'indices.npy','smoke-indices':'smoke_indices.npy','ani-smoke-indices':'chno_smoke_indices.npy'}
        cmd = [sys.executable,'-m','dorts9k.prepare','--source-h5',args.source]
        for flag, filename in outputs.items(): cmd += ['--'+flag, str(args.outdir/filename)]
        subprocess.run(cmd, check=True); return
    if args.command == 'evaluate':
        cmd = evaluation_command(args)
        if args.dry_run:
            import shlex
            print(shlex.join(cmd))
        else:
            subprocess.run(cmd, check=True)
        return
    if args.command == 'summarize':
        from .summarize import summarize
        summarize(args)


if __name__ == '__main__':
    main()
