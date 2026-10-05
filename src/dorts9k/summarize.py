"""Reduce saved evaluation shards to the processed tables in data/ (CPU only)."""
from collections import defaultdict
import json

import h5py
import numpy as np
import pandas as pd

from .models import MODELS, CHNO_ONLY

LOCATIONS = ['reactant','F_rev','ts','F_forw','product']
KCAL_PER_EV = 23.06054783061903


def load_manifest(path):
    with h5py.File(path, 'r') as h:
        data = {k: np.asarray(h['records'][k]) for k in ['reaction_index','source_kind','subset_code','natoms','ani_chno_supported']}
        data['names'] = h['reaction_names'].asstr()[:]
        roles = {int(k):v for k,v in json.loads(str(h.attrs['subset_names_json'])).items()}
    n = len(data['names'])
    anchors = np.full((n,5), -1, dtype=int)
    for col, name in enumerate(LOCATIONS):
        code = [k for k,v in roles.items() if v.lower() == name.lower()]
        if len(code) != 1: raise ValueError(f'Missing or ambiguous role: {name}')
        indices = np.flatnonzero((data['source_kind']==1) & (data['subset_code']==code[0]))
        rxn = data['reaction_index'][indices]
        if len(set(rxn)) != len(rxn): raise ValueError(f'Duplicate direct {name}')
        anchors[rxn,col] = indices
    data['anchors'] = anchors
    return data


def load_predictions(paths, count):
    arrays = {k:np.full(count,np.nan) for k in ['force','residual','reference','time']}
    seen = np.zeros(count,dtype=bool)
    for path in sorted(paths):
        with h5py.File(path,'r') as h:
            if not bool(h.attrs.get('complete',0)):
                raise ValueError(f'Incomplete shard: {path}')
            for key,g in h['records'].items():
                a = g.attrs
                index = int(a['key_index'])
                if not 0 <= index < count or seen[index]:
                    raise ValueError(f'Invalid or duplicated frame {index} in {path}')
                if not bool(a.get('complete',0)) or bool(a.get('hessian_calculated',0)):
                    raise ValueError(f'Incomplete or Hessian-inclusive record: {path}:{key}')
                seen[index] = True
                arrays['force'][index] = a['model_reference_force_component_rmse_ev_per_a']
                arrays['reference'][index] = a['reference_energy_ev']
                arrays['residual'][index] = a['raw_model_minus_reference_energy_ev']
                arrays['time'][index] = a['calc_time_s']
        print(f'Read {path.name}',flush=True)
    if any(not np.isfinite(a[seen]).all() for a in arrays.values()):
        raise ValueError('Non-finite saved prediction or timing')
    if np.any(arrays['force'][seen] < 0) or np.any(arrays['time'][seen] < 0):
        raise ValueError('Negative saved error or timing')
    return arrays, seen


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected finite, nonnegative errors')
    q05, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    return dict(mean=values.mean(), q05=q05, q25=q25, median=median, q75=q75, q95=q95,
                minimum=values.min(), maximum=values.max())


def reference_endpoints(reference, anchors):
    """Per reaction, the explicit endpoint with the lower DFT energy (reactant on a tie), or -1 without both."""
    reactant, product = anchors[:, 0], anchors[:, 4]
    both = (reactant >= 0) & (product >= 0)
    chosen = np.full(len(anchors), -1)
    chosen[both] = np.where(reference[reactant[both]] <= reference[product[both]], reactant[both], product[both])
    return chosen


def summarize_model(manifest, meta, model, arrays, seen):
    """Global, direct-TS, profile, and timing rows of one checkpoint.

    Relative energies are measured from the reference endpoint of each reaction, per atom in meV for
    the global statistics and in total kcal/mol for barriers and profiles.
    """
    rxn, natoms, anchors, names = (manifest[k] for k in ['reaction_index', 'natoms', 'anchors', 'names'])
    force, residual, reference = arrays['force'], arrays['residual'], arrays['reference']
    chno_reaction = np.zeros(len(names), dtype=bool)
    chno_reaction[rxn[manifest['ani_chno_supported'].astype(bool)]] = True
    endpoint = reference_endpoints(reference, anchors)
    with_reference = (endpoint >= 0)[rxn]
    relative = np.full(len(rxn), np.nan)
    relative[with_reference] = 1000 * np.abs(residual[with_reference] - residual[endpoint[rxn[with_reference]]]) / natoms[with_reference]
    cohorts = ['chno'] if model in CHNO_ONLY else ['full', 'chno']
    rows = defaultdict(list)
    for cohort in cohorts:
        in_cohort = chno_reaction if cohort == 'chno' else np.ones(len(names), dtype=bool)
        mask = in_cohort[rxn]
        if not seen[mask].all():
            raise ValueError(f'{model}/{cohort}: missing {int((mask & ~seen).sum())} configurations')
        for metric, values, selected in [('force', force, mask), ('energy', relative, mask & with_reference)]:
            rows['global_statistics'].append(dict(cohort=cohort, model=model, metric=metric,
                n_configurations=int(selected.sum()), n_reactions=len(np.unique(rxn[selected])),
                **statistics(values[selected])))
        complete = np.flatnonzero((anchors >= 0).all(axis=1) & in_cohort)
        low = endpoint[complete]
        idx = {role: anchors[complete, j] for j, role in enumerate(LOCATIONS)}
        energy = {role: np.abs(residual[idx[role]] - residual[low]) * KCAL_PER_EV for role in ['F_rev', 'ts', 'F_forw']}
        energy['reaction_energy'] = np.abs(residual[idx['product']] - residual[idx['reactant']]) * KCAL_PER_EV
        for metric, unit, values in [('force', 'eV/angstrom', {role: force[i] for role, i in idx.items()}),
                                     ('energy', 'kcal/mol', energy)]:
            for location, v in values.items():
                q25, median, q75 = np.percentile(v, [25, 50, 75])
                rows['profile_statistics'].append(dict(cohort=cohort, model=model, metric=metric, location=location,
                    n_reactions=len(complete), unit=unit, mean=v.mean(), q25=q25, median=median, q75=q75))
        if cohort == 'full':
            seconds = float(np.sum(arrays['time'][mask], dtype=np.float64))
            rows['timing'].append(dict(model=model, n_configurations=int(mask.sum()), model_calc_time_s=seconds,
                                       milliseconds_per_configuration=1000 * seconds / mask.sum()))
    ts_reactions = np.flatnonzero((anchors[:, 2] >= 0) & (chno_reaction if model in CHNO_ONLY else True))
    ts = anchors[ts_reactions, 2]
    low = endpoint[ts_reactions]
    has = low >= 0
    barrier, reference_barrier = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    barrier[has] = np.abs(residual[ts[has]] - residual[low[has]]) * KCAL_PER_EV
    reference_barrier[has] = (reference[ts[has]] - reference[low[has]]) * KCAL_PER_EV
    label = np.where(~has, '', np.where(low == anchors[ts_reactions, 0], 'reactant', 'product'))
    part = meta.loc[names[ts_reactions], ['reaction_class', 'formula']].reset_index()
    rows['ts_errors'].append(pd.DataFrame(dict(model=model, reaction_id=part.reaction_id,
        reaction_class=part.reaction_class, formula=part.formula, natoms=natoms[ts],
        chno=chno_reaction[ts_reactions].astype(int), reference_endpoint=label,
        reference_barrier_kcal_mol=reference_barrier, barrier_error_kcal_mol=barrier,
        ts_force_rmse_ev_per_a=force[ts])).sort_values('reaction_id'))
    return rows


def summarize(args):
    manifest = load_manifest(args.manifest)
    meta = pd.read_csv(args.reactions).set_index('reaction_id')
    if set(meta.index) != set(manifest['names']):
        raise ValueError('Reaction metadata does not match the manifest')
    shards = defaultdict(list)
    for path in args.runs.rglob('*.h5'):
        with h5py.File(path, 'r') as h:
            model = str(h.attrs.get('model_id', ''))
            if model in MODELS and 'records' in h:
                shards[model].append(path)
    if not shards:
        raise ValueError('No model evaluation shards found')
    tables = defaultdict(list)
    for model in [m for m in MODELS if m in shards]:
        arrays, seen = load_predictions(shards[model], len(manifest['reaction_index']))
        for name, rows in summarize_model(manifest, meta, model, arrays, seen).items():
            tables[name] += rows
        print(f'Summarized {model}', flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if name == 'ts_errors':
            pd.concat(rows, ignore_index=True).to_csv(args.outdir / 'ts_errors.csv.gz', index=False,
                                                      compression={'method': 'gzip', 'mtime': 0})
        else:
            pd.DataFrame(rows).to_csv(args.outdir / f'{name}.csv', index=False)
