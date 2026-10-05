"""Table 2 and the computed Supporting Information tables (S1-S5, S7), rounded as in the paper."""
import numpy as np
import pandas as pd

from common import CHNO_MODELS, CLASSES, LABELS, TABLES, read, tail_statistics, ts_errors

FORCE, BARRIER = 'ts_force_rmse_ev_per_a', 'barrier_error_kcal_mol'
REPLICATES, SEED = 2000, 20260909
NITROUS_ACID = [f'NO2-A-{i}-c' for i in range(1, 6)]


def sig(value, digits=3):
    """Round to significant figures and keep trailing zeros, e.g. 0.909, 3.60, 15.3, 1.00."""
    if value == 0:
        return '0'
    rounded = float(f'{value:.{digits}g}')
    places = digits - 1 - int(np.floor(np.log10(abs(rounded))))
    return f'{rounded:.{max(0, places)}f}'


def write(name, frame):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f'{name}.csv', index=False)
    print(f'\n{name}\n{frame.to_string(index=False)}')


def table2(data):
    """Barrier MAE, the MAE as a percentage of the mean DFT barrier, and the mean TS force RMSE."""
    barriers = data[data.model.eq('uma_m_omol')].reference_barrier_kcal_mol.dropna()
    assert len(barriers) == 3713
    mean_barrier = barriers.mean()
    rows = []
    for model in CHNO_MODELS:
        part = data[data.model.eq(model)]
        mae = part[BARRIER].mean()
        rows.append({'Checkpoint': LABELS[model], 'Total barrier MAE (kcal/mol)': f'{mae:.3f}',
                     'Total barrier MAE (%)': f'{100 * mae / mean_barrier:.2f}',
                     'Mean TS force RMSE (eV/Å)': f'{part[FORCE].mean():.3f}'})
    print(f'Mean DFT barrier of the {len(barriers):,} reactions: {mean_barrier:.1f} kcal/mol')
    write('table2', pd.DataFrame(rows))


def table_s1():
    stats = read('global_statistics.csv')
    stats = stats[stats.cohort.eq('chno')].set_index(['model', 'metric'])
    rows = []
    for model in CHNO_MODELS:
        row = {'Checkpoint': LABELS[model]}
        for metric, name in [('force', 'Force'), ('energy', 'Energy')]:
            for column, statistic in [('mean', 'mean'), ('median', 'median'), ('P95', 'q95'), ('max.', 'maximum')]:
                row[f'{name} {column}'] = sig(stats.loc[(model, metric), statistic])
        rows.append(row)
    write('tableS1', pd.DataFrame(rows))


def table_s2(data):
    rows = []
    for reaction_class in CLASSES:
        for model in CHNO_MODELS:
            part = data[data.model.eq(model) & data.reaction_class.eq(reaction_class)]
            barrier, force = part[BARRIER].dropna().to_numpy(), part[FORCE].to_numpy()
            rows.append({'Reaction class': reaction_class, 'Checkpoint': LABELS[model],
                         'Barrier reactions': len(barrier), 'Force reactions': len(force),
                         'Barrier MAE': sig(barrier.mean()), 'Barrier median': sig(np.median(barrier)),
                         'Barrier P95': sig(np.quantile(barrier, .95)), 'Barrier P99': sig(np.quantile(barrier, .99)),
                         'Within 1 kcal/mol (%)': sig(100 * (barrier <= 1).mean()),
                         'Force mean': sig(force.mean()), 'Force median': sig(np.median(force)),
                         'Force P95': sig(np.quantile(force, .95))})
    write('tableS2', pd.DataFrame(rows))


def bootstrap(data):
    """Pointwise percentile 95% intervals from 2,000 paired resamples of reactions.

    One draw of reaction indices is shared by all checkpoints within a metric. Force is resampled
    first and barrier second from one continuous random stream. Returns the sorted checkpoint ids and,
    per metric, the statistic names, estimates, lower and upper bounds, and the number of reactions.
    """
    rng = np.random.default_rng(SEED)
    models = sorted(CHNO_MODELS)
    results = {}
    for metric, names in [(FORCE, ['mean', 'median']),
                          (BARRIER, ['MAE', 'median', 'P99', 'within 1 kcal/mol (%)'])]:
        wide = data.pivot(index='reaction_id', columns='model', values=metric)[models].sort_index().dropna()
        values = wide.to_numpy()
        n = len(values)

        def summarize(x):
            out = [x.mean(axis=0), np.median(x, axis=0)]
            if metric == BARRIER:
                out += [np.quantile(x, .99, axis=0, method='linear'), 100 * (x <= 1.0).mean(axis=0)]
            return np.stack(out)

        estimate = summarize(values)
        draws = np.empty((REPLICATES, len(names), len(models)))
        for i in range(REPLICATES):
            draws[i] = summarize(values[rng.integers(0, n, size=n)])
        lower, upper = np.quantile(draws, [.025, .975], axis=0, method='linear')
        results[metric] = (names, estimate, lower, upper, n)
    return models, results


def table_s3(data):
    models, results = bootstrap(data)
    for metric, part in [(FORCE, 'B'), (BARRIER, 'A')]:
        names, estimate, lower, upper, n = results[metric]
        rows = []
        for model in CHNO_MODELS:
            j = models.index(model)
            row = {'Checkpoint': LABELS[model]}
            for k, name in enumerate(names):
                row[f'{name} [95% CI]'] = f'{sig(estimate[k, j])} [{sig(lower[k, j])}, {sig(upper[k, j])}]'
            rows.append(row)
        print(f'\nTable S3 part {part}: {n:,} reactions per checkpoint')
        write(f'tableS3_part{part}', pd.DataFrame(rows))


def table_s4(data):
    for metric, part in [(FORCE, 'A'), (BARRIER, 'B')]:
        rows = []
        for model in CHNO_MODELS:
            s = tail_statistics(data.loc[data.model.eq(model), metric])
            rows.append({'Checkpoint': LABELS[model], 'Reactions': s['n_reactions'],
                         **{c: sig(s[k]) for c, k in [('Minimum', 'minimum'), ('Median', 'median'), ('Mean', 'mean'),
                                                      ('P95', 'p95'), ('P99', 'p99'), ('Maximum', 'maximum'),
                                                      ('Worst 1% mean', 'worst_1pct_mean'),
                                                      ('Worst 1% error share (%)', 'worst_1pct_error_share_percent')]}})
        write(f'tableS4_part{part}', pd.DataFrame(rows))


def table_s5(data):
    """UMA-M OMol against AIMNet2-RXN overall and on each checkpoint's own worst 1%."""
    worst = {}
    for metric, part in [(FORCE, 'A'), (BARRIER, 'B')]:
        pair = data.pivot(index='reaction_id', columns='model', values=metric)[['uma_m_omol', 'aimnet2_rxn']].dropna()
        u, r = pair.uma_m_omol.to_numpy(), pair.aimnet2_rxn.to_numpy()
        k = int(np.ceil(.01 * len(pair)))
        own = {'UMA': np.argsort(u)[-k:], 'RXN': np.argsort(r)[-k:]}
        worst[metric] = {name: set(pair.index[idx]) for name, idx in own.items()}
        rows = []
        for selection, idx in [('All reactions', np.arange(len(pair))), ('UMA own worst 1%', own['UMA']),
                               ('RXN own worst 1%', own['RXN'])]:
            rows.append({'Selection': selection, 'N': len(idx), 'UMA mean': sig(u[idx].mean()),
                         'RXN mean': sig(r[idx].mean()), 'UMA median': sig(np.median(u[idx])),
                         'RXN median': sig(np.median(r[idx])), 'UMA smaller count': int((u[idx] < r[idx]).sum()),
                         'RXN smaller count': int((r[idx] < u[idx]).sum())})
        write(f'tableS5_part{part}', pd.DataFrame(rows))
    print('Shared worst-1% reactions: force', len(worst[FORCE]['UMA'] & worst[FORCE]['RXN']),
          ', barrier', len(worst[BARRIER]['UMA'] & worst[BARRIER]['RXN']),
          ', UMA force and barrier', len(worst[FORCE]['UMA'] & worst[BARRIER]['UMA']))

    signed = read('largest_uma_barrier_errors_signed.csv').set_index('reaction_id')
    assert set(signed.index) == worst[BARRIER]['UMA']
    stored = data[data.reaction_id.isin(signed.index)].pivot(index='reaction_id', columns='model', values=BARRIER)
    assert np.allclose(signed[CHNO_MODELS].abs(), stored.loc[signed.index, CHNO_MODELS], rtol=0, atol=1e-8)
    template = signed.index.str.split('_', n=3).str[-1]
    groups = {'Nitrous acid median (23)': template.isin(NITROUS_ACID), 'RR-B-4-b median (12)': template == 'RR-B-4-b'}
    assert [int(g.sum()) for g in groups.values()] == [23, 12]
    rows = []
    for model in CHNO_MODELS:
        values = signed[model].to_numpy()
        rows.append({'Checkpoint': LABELS[model], 'Median': sig(np.median(values)), 'Minimum': sig(values.min()),
                     'Maximum': sig(values.max()), 'MAE': sig(np.abs(values).mean()),
                     'Negative (of 38)': int((values < 0).sum()),
                     **{name: sig(np.median(values[mask])) for name, mask in groups.items()}})
    write('tableS5_partC', pd.DataFrame(rows))


def table_s7(data):
    """Reactions in two recurring templates where AIMNet2-RXN has the smaller error."""
    data = data.assign(template=data.reaction_id.str.split('_', n=3).str[-1])
    rows = []
    for template in ['NO2-A-1-c', 'RR-B-4-b']:
        part = data[data.template.eq(template)]
        counts = {}
        for metric in [FORCE, BARRIER]:
            pair = part.pivot(index='reaction_id', columns='model', values=metric)[['uma_m_omol', 'aimnet2_rxn']].dropna()
            counts[metric] = int((pair.aimnet2_rxn < pair.uma_m_omol).sum())
        rows.append({'Template': template, 'Reactions': part.reaction_id.nunique(),
                     'Lower force error': counts[FORCE], 'Lower barrier error': counts[BARRIER]})
    write('tableS7', pd.DataFrame(rows))


def main():
    data = ts_errors('chno')
    table2(data)
    table_s1()
    table_s2(data)
    table_s3(data)
    table_s4(data)
    table_s5(data)
    table_s7(data)


if __name__ == '__main__':
    main()
