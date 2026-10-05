"""Figure S3: median, P95, P99, and maximum CHNO TS errors of all seventeen checkpoints."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np

from common import CHNO_MODELS, LABELS, save, tail_statistics, ts_errors

FAMILY_COLORS = {'UMA': '#146F75', 'OrbMol': '#8A923B', 'MACE-POLAR': '#4D64A2',
                 'SevenNet': '#84628F', 'MACE-MH': '#A9653B', 'AIMNet2': '#BB6470',
                 'NEP89': '#667077', 'ANI': '#AA8747'}
STYLE = {'font.family': 'Arial', 'font.size': 8.3, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.linewidth': .65, 'axes.edgecolor': '#7C898C',
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}
QUANTILES = [('median', 'o', True, 'Median'), ('p95', 's', False, 'P95'),
             ('p99', '^', False, 'P99'), ('maximum', 'x', True, 'Maximum')]


def family(model):
    for prefix, name in [('uma_', 'UMA'), ('orbmol', 'OrbMol'), ('mace_polar', 'MACE-POLAR'),
                         ('sevennet', 'SevenNet'), ('mace_mh', 'MACE-MH'), ('aimnet', 'AIMNet2'),
                         ('ani', 'ANI')]:
        if model.startswith(prefix):
            return name
    return 'NEP89'


def main():
    plt.rcParams.update(STYLE)
    table = ts_errors('chno')
    metrics = ['ts_force_rmse_ev_per_a', 'barrier_error_kcal_mol']
    summary = {(m, metric): tail_statistics(table.loc[table.model.eq(m), metric])
               for m in CHNO_MODELS for metric in metrics}
    order = sorted(CHNO_MODELS, key=lambda m: summary[m, metrics[0]]['median'])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.45), sharey=True)
    fig.subplots_adjust(left=.265, right=.985, bottom=.115, top=.88, wspace=.22)
    for ax, metric, letter, title in zip(axes, metrics, ['a', 'b'],
                                         ['Direct-TS force errors', 'Absolute barrier errors']):
        for i, model in enumerate(order):
            color = FAMILY_COLORS[family(model)]
            stats = summary[model, metric]
            ax.plot([stats[q[0]] for q in QUANTILES], [i] * 4, color=color, alpha=.55, lw=.9)
            for statistic, marker, filled, _ in QUANTILES:
                kwargs = dict(color=color) if filled else dict(facecolors='white', edgecolors=color)
                ax.scatter(stats[statistic], i, marker=marker, s=25, linewidths=.9, zorder=4, **kwargs)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='x', color='#E4E9EA', lw=.55)
        ax.set_axisbelow(True)
        ax.set_xlabel('TS force RMSE (eV Å$^{-1}$)' if metric == metrics[0] else 'Absolute barrier error (kcal mol$^{-1}$)')
        ax.set_yticks(np.arange(17), [LABELS[m] for m in order])
        ax.set_ylim(16.6, -.6)
        ax.set_title(f'{letter}) {title}', loc='left', pad=10, fontweight='semibold')
    handles = [Line2D([0], [0], marker=mk, linestyle='none', color='black',
                      markerfacecolor='black' if fill else 'white', markersize=5, label=label)
               for _, mk, fill, label in QUANTILES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.60, .995), ncol=4, frameon=False,
               columnspacing=1.15)
    fig.canvas.draw()
    save(fig, 'figureS3')


if __name__ == '__main__':
    main()
