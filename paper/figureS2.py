"""Figure S2: ECDFs of TS errors by reaction class for the seventeen checkpoints on the CHNO subset.

The barrier axis spans the decades that hold every positive barrier error of both the full dataset
and the CHNO subset. Every error enters its ECDF.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from common import CHNO_MODELS, CLASSES, COLORS, LABELS, check_text_inside, save, ts_errors
from figure3 import STYLE

METRICS = [('barrier', 'barrier_error_kcal_mol'), ('force', 'ts_force_rmse_ev_per_a')]


def line_style(model):
    if model.startswith('mace_mh1'):
        return '-'
    if model.startswith('mace_polar'):
        return (0, (4, 1.3))
    if model.startswith('uma'):
        return (0, (6, 1.3, 1.5, 1.3))
    if model.startswith('sevennet'):
        return (0, (8, 2))
    if model.startswith('nep'):
        return (0, (5, 1.3, 1, 1.3, 1, 1.3))
    if model.startswith('aimnet') or model.startswith('orb'):
        return (0, (1, 1.2))
    return (0, (3, 2))


def barrier_limits():
    values = ts_errors('full').barrier_error_kcal_mol.dropna().to_numpy()
    values = np.concatenate([values, ts_errors('chno').barrier_error_kcal_mol.dropna().to_numpy()])
    positive = values[values > 0]
    return 10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max()))


def main():
    plt.rcParams.update(STYLE)
    table = ts_errors('chno')
    models = CHNO_MODELS
    limits = {'barrier': barrier_limits(), 'force': (.003, 3.)}
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 5.85), sharey=True)
    fig.subplots_adjust(left=.069, right=.984, top=.942, bottom=.309, wspace=.24, hspace=.48)
    for i, (metric, field) in enumerate(METRICS):
        lower, upper = limits[metric]
        for j, reaction_class in enumerate(CLASSES):
            ax = axes[i, j]
            ax.set_xscale('log')
            ax.set_xlim(lower, upper)
            ax.set_ylim(0, 1.015)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'])
            if metric == 'barrier':
                ticks, ticklabels = [1e-4, .01, 1, 100], ['$10^{-4}$', '0.01', '1', '100']
                xlabel = 'Absolute barrier error\n(kcal/mol)'
            else:
                ticks, ticklabels = [.01, .1, 1], ['0.01', '0.1', '1']
                xlabel = 'TS force RMSE' + '\n' + r'(eV Å$^{-1}$)'
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='both', labelsize=8, length=3, width=.6, pad=2)
            ax.grid(axis='y', color='#DCE1E4', linewidth=.55)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_linewidth(.65)
            ax.set_xlabel(xlabel, fontsize=8.5, labelpad=3)
            if j == 0:
                ax.set_ylabel('Fraction of reactions', fontsize=8.5, labelpad=3)
            counts = []
            for model in models:
                select = table.model.eq(model) & table.reaction_class.eq(reaction_class)
                values = np.sort(table.loc[select, field].dropna().to_numpy())
                assert len(values) and lower <= values[values > 0].min() and values.max() <= upper
                counts.append(len(values))
                zeros = int((values == 0).sum())
                x = np.concatenate([[lower], values[values > 0], [upper]])
                y = np.concatenate([[zeros / len(values)], np.arange(zeros + 1, len(values) + 1) / len(values), [1.]])
                ax.step(x, y, where='post', color=COLORS[model], linestyle=line_style(model),
                        linewidth=1.1, alpha=.98)
                if zeros:
                    ax.text(.02, .05 + models.index(model) * .035,
                            f'{LABELS[model]}: P(error=0)={zeros / len(values):.3g}',
                            transform=ax.transAxes, fontsize=6.5, color=COLORS[model])
            assert len(set(counts)) == 1
            ax.set_title(f'({chr(97 + i * 4 + j)})  {reaction_class}', loc='left', fontsize=9.4, pad=5,
                         weight='semibold')
            ax.text(.035, .94, f'n = {counts[0]:,}', transform=ax.transAxes, ha='left', va='top', fontsize=7.8)
    handles = [Line2D([0], [0], color=COLORS[m], linestyle=line_style(m), linewidth=1.4, label=LABELS[m])
               for m in models]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .011), ncol=3,
               frameon=False, fontsize=8.2, columnspacing=2.4, handlelength=3.0,
               handletextpad=.7, labelspacing=.7)
    check_text_inside(fig)
    save(fig, 'figureS2')


if __name__ == '__main__':
    main()
