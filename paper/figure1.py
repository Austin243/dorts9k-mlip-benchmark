"""Figure 1: global force and relative-energy error distributions of the fifteen full-dataset checkpoints."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from common import COLORS, FULL_MODELS, LABELS, check_text_inside, read, save

POSITIONS = np.array([0, 1, 2, 3.35, 4.35, 5.35, 6.70, 8.05, 9.05,
                      10.40, 11.40, 12.75, 14.10, 15.45, 16.45])
STYLE = {'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 10, 'axes.titlesize': 11,
         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'figure.dpi': 180,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .65,
         'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
         'ytick.color': 'black'}
QUANTILES = ['minimum', 'q05', 'q25', 'median', 'q75', 'q95', 'maximum']


def global_statistics():
    """Configuration-pooled statistics; energy errors are in meV/atom."""
    table = read('global_statistics.csv')
    table = table[table.cohort.eq('full')]
    assert set(table.model) == set(FULL_MODELS) and len(table) == 30
    assert table.n_configurations.eq(table.metric.map({'force': 909077, 'energy': 865669})).all()
    q = table[QUANTILES].to_numpy(float)
    assert np.isfinite(q).all() and (q >= 0).all() and (np.diff(q, axis=1) >= 0).all()
    return table


def black_text(fig):
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            text.set_color('#000000')
            text.set_alpha(1.)


def main():
    plt.rcParams.update(STYLE)
    table = global_statistics()
    panels = [
        ('force', '(a)  Force Errors', 'Force RMSE\n(eV Å$^{-1}$) · log scale',
         (.005, 4), [.01, .1, 1], ['0.01', '0.1', '1'], '.3f'),
        ('energy', '(b)  Relative-Energy Errors',
         'Absolute relative-energy error\n(meV atom$^{-1}$) · log scale',
         (.03, 500), [.1, 1, 10, 100], ['0.1', '1', '10', '100'], '.2f')]
    fig, axes = plt.subplots(1, 2, figsize=(7, 5.2), sharey=True)
    fig.subplots_adjust(left=.25, right=.984, bottom=.15, top=.94, wspace=.15)
    for ax, (metric, title, xlabel, limits, ticks, ticklabels, fmt) in zip(axes, panels):
        selected = table[table.metric.eq(metric)].set_index('model').loc[FULL_MODELS]
        stats = [{'label': LABELS[m], 'q1': r.q25, 'med': r['median'], 'q3': r.q75,
                  'whislo': r.q05, 'whishi': r.q95, 'fliers': []} for m, r in selected.iterrows()]
        artists = ax.bxp(stats, positions=POSITIONS, widths=.58, vert=False,
                         showfliers=False, showmeans=False, patch_artist=True, manage_ticks=False,
                         medianprops={'color': '#202020', 'linewidth': 1.35},
                         whiskerprops={'color': '#646464', 'linewidth': .85},
                         capprops={'color': '#646464', 'linewidth': .85},
                         boxprops={'edgecolor': '#414141', 'linewidth': .75})
        for box, model in zip(artists['boxes'], FULL_MODELS):
            box.set_facecolor(COLORS[model])
            box.set_alpha(.88)
        for y, (_, row) in zip(POSITIONS, selected.iterrows()):
            ax.text(row.q95 * 1.12, y, format(row['median'], fmt), ha='left', va='center',
                    fontsize=8.5, fontweight='semibold', color='black', clip_on=False)
        ax.set_xscale('log')
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(POSITIONS[-1] + .62, POSITIONS[0] - .62)
        ax.set_yticks(POSITIONS)
        if metric == 'force':
            ax.set_yticklabels([LABELS[m] for m in FULL_MODELS], fontweight='semibold')
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.set_xlabel(xlabel, labelpad=5.5)
        ax.set_title(title, loc='left', fontweight='semibold', pad=8)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#4C4C4C')
        ax.tick_params(axis='x', direction='out', length=2.7, width=.65)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.grid(axis='x', color='#E4E4E4', linewidth=.6)
        ax.set_axisbelow(True)
    black_text(fig)
    check_text_inside(fig)
    save(fig, 'figure1')


if __name__ == '__main__':
    main()
