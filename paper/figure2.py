"""Figure 2: median global errors against energy/force evaluation time, with Pareto frontiers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from common import COLORS, FULL_MODELS, LABELS, check_text_inside, read, save
from figure1 import STYLE, black_text, global_statistics

MARKERS = dict(zip(FULL_MODELS, ['o', 'o', 'o', 's', 's', 's', 'D', '^', '^',
                                 '^', '^', 'P', 'v', 'v', 'v']))
SIZE_PATHS = [['mace_polar_1_s', 'mace_polar_1_m', 'mace_polar_1_l'],
              ['uma_s_omol', 'uma_m_omol'], ['uma_s_omc', 'uma_m_omc']]
LEGEND_ORDER = ['mace_mh1_omol', 'mace_mh1_omat', 'mace_mh1_spice', 'orbmol_v2', 'aimnet2_wb97m_d3',
                'mace_polar_1_s', 'mace_polar_1_m', 'mace_polar_1_l', 'sevennet_omni_i12_omol25_low',
                'sevennet_omni_i12_spice', 'uma_s_omol', 'uma_m_omol', 'uma_s_omc', 'uma_m_omc',
                'nep89_20250409']
# Label offsets in points for checkpoints on a frontier.
OFFSETS = {'uma_m_omol': (-6, -4, 'right'), 'uma_s_omol': (-4, -20, 'right'),
           'orbmol_v2': (-6, 17, 'right'), 'aimnet2_wb97m_d3': (5, 13, 'left')}
FRONT_COLOR = '#263746'
CONNECTOR_COLOR = '#B8BDC1'


def pareto_mask(time, error):
    return np.array([not np.any((time <= t) & (error <= e) & ((time < t) | (error < e)))
                     for t, e in zip(time, error)], dtype=bool)


def prepare_table():
    """Evaluation time is the summed per-configuration model time divided by the configuration count."""
    summary = global_statistics()
    table = read('timing.csv').set_index('model').loc[FULL_MODELS].copy()
    assert table.n_configurations.eq(909077).all()
    assert np.allclose(table.milliseconds_per_configuration,
                       1000 * table.model_calc_time_s / table.n_configurations, rtol=1e-12)
    for metric in ['force', 'energy']:
        values = summary[summary.metric.eq(metric)].set_index('model').loc[FULL_MODELS, 'median'].to_numpy()
        table[metric] = values
        table['pareto_' + metric] = pareto_mask(table.milliseconds_per_configuration.to_numpy(), values)
    return table.reset_index()


def draw_panel(ax, table, metric):
    indexed = table.set_index('model')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.065, 45)
    ax.xaxis.set_major_locator(FixedLocator([.1, .3, 1, 3, 10, 30]))
    ax.xaxis.set_major_formatter(FixedFormatter(['0.1', '0.3', '1', '3', '10', '30']))
    if metric == 'force':
        ax.set_ylim(.85, .016)
        ticks = [.02, .05, .1, .2, .5]
        title = 'a)  Force'
        ylabel = 'Median force RMSE\n(eV Å$^{-1}$)'
    else:
        ax.set_ylim(50, .45)
        ticks = [.5, 1, 2, 5, 10, 20, 50]
        title = 'b)  Relative energy'
        ylabel = 'Median absolute relative-energy error\n(meV atom$^{-1}$)'
    assert table[metric].between(min(ax.get_ylim()), max(ax.get_ylim())).all(), 'A point lies outside the axes'
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(color='#E3E6E8', linewidth=.62)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_title(title, loc='left', fontsize=11, fontweight='semibold', pad=9)
    ax.tick_params(direction='out', length=3, width=.65, labelsize=8.5)
    for path in SIZE_PATHS:
        subset = indexed.loc[path]
        ax.plot(subset.milliseconds_per_configuration, subset[metric], color=CONNECTOR_COLOR,
                linewidth=.75, linestyle=(0, (2, 2)), zorder=1)
    frontier = table[table['pareto_' + metric]].sort_values('milliseconds_per_configuration', ascending=False)
    ax.plot(frontier.milliseconds_per_configuration, frontier[metric], color=FRONT_COLOR,
            linewidth=1.25, zorder=2)
    for row in table.itertuples(index=False):
        value = getattr(row, metric)
        front = bool(getattr(row, 'pareto_' + metric))
        ax.scatter(row.milliseconds_per_configuration, value, s=59 if front else 36,
                   marker=MARKERS[row.model], facecolor=COLORS[row.model],
                   edgecolor=FRONT_COLOR if front else 'white', linewidth=1.05 if front else .65,
                   alpha=1 if front else .78, zorder=4 if front else 3)
        if not front:
            continue
        dx, dy, alignment = OFFSETS.get(row.model, (-7, 11, 'right'))
        ax.annotate(f'{LABELS[row.model]}\n{value:.3g}', xy=(row.milliseconds_per_configuration, value),
                    xytext=(dx, dy), textcoords='offset points', ha=alignment, va='center',
                    fontsize=8, fontweight='semibold', linespacing=1,
                    arrowprops={'arrowstyle': '-', 'color': '#9AA0A6', 'linewidth': .5,
                                'shrinkA': 2, 'shrinkB': 3.8},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': .25},
                    annotation_clip=False, zorder=6)


def check_labels_apart(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        labels = [t for t in ax.texts if t.get_visible() and t.get_text()]
        boxes = [t.get_bbox_patch().get_window_extent(renderer) for t in labels]
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                assert not a.overlaps(boxes[j]), f'Overlapping labels: {labels[i].get_text()}, {labels[j].get_text()}'


def main():
    plt.rcParams.update(STYLE)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
    table = prepare_table()
    fig, axes = plt.subplots(1, 2, figsize=(7, 5.15))
    fig.subplots_adjust(left=.110, right=.986, bottom=1.55/5.15, top=1-.30/5.15, wspace=.34)
    for ax, metric in zip(axes, ['force', 'energy']):
        draw_panel(ax, table, metric)
    fig.supxlabel('Energy/force evaluation time (ms per configuration)', x=.54, y=1.12/5.15, fontsize=10)
    key = [Line2D([0], [0], color=FRONT_COLOR, lw=1.25, marker='o', markersize=4.8,
                  markerfacecolor='white', markeredgewidth=.9, label='Pareto frontier'),
           Line2D([0], [0], color=CONNECTOR_COLOR, lw=.75, ls=(0, (2, 2)),
                  label='Within-family size sequence')]
    fig.legend(handles=key, loc='lower center', bbox_to_anchor=(.54, .92/5.15), ncol=2,
               frameon=False, fontsize=8, handlelength=2, columnspacing=2, borderaxespad=0)
    handles = [Line2D([0], [0], linestyle='none', marker=MARKERS[m], markersize=6,
                      markerfacecolor=COLORS[m], markeredgecolor='white', markeredgewidth=.55,
                      label=LABELS[m]) for m in LEGEND_ORDER]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .09/5.15), ncol=3,
               frameon=False, fontsize=8.5, handlelength=.9, handletextpad=.4,
               columnspacing=2, labelspacing=.32, borderaxespad=0)
    check_labels_apart(fig, axes)
    black_text(fig)
    check_text_inside(fig)
    save(fig, 'figure2')


if __name__ == '__main__':
    main()
