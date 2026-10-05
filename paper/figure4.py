"""Figure 4: empirical cumulative distributions of TS errors by reaction class on the full dataset.

Rows are reaction classes. The left column is the absolute total barrier error (a-d) and the right
column the TS force RMSE (e-h). Every error enters its ECDF. Errors below the lower axis limit set
the starting height of the curve.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, MultipleLocator, NullFormatter
import numpy as np

from common import CLASSES, COLORS, FULL_MODELS, LABELS, check_text_inside, save, ts_errors
from figure2 import LEGEND_ORDER
from figure3 import STYLE

LIMITS = {'barrier': (1e-3, 100.), 'force': (5e-3, 3.)}
FIELDS = {'barrier': 'barrier_error_kcal_mol', 'force': 'ts_force_rmse_ev_per_a'}
XLABELS = {'barrier': 'Absolute barrier error (kcal/mol)',
           'force': 'TS force RMSE (eV Å⁻¹)'}  # Arial lacks U+207B, Arial Unicode MS supplies it
SOLID, DASH, DOT, DASHDOT, LONGDASH = '-', (0, (3.2, 1.6)), (0, (.05, 1.9)), (0, (5, 1.5, 1, 1.5)), (0, (6.5, 2))
# Checkpoints of one family share a hue, so the line pattern separates them.
PATTERN = {'mace_mh1_omol': SOLID, 'mace_mh1_omat': SOLID, 'mace_mh1_spice': SOLID,
           'mace_polar_1_s': DOT, 'mace_polar_1_m': DASH, 'mace_polar_1_l': SOLID,
           'orbmol_v2': SOLID, 'uma_s_omol': DASH, 'uma_m_omol': SOLID,
           'uma_s_omc': DASH, 'uma_m_omc': SOLID, 'aimnet2_wb97m_d3': DASHDOT,
           'nep89_20250409': DASHDOT, 'sevennet_omni_i12_omol25_low': LONGDASH,
           'sevennet_omni_i12_spice': LONGDASH}
CAP = {m: 'round' if PATTERN[m] == DOT else 'butt' for m in PATTERN}
LW = 1.6
GRACE = {'font.family': ['Arial', 'Arial Unicode MS'], 'axes.linewidth': 1.0,
         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
         'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.minor.size': 2.6, 'ytick.minor.size': 2.6,
         'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.minor.width': .8, 'ytick.minor.width': .8,
         'xtick.major.pad': 4, 'ytick.major.pad': 4, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
         'axes.labelsize': 11, 'legend.fontsize': 10}


def ecdf_xy(values, lower, upper):
    """Exact ECDF as a post-step line on [lower, upper]."""
    v = np.sort(values)
    n = len(v)
    assert n and np.isfinite(v).all() and v.max() <= upper
    k = int(np.searchsorted(v, lower, side='right'))
    x = np.concatenate([[lower], v[k:], [upper]])
    y = np.concatenate([[k / n], np.arange(k + 1, n + 1) / n, [1.]])
    return x, y


def style_axis(ax, metric, show_xticklabels, show_yticklabels):
    lower, upper = LIMITS[metric]
    ax.set_xscale('log')
    ax.set_xlim(lower, upper)
    ax.set_ylim(-.015, 1.015)
    if metric == 'barrier':
        ticks = [1e-3, 1e-2, .1, 1, 10, 100]
        labels = ['0.001', '0.01', '0.1', '1', '10', '100']
    else:
        ticks = [.01, .1, 1]
        labels = ['0.01', '0.1', '1']
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels if show_xticklabels else [''] * len(ticks)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0, .2, .4, .6, .8, 1.]))
    ax.yaxis.set_major_formatter(FixedFormatter(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
                                                if show_yticklabels else [''] * 6))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.tick_params(which='both', top=True, right=True)


def draw(ax, table, metric, reaction_class):
    lower, upper = LIMITS[metric]
    counts = set()
    for model in FULL_MODELS:
        values = table.loc[table.model.eq(model) & table.reaction_class.eq(reaction_class), FIELDS[metric]]
        values = values.dropna().to_numpy()
        counts.add(len(values))
        x, y = ecdf_xy(values, lower, upper)
        ax.step(x, y, where='post', color=COLORS[model], linestyle=PATTERN[model], linewidth=LW,
                solid_capstyle='butt', dash_capstyle=CAP[model])
    assert len(counts) == 1


def main():
    plt.rcParams.update(STYLE)
    plt.rcParams.update(GRACE)
    table = ts_errors('full')
    width, height = 6.5, 7.45
    left, right, wgap = .62, .1, .2
    top, ph, hgap = .1, 1.3, .15
    pw = (width - left - right - wgap) / 2
    fig = plt.figure(figsize=(width, height))
    for i, metric in enumerate(['barrier', 'force']):
        for j, reaction_class in enumerate(CLASSES):
            bottom = height - top - (j + 1) * ph - j * hgap
            ax = fig.add_axes([(left + i * (pw + wgap)) / width, bottom / height, pw / width, ph / height])
            style_axis(ax, metric, j == len(CLASSES) - 1, i == 0)
            draw(ax, table, metric, reaction_class)
            label = f'({chr(97 + 4 * i + j)}) {reaction_class}' if metric == 'barrier' else f'({chr(97 + 4 * i + j)})'
            ax.text(.022, .935, label, transform=ax.transAxes, ha='left', va='top', fontsize=11, weight='bold')
            if j == len(CLASSES) - 1:
                ax.set_xlabel(XLABELS[metric], fontsize=11, va='baseline')
                ax.xaxis.set_label_coords(.5, -.3)
    stack_bottom = height - top - 4 * ph - 3 * hgap
    fig.text(.06 / width, (stack_bottom + (height - top - stack_bottom) / 2) / height, 'Fraction of reactions',
             rotation=90, ha='left', va='center', fontsize=11)
    handles = [Line2D([0], [0], color=COLORS[m], linestyle=PATTERN[m], linewidth=LW + .2, label=LABELS[m],
                      dash_capstyle=CAP[m]) for m in LEGEND_ORDER]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .04 / height), ncol=3,
                        frameon=True, fancybox=False, edgecolor='black', framealpha=1,
                        handlelength=3.2, handletextpad=.6, columnspacing=1.6, labelspacing=.32, borderpad=.5)
    legend.get_frame().set_linewidth(.9)
    check_text_inside(fig)
    save(fig, 'figure4')


if __name__ == '__main__':
    main()
