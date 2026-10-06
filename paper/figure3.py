"""Figures 3 and S1: median errors at the five direct anchors of each reaction.

Forces are TS-style RMSEs in eV/angstrom at all five anchors. Energies are absolute total errors in
kcal/mol relative to the reference endpoint at the three interior anchors, plus the reaction-energy
error abs[(E_model,P - E_model,R) - (E_DFT,P - E_DFT,R)].
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from common import COHORT_MODELS, LABELS, check_text_inside, read, save

FORCE_ROLES = ['reactant', 'F_rev', 'ts', 'F_forw', 'product']
ENERGY_ROLES = ['F_rev', 'ts', 'F_forw', 'reaction_energy']
FORCE_LABELS = ['Reactant\nendpoint', 'Reverse\nmax-slope', 'Transition\nstate',
                'Forward\nmax-slope', 'Product\nendpoint']
ENERGY_LABELS = ['Reverse\nmax-slope', 'Transition\nstate', 'Forward\nmax-slope',
                 'Reaction\nenergy error']
ERROR_CMAP = LinearSegmentedColormap.from_list('dorts_error_quality',
    ['#B8DED8', '#E5F0E5', '#FFF0C2', '#E79A67', '#9E413B'])
ERROR_CMAP.set_over('#7B2E2E')
FORCE_NORM = LogNorm(.01, 1.)
ENERGY_NORM = LogNorm(.1, 100.)
STYLE = {'font.family': 'Arial', 'font.size': 8.5, 'text.color': 'black',
         'axes.labelcolor': 'black', 'axes.titlecolor': 'black', 'xtick.color': 'black',
         'ytick.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}


def render(summary, models, stem, colorbar_line):
    nr = len(models)
    fig = plt.figure(figsize=(7.7, 5.30 if nr == 15 else 5.78))
    ax = fig.add_axes([.013, .154, .974, .829])
    model_width, force_width, energy_width, gap = 1.75, .55, .63, .17
    starts = [model_width, model_width + 5 * force_width + gap]
    width = starts[1] + 4 * energy_width
    ax.set_xlim(0, width)
    ax.set_ylim(nr + .02, -2.0)
    ax.axis('off')
    ax.hlines([-1.96, nr], 0, width, color='#4E5559', linewidth=.9)
    ax.hlines(0, 0, width, color='#697278', linewidth=.75)
    for boundary in [3, 6, 7, 9, 11, 12, 13, 15, 16]:
        if boundary < nr:
            ax.hlines(boundary, 0, width, color='#B8C0C5', linewidth=.55)
    ax.vlines(starts[1] - gap / 2, -1.92, nr, color='#B5BDC2', linewidth=.65)
    ax.text(.02, -.62, 'Model', va='center', fontsize=9, weight='semibold')
    for i, (n, cell_width, labels, title) in enumerate([
            (5, force_width, FORCE_LABELS, r'(a)  Force error  (eV Å$^{-1}$)'),
            (4, energy_width, ENERGY_LABELS, '(b)  Energy error  (kcal/mol)')]):
        centers = starts[i] + (np.arange(n) + .5) * cell_width
        ax.hlines(-1.18, starts[i], starts[i] + n * cell_width, color='#AEB8BE', linewidth=.55)
        ax.text(centers.mean(), -1.57, title, ha='center', va='center', fontsize=10, weight='semibold')
        ts_col = 2 if i == 0 else 1
        ax.vlines([starts[i] + ts_col * cell_width, starts[i] + (ts_col + 1) * cell_width],
                  -1.14, nr, color='#B7A985', linewidth=.55)
        for j, label in enumerate(labels):
            ax.text(centers[j], -.60, label, ha='center', va='center', fontsize=8,
                    linespacing=1.05, weight='semibold' if j == ts_col else 'normal')
    for i, model in enumerate(models):
        ax.text(.02, i + .5, LABELS[model], va='center', fontsize=8.5)
    for block, metric, roles, norm, cell_width in [
            (0, 'force', FORCE_ROLES, FORCE_NORM, force_width),
            (1, 'energy', ENERGY_ROLES, ENERGY_NORM, energy_width)]:
        values = (summary[summary.metric.eq(metric)]
                  .pivot(index='model', columns='location', values='median')
                  .reindex(index=models, columns=roles).to_numpy())
        assert np.isfinite(values).all()
        for j in range(len(roles)):
            center = starts[block] + (j + .5) * cell_width
            minimum = values[:, j].min()
            for i in range(nr):
                value = float(values[i, j])
                best = value == minimum and value > 0
                rgba = (.94, .94, .94, 1.) if value == 0 else ERROR_CMAP(norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                color = 'white' if luminance < .48 else 'black'
                pw = cell_width * .88
                ax.add_patch(FancyBboxPatch((center - pw / 2, i + .17), pw, .66,
                             boxstyle='round,pad=.007,rounding_size=.025', facecolor=rgba,
                             edgecolor='#123F4A' if best else (1, 1, 1, .75),
                             linewidth=1.45 if best else .5))
                ax.text(center, i + .5, f'{value:.3g}', ha='center', va='center',
                        fontsize=8.5, color=color, weight='bold' if best else 'normal')
    pos = ax.get_position()
    for start, n, cw, norm, ticks, label in [
            (starts[0], 5, force_width, FORCE_NORM, [.01, .03, .1, .3, 1], r'Force RMSE (eV Å$^{-1}$)'),
            (starts[1], 4, energy_width, ENERGY_NORM, [.1, 1, 10, 100], 'Energy error (kcal/mol)')]:
        cax = fig.add_axes([pos.x0 + pos.width * start / width, .103, pos.width * n * cw / width, .014])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=ERROR_CMAP), cax=cax,
                           orientation='horizontal', extend='max', extendfrac=.035)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor('face')
        bar.ax.xaxis.set_major_locator(FixedLocator(ticks))
        bar.ax.xaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in ticks]))
        bar.ax.xaxis.set_minor_locator(NullLocator())
        bar.ax.tick_params(labelsize=8, length=1.8, width=colorbar_line, pad=1.3)
        bar.outline.set_linewidth(colorbar_line)
        bar.set_label(label + ' · log color scale', fontsize=8, labelpad=2)
    fig.text(.014, .016, f'Medians across {int(summary.n_reactions.iloc[0]):,} reactions; direct anchors only.',
             fontsize=8, va='bottom')
    check_text_inside(fig)
    save(fig, stem)
    plt.close(fig)


def main():
    plt.rcParams.update(STYLE)
    statistics = read('profile_statistics.csv')
    # Figure 3 is printed at 7 in, so its colorbar ticks and outlines are 0.6 pt and stay above 0.5 pt.
    for cohort, n_reactions, stem, colorbar_line in [('full', 8474, 'figure3', .6), ('chno', 3706, 'figureS1', .5)]:
        summary = statistics[statistics.cohort.eq(cohort)]
        models = COHORT_MODELS[cohort]
        assert set(summary.model) == set(models) and summary.n_reactions.eq(n_reactions).all()
        assert len(summary) == 9 * len(models)
        render(summary, models, stem, colorbar_line)


if __name__ == '__main__':
    main()
