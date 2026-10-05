"""Checkpoint order, labels, colors, data access, and output paths shared by the paper scripts."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGURES = ROOT / 'results' / 'figures'
TABLES = ROOT / 'results' / 'tables'
KCAL_PER_EV = 23.06054783061903

FULL_MODELS = ['mace_mh1_omol', 'mace_mh1_omat', 'mace_mh1_spice',
               'mace_polar_1_s', 'mace_polar_1_m', 'mace_polar_1_l', 'orbmol_v2',
               'uma_s_omol', 'uma_m_omol', 'uma_s_omc', 'uma_m_omc',
               'aimnet2_wb97m_d3', 'nep89_20250409',
               'sevennet_omni_i12_omol25_low', 'sevennet_omni_i12_spice']
CHNO_MODELS = FULL_MODELS + ['ani1xnr', 'aimnet2_rxn']
LABELS = dict(zip(CHNO_MODELS, [
    'MACE-MH-1 OMol', 'MACE-MH-1 OMat', 'MACE-MH-1 SPICE',
    'MACE-POLAR-1-S', 'MACE-POLAR-1-M', 'MACE-POLAR-1-L', 'OrbMol-v2',
    'UMA-S OMol', 'UMA-M OMol', 'UMA-S OMC', 'UMA-M OMC', 'AIMNet2', 'NEP89',
    'SevenNet-i12 OMol25-low', 'SevenNet-i12 SPICE', 'ANI-1xnr', 'AIMNet2-RXN']))
COLORS = dict(zip(CHNO_MODELS, [
    '#2F6F9F', '#B45F06', '#D9A441', '#C59AD4', '#9866B3', '#6F3C97', '#4F7A38',
    '#E39AAF', '#B84268', '#76C6BA', '#238C80', '#5F6B75', '#8A7D2D', '#4C78A8', '#C17A47',
    '#AA8747', '#BB6470']))
CLASSES = ['Addition', 'Substitution', 'Cyclization', 'Rearrangement']
COHORT_MODELS = {'full': FULL_MODELS, 'chno': CHNO_MODELS}


def read(name):
    return pd.read_csv(DATA / name, float_precision='round_trip')


def ts_errors(cohort):
    """One direct transition state per reaction and checkpoint.

    The barrier error is the absolute total error in kcal/mol relative to the reference endpoint.
    It is missing when a reaction lacks an explicit reactant or product.
    """
    table = read('ts_errors.csv.gz')
    if cohort == 'chno':
        table = table[table.chno.eq(1)]
    table = table[table.model.isin(COHORT_MODELS[cohort])]
    assert not table.duplicated(['model', 'reaction_id']).any()
    assert set(table.model) == set(COHORT_MODELS[cohort])
    assert table.groupby('model').size().nunique() == 1
    assert table.groupby('model').barrier_error_kcal_mol.count().nunique() == 1
    return table.reset_index(drop=True)


def tail_statistics(values):
    """Linear-interpolation quantiles. The worst 1% holds the largest ceil(0.01 n) errors."""
    a = np.sort(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    assert len(a) and (a >= 0).all()
    k = int(np.ceil(.01 * len(a)))
    return dict(n_reactions=len(a), minimum=a[0], median=np.median(a), mean=a.mean(),
                p95=np.quantile(a, .95), p99=np.quantile(a, .99), maximum=a[-1],
                worst_1pct_mean=a[-k:].mean(), worst_1pct_count=k,
                worst_1pct_error_share_percent=100 * a[-k:].sum() / a.sum())


def check_text_inside(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_text() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
            outside.append(text.get_text())
    assert not outside, f'Text outside the figure canvas: {outside}'


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(FIGURES / f'{stem}.{extension}', dpi=600, facecolor='white')
    print('Wrote', FIGURES / f'{stem}.png')
