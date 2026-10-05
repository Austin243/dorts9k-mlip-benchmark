"""Interactive versions of the figures, written as a small website to results/site.

Every data figure becomes a Plotly page with hover readouts and zoom, Table 2 becomes a bar chart
with its bootstrap intervals, and Figure 5 becomes a page of 3D molecule viewers (3Dmol.js). GitHub
Pages publishes the site from the main branch through .github/workflows/pages.yml.
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import CHNO_MODELS, CLASSES, COHORT_MODELS, COLORS, DATA, FULL_MODELS, LABELS, ROOT, read, tail_statistics, ts_errors
from figure2 import CONNECTOR_COLOR, FRONT_COLOR, LEGEND_ORDER, MARKERS, SIZE_PATHS, prepare_table
from figure4 import LEGENDS, PATTERN
from figureS3 import FAMILY_COLORS, family
from tables import BARRIER, FORCE, bootstrap

SITE = ROOT / 'results' / 'site'
REPO = 'https://github.com/Austin243/dorts9k-mlip-benchmark'
FONT = 'Inter, "Helvetica Neue", Arial, sans-serif'
INK, GRID, AXIS = '#0f172a', '#e9edf2', '#cbd5e1'
SYMBOLS = {'o': 'circle', 's': 'square', 'D': 'diamond', '^': 'triangle-up', 'v': 'triangle-down', 'P': 'cross'}
DASHES = {'-': 'solid', (0, (3.2, 1.6)): 'dash', (0, (.05, 1.9)): 'dot',
          (0, (5, 1.5, 1, 1.5)): 'dashdot', (0, (6.5, 2)): 'longdash'}
LINE_DASH = {m: DASHES[p] for m, p in PATTERN.items()}
ERROR_SCALE = ['#B8DED8', '#E5F0E5', '#FFF0C2', '#E79A67', '#9E413B']
ATOMS = {'C': ('#555B66', .36), 'H': ('#EEF0F3', .24), 'N': ('#4267C8', .35), 'O': ('#D44E4B', .35)}
MODEL_COLORS = {'uma_m_omol': '#146F75', 'aimnet2_rxn': '#BB6470'}
FORCE_UNIT, ENERGY_UNIT = 'eV/Å', 'kcal/mol'
AXIS_STYLE = dict(showgrid=True, gridcolor=GRID, linecolor=AXIS, ticks='outside', ticklen=4, tickcolor=AXIS,
                  zeroline=False, automargin=True, title_font=dict(color='#334155'))
TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK), paper_bgcolor='white', plot_bgcolor='white',
    hoverlabel=dict(bgcolor='white', bordercolor=AXIS, font=dict(family=FONT, size=13, color=INK)),
    legend=dict(font=dict(size=12), itemclick='toggle', itemdoubleclick='toggleothers', groupclick='togglegroup'),
    xaxis=AXIS_STYLE, yaxis=AXIS_STYLE))

# Each page: stem, label, title, one-line summary for the gallery, caption, whether it has a legend.
PAGES = [
    ('figure1', 'Figure 1', 'Global error distributions',
     'Force and per-atom energy errors over all 909,077 configurations',
     'Force RMSE (a) and absolute per-atom relative-energy error (b) of fifteen checkpoints over all DORTS-9K '
     'configurations. Boxes span the 25th to 75th percentiles and whiskers the 5th to 95th percentiles.', False),
    ('figure2', 'Figure 2', 'Accuracy and evaluation time',
     'Median errors against the cost of an energy and force call',
     'Median force RMSE (a) and median absolute per-atom relative-energy error (b) against the mean energy-and-force '
     'evaluation time per configuration on one A100 GPU. The solid line is the Pareto frontier, and dotted lines join '
     'model sizes within a family.', True),
    ('figure3', 'Figure 3', 'Errors along the reaction profile',
     'Median errors at the five direct anchors of each reaction',
     'Median force RMSE and median absolute total energy error at the five direct anchors of the 8,474 reactions that '
     'have all five. Energies are measured from the reference endpoint, and the last column is the error in the '
     'reaction energy. Outlined cells mark the smallest value in each column.', False),
    ('figure4', 'Figure 4', 'Transition-state errors by reaction class',
     'Cumulative distributions of barrier and transition-state force errors',
     'Cumulative distributions of the absolute total barrier error (a to d) and the transition-state force RMSE '
     '(e to h) of fifteen checkpoints on the full dataset. Hovering over a curve names the reaction at each step.', True),
    ('table2', 'Table 2', 'Mean transition-state errors on the CHNO subset',
     'Barrier MAE and TS force RMSE of all seventeen checkpoints with 95% intervals',
     'Total barrier MAE (a) and mean transition-state force RMSE (b) of seventeen checkpoints on the 3,713 and 3,871 '
     'CHNO reactions, sorted by barrier MAE. Error bars are the pointwise 95% bootstrap intervals of Table S3. '
     'Hovering also gives the MAE as a percentage of the mean DFT barrier, 62.9 kcal/mol.', False),
    ('figure5', 'Figure 5', 'Reactions with large UMA-M OMol errors',
     'Six CHNO reactions in 3D, from reactant through transition state to product',
     'Six CHNO reactions with large UMA-M OMol errors, with the transition-state force RMSE and absolute total barrier '
     'error of UMA-M OMol and AIMNet2-RXN. Dashed lines mark contacts that form or break, and their lengths are listed '
     'under each view for the reactant, transition state, and product.', '3d'),
    ('figureS1', 'Figure S1', 'Reaction-profile errors on the CHNO subset',
     'Figure 3 for seventeen checkpoints on the CHNO reactions',
     'The same comparison as Figure 3 for seventeen checkpoints on the 3,706 CHNO reactions with all five anchors.', False),
    ('figureS2', 'Figure S2', 'Transition-state errors on the CHNO subset',
     'Figure 4 for seventeen checkpoints on the CHNO reactions',
     'The same comparison as Figure 4 for seventeen checkpoints on the CHNO reactions.', True),
    ('figureS3', 'Figure S3', 'Error tails on the CHNO subset',
     'Median, P95, P99, and maximum errors of every checkpoint',
     'Median, P95, P99, and maximum transition-state force RMSE (a) and absolute total barrier error (b) of seventeen '
     'checkpoints on the CHNO subset, ordered by the median force error.', True),
]
TIPS = {True: ['Hover for values', 'Click a legend entry to hide it', 'Double-click an entry to isolate it',
               'Drag to zoom, double-click to reset'],
        False: ['Hover for values', 'Drag to zoom', 'Double-click to reset'],
        '3d': ['Drag to rotate', 'Scroll to zoom', 'Hover over an atom to name it', 'Play steps from reactant to product']}


def style(fig, height):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=70, r=20, t=60, b=60))
    fig.update_annotations(font=dict(size=15, color=INK, weight=600))
    for update in [fig.update_xaxes, fig.update_yaxes]:
        update(selector=dict(type='log'), dtick=1, exponentformat='none')
    return fig


def figure1():
    stats = read('global_statistics.csv')
    stats = stats[stats.cohort.eq('full')]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.04,
                        subplot_titles=[f'(a) Force RMSE ({FORCE_UNIT})', '(b) Relative-energy error (meV/atom)'])
    for col, metric in enumerate(['force', 'energy'], 1):
        table = stats[stats.metric.eq(metric)].set_index('model').loc[FULL_MODELS]
        for model, r in table.iterrows():
            fig.add_trace(go.Box(y=[LABELS[model]], q1=[r.q25], median=[r['median']], q3=[r.q75],
                                 lowerfence=[r.q05], upperfence=[r.q95], orientation='h', name=LABELS[model],
                                 fillcolor=COLORS[model], line=dict(color='#3f4650', width=1), width=.62,
                                 hoverinfo='skip', showlegend=False), row=1, col=col)
        unit = FORCE_UNIT if metric == 'force' else 'meV/atom'
        fig.add_trace(go.Scatter(
            x=table['median'], y=[LABELS[m] for m in FULL_MODELS], mode='markers', showlegend=False,
            marker=dict(size=18, opacity=0), customdata=table[['q05', 'q25', 'q75', 'q95']].to_numpy(),
            hoverlabel=dict(bordercolor=[COLORS[m] for m in FULL_MODELS]),
            hovertemplate=f'<b>%{{y}}</b><br>median %{{x:.3g}} {unit}<br>P25 to P75 %{{customdata[1]:.3g}} to '
                          f'%{{customdata[2]:.3g}}<br>P5 to P95 %{{customdata[0]:.3g}} to %{{customdata[3]:.3g}}'
                          '<extra></extra>'), row=1, col=col)
        fig.update_xaxes(type='log', row=1, col=col)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    return style(fig, 640)


def figure2():
    table = prepare_table()
    indexed = table.set_index('model')
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.1, subplot_titles=['(a) Force', '(b) Relative energy'])
    for col, (metric, unit, ylabel) in enumerate([('force', FORCE_UNIT, f'Median force RMSE ({FORCE_UNIT})'),
                                                  ('energy', 'meV/atom', 'Median relative-energy error (meV/atom)')], 1):
        for i, path in enumerate(SIZE_PATHS):
            part = indexed.loc[path]
            fig.add_trace(go.Scatter(x=part.milliseconds_per_configuration, y=part[metric], mode='lines',
                                     line=dict(color=CONNECTOR_COLOR, width=1.5, dash='dot'), hoverinfo='skip',
                                     name='Model sizes within a family', legendgroup='sizes',
                                     showlegend=col == 1 and i == 0), row=1, col=col)
        front = table[table['pareto_' + metric]].sort_values('milliseconds_per_configuration')
        fig.add_trace(go.Scatter(x=front.milliseconds_per_configuration, y=front[metric], mode='lines',
                                 line=dict(color=FRONT_COLOR, width=2.5), hoverinfo='skip', name='Pareto frontier',
                                 legendgroup='front', showlegend=col == 1), row=1, col=col)
        for model in LEGEND_ORDER:
            r = indexed.loc[model]
            fig.add_trace(go.Scatter(
                x=[r.milliseconds_per_configuration], y=[r[metric]], mode='markers', name=LABELS[model],
                legendgroup=model, showlegend=col == 1, hoverlabel=dict(bordercolor=COLORS[model]),
                marker=dict(symbol=SYMBOLS[MARKERS[model]], size=14, color=COLORS[model],
                            line=dict(color=FRONT_COLOR if r['pareto_' + metric] else 'white', width=1.6)),
                hovertemplate=f'<b>{LABELS[model]}</b><br>median %{{y:.3g}} {unit}<br>'
                              '%{x:.3g} ms per configuration<extra></extra>'), row=1, col=col)
        low, high = table[metric].min(), table[metric].max()
        ticks = [.02, .05, .1, .2, .5] if metric == 'force' else [.5, 1, 2, 5, 10, 20, 50]
        fig.update_yaxes(type='log', title_text=ylabel, row=1, col=col,
                         range=[np.log10(high) + .15, np.log10(low) - .15])
        fig.update_xaxes(type='log', title_text='Evaluation time (ms per configuration)', row=1, col=col)
        style(fig, 620)
        fig.update_yaxes(tickvals=ticks, ticktext=[f'{v:g}' for v in ticks], row=1, col=col)
        fig.update_xaxes(tickvals=[.1, .3, 1, 3, 10, 30], ticktext=['0.1', '0.3', '1', '3', '10', '30'], row=1, col=col)
    return fig


def profile(cohort):
    stats = read('profile_statistics.csv')
    stats = stats[stats.cohort.eq(cohort)]
    models = COHORT_MODELS[cohort]
    labels = [LABELS[m] for m in models]
    blocks = [('force', ['reactant', 'F_rev', 'ts', 'F_forw', 'product'],
               ['Reactant', 'Reverse<br>max-slope', 'Transition<br>state', 'Forward<br>max-slope', 'Product'],
               (.01, 1.), [.01, .03, .1, .3, 1], FORCE_UNIT),
              ('energy', ['F_rev', 'ts', 'F_forw', 'reaction_energy'],
               ['Reverse<br>max-slope', 'Transition<br>state', 'Forward<br>max-slope', 'Reaction<br>energy'],
               (.1, 100.), [.1, 1, 10, 100], ENERGY_UNIT)]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.03, column_widths=[.53, .47],
                        subplot_titles=[f'(a) Force RMSE ({FORCE_UNIT})', f'(b) Energy error ({ENERGY_UNIT})'])
    for col, (metric, roles, names, limits, ticks, unit) in enumerate(blocks, 1):
        part = stats[stats.metric.eq(metric)]
        wide = {k: part.pivot(index='model', columns='location', values=k).reindex(index=models, columns=roles)
                for k in ['median', 'mean', 'q25', 'q75']}
        median = wide['median'].to_numpy()
        fig.add_trace(go.Heatmap(
            z=np.log10(median), x=names, y=labels, text=[[f'{v:.3g}' for v in row] for row in median],
            texttemplate='%{text}', textfont=dict(size=12), colorscale=ERROR_SCALE, zmin=np.log10(limits[0]),
            zmax=np.log10(limits[1]), xgap=3, ygap=3,
            customdata=np.dstack([wide[k].to_numpy() for k in ['median', 'mean', 'q25', 'q75']]),
            hovertemplate=f'<b>%{{y}}</b><br>%{{x}}<br>median %{{customdata[0]:.3g}} {unit}<br>mean '
                          f'%{{customdata[1]:.3g}}<br>P25 to P75 %{{customdata[2]:.3g}} to %{{customdata[3]:.3g}}'
                          '<extra></extra>',
            colorbar=dict(orientation='h', x=.27 if col == 1 else .77, y=-.12, len=.42, thickness=12,
                          tickvals=np.log10(ticks), ticktext=[f'{t:g}' for t in ticks],
                          title=dict(text=unit, side='right'))), row=1, col=col)
        for j in range(len(roles)):
            i = int(np.argmin(median[:, j]))
            fig.add_shape(type='rect', x0=j - .5, x1=j + .5, y0=i - .5, y1=i + .5, fillcolor='rgba(0,0,0,0)',
                          line=dict(color='#123F4A', width=2.5), row=1, col=col)
        fig.update_xaxes(side='top', showgrid=False, showline=False, ticks='', row=1, col=col)
    fig.update_yaxes(autorange='reversed', showgrid=False, showline=False, ticks='')
    style(fig, 160 + 34 * len(models))
    fig.update_layout(margin=dict(t=120, b=90))
    fig.update_annotations(yshift=48)
    return fig


def ecdf(cohort):
    table = ts_errors(cohort)
    models = LEGENDS[cohort]
    titles = [f'({chr(97 + 4 * j + i)}) {cls}' for i, cls in enumerate(CLASSES) for j in range(2)]
    fig = make_subplots(rows=4, cols=2, shared_yaxes=True, vertical_spacing=.06, horizontal_spacing=.05,
                        subplot_titles=titles)
    for col, (field, unit) in enumerate([('barrier_error_kcal_mol', ENERGY_UNIT),
                                         ('ts_force_rmse_ev_per_a', FORCE_UNIT)], 1):
        for row, reaction_class in enumerate(CLASSES, 1):
            part = table[table.reaction_class.eq(reaction_class)]
            for model in models:
                rows = part[part.model.eq(model)].dropna(subset=[field]).sort_values(field)
                n = len(rows)
                fig.add_trace(go.Scatter(
                    x=rows[field].to_numpy(np.float32), y=(np.arange(1, n + 1) / n).astype(np.float32),
                    customdata=rows.reaction_id.to_numpy(), mode='lines', line_shape='hv',
                    line=dict(color=COLORS[model], width=1.8, dash=LINE_DASH[model]), name=LABELS[model],
                    legendgroup=model, showlegend=row == 1 and col == 1, hoverlabel=dict(bordercolor=COLORS[model]),
                    hovertemplate=f'<b>{LABELS[model]}</b><br>%{{x:.3g}} {unit}<br>fraction %{{y:.3f}}<br>'
                                  '%{customdata}<extra></extra>'), row=row, col=col)
            fig.update_xaxes(type='log', row=row, col=col,
                             range=[-4, 2] if col == 1 else [np.log10(.003), np.log10(3)])
            fig.update_yaxes(range=[-.02, 1.02], row=row, col=col)
        fig.update_xaxes(title_text='Absolute barrier error (kcal/mol)' if col == 1 else f'TS force RMSE ({FORCE_UNIT})',
                         row=4, col=col)
    for row in range(1, 5):
        fig.update_yaxes(title_text='Fraction of reactions', row=row, col=1)
    return style(fig, 1150)


def tails():
    table = ts_errors('chno')
    metrics = [('ts_force_rmse_ev_per_a', FORCE_UNIT), ('barrier_error_kcal_mol', ENERGY_UNIT)]
    summary = {(m, f): tail_statistics(table.loc[table.model.eq(m), f]) for m in CHNO_MODELS for f, _ in metrics}
    order = sorted(CHNO_MODELS, key=lambda m: summary[m, metrics[0][0]]['median'])
    labels = [LABELS[m] for m in order]
    colors = [FAMILY_COLORS[family(m)] for m in order]
    markers = [('median', 'Median', 'circle'), ('p95', 'P95', 'square-open'),
               ('p99', 'P99', 'triangle-up-open'), ('maximum', 'Maximum', 'x')]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.04,
                        subplot_titles=[f'(a) TS force RMSE ({FORCE_UNIT})', f'(b) Absolute barrier error ({ENERGY_UNIT})'])
    for col, (field, unit) in enumerate(metrics, 1):
        for model, label, color in zip(order, labels, colors):
            values = [summary[model, field][k] for k, _, _ in markers]
            fig.add_trace(go.Scatter(x=values, y=[label] * 4, mode='lines', hoverinfo='skip', showlegend=False,
                                     line=dict(color=color, width=2), opacity=.45), row=1, col=col)
        for key, name, symbol in markers:
            fig.add_trace(go.Scatter(
                x=[summary[m, field][key] for m in order], y=labels, mode='markers', name=name, legendgroup=key,
                showlegend=False, marker=dict(symbol=symbol, size=11, color=colors, line=dict(width=1.6, color=colors)),
                hoverlabel=dict(bordercolor=colors),
                hovertemplate=f'<b>%{{y}}</b><br>{name} %{{x:.3g}} {unit}<extra></extra>'), row=1, col=col)
            if col == 1:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name, legendgroup=key,
                                         marker=dict(symbol=symbol, size=11, color='#334155', line=dict(width=1.6))),
                              row=1, col=col)
        fig.update_xaxes(type='log', row=1, col=col)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    return style(fig, 720)


def table2():
    """Table 2 as bars, with the Table S3 intervals as error bars."""
    data = ts_errors('chno')
    models, results = bootstrap(data)
    mean_barrier = data[data.model.eq('uma_m_omol')].reference_barrier_kcal_mol.dropna().mean()
    value = {(metric, m): [results[metric][s][0, j] for s in (1, 2, 3)]
             for metric in [BARRIER, FORCE] for j, m in enumerate(models)}
    within = {m: results[BARRIER][1][3, j] for j, m in enumerate(models)}
    order = sorted(CHNO_MODELS, key=lambda m: value[BARRIER, m][0])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.04,
                        subplot_titles=['(a) Total barrier MAE (kcal/mol)', f'(b) Mean TS force RMSE ({FORCE_UNIT})'])
    for col, metric in enumerate([BARRIER, FORCE], 1):
        estimate, lower, upper = np.array([value[metric, m] for m in order]).T
        unit = ENERGY_UNIT if metric == BARRIER else FORCE_UNIT
        custom = [lower, upper]
        detail = ''
        if metric == BARRIER:
            custom += [100 * estimate / mean_barrier, np.array([within[m] for m in order])]
            detail = '<br>%{customdata[2]:.2f}% of the mean DFT barrier<br>%{customdata[3]:.1f}% of barriers within 1 kcal/mol'
        fig.add_trace(go.Bar(
            x=estimate, y=[LABELS[m] for m in order], orientation='h', marker=dict(color=[COLORS[m] for m in order]),
            error_x=dict(type='data', symmetric=False, array=upper - estimate, arrayminus=estimate - lower,
                         color='#334155', thickness=1.4, width=4),
            showlegend=False, customdata=np.column_stack(custom),
            hoverlabel=dict(bordercolor=[COLORS[m] for m in order]),
            hovertemplate=f'<b>%{{y}}</b><br>%{{x:.3g}} {unit}, 95% interval %{{customdata[0]:.3g}} to '
                          f'%{{customdata[1]:.3g}}{detail}<extra></extra>'), row=1, col=col)
        fig.update_xaxes(range=[0, upper.max() * 1.05], row=1, col=col)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    style(fig, 680)
    fig.update_layout(bargap=.28)
    return fig


def molecule(reaction, frame='ts'):
    """A 2D ball-and-stick projection, used as the gallery image of Figure 5."""
    shown = reaction['frames'][frame]
    xyz = np.array(shown['xyz'])
    fig = go.Figure()
    for pairs, color, width, dash in [(shown['bonds'], '#9AA1AB', 7, 'solid'), (shown['dashed'], '#E08A1E', 5, 'dot')]:
        for a, b in pairs:
            fig.add_trace(go.Scatter(x=xyz[[a, b], 0], y=xyz[[a, b], 1], mode='lines',
                                     line=dict(color=color, width=width, dash=dash)))
    order = np.argsort(xyz[:, 2])
    symbols = [reaction['symbols'][i] for i in order]
    fig.add_trace(go.Scatter(x=xyz[order, 0], y=xyz[order, 1], mode='markers', marker=dict(
        color=[ATOMS[s][0] for s in symbols], size=[150 * ATOMS[s][1] for s in symbols],
        line=dict(color='#4F535A', width=1.2))))
    fig.update_layout(template=TEMPLATE, showlegend=False, height=600, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
    return fig


def hero_curves(width=1200, height=420):
    """Background lines drawn from the Figure 4 data, the barrier-error ECDF of each checkpoint."""
    table = ts_errors('full').dropna(subset=['barrier_error_kcal_mol'])
    q = np.linspace(.002, .998, 140)
    lines = []
    for model in FULL_MODELS:
        x = np.log10(np.clip(np.quantile(table.loc[table.model.eq(model), 'barrier_error_kcal_mol'], q), 1e-3, 100))
        points = ' '.join(f'{(v + 3) / 5 * width:.1f},{height * (.95 - .9 * p):.1f}' for v, p in zip(x, q))
        lines.append(f'<polyline points="{points}" stroke="{COLORS[model]}"/>')
    return (f'<svg class="curves" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<g fill="none" stroke-width="2.2" vector-effect="non-scaling-stroke">{"".join(lines)}</g></svg>')


CSS = '''
:root { --ink: #0f172a; --muted: #64748b; --line: #e2e8f0; --bg: #f5f7fb; --accent: #2563eb; --navy: #0b1f3a; }
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, "Helvetica Neue", Arial, sans-serif; color: var(--ink); background: var(--bg);
  line-height: 1.55; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 0 20px; }
.topbar { position: sticky; top: 0; z-index: 10; background: rgba(11, 31, 58, .94); backdrop-filter: blur(8px); }
.topbar .wrap { display: flex; align-items: center; gap: 18px; height: 56px; }
.brand { color: #fff; font-weight: 700; white-space: nowrap; }
.brand span { color: #7dd3fc; }
.brand:hover { text-decoration: none; }
.nav { display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none; margin-left: auto; }
.nav a { color: #cbd5e1; padding: 6px 10px; border-radius: 999px; font-size: 14px; white-space: nowrap; }
.nav a:hover { background: rgba(255, 255, 255, .1); color: #fff; text-decoration: none; }
.nav a.current { background: #fff; color: var(--navy); font-weight: 600; }
.hero { position: relative; overflow: hidden; color: #fff;
  background: radial-gradient(900px 420px at 85% 0%, #1e4f8a 0%, rgba(30, 79, 138, 0) 70%),
              linear-gradient(160deg, #0b1f3a 0%, #10294a 60%, #0c2240 100%); }
.hero .curves { position: absolute; left: 0; bottom: 0; width: 100%; height: 78%; opacity: .5; }
.hero .shade { position: absolute; inset: 0; background: linear-gradient(90deg, rgba(11, 31, 58, .92) 0%,
  rgba(11, 31, 58, .55) 55%, rgba(11, 31, 58, .15) 100%); }
.hero .wrap { position: relative; padding: 70px 20px 60px; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 12px; color: #7dd3fc; font-weight: 600; }
.hero h1 { font-size: clamp(30px, 4.4vw, 50px); line-height: 1.1; margin: 12px 0 16px; max-width: 860px; }
.lede { font-size: 18px; color: #dbe5f1; max-width: 720px; margin: 0; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 36px; max-width: 980px; }
.stat { background: rgba(255, 255, 255, .07); border: 1px solid rgba(255, 255, 255, .16); border-radius: 14px;
  padding: 14px 16px; backdrop-filter: blur(4px); }
.stat b { display: block; font-size: 28px; line-height: 1.2; }
.stat span { color: #c7d4e4; font-size: 13px; }
.cta { display: flex; gap: 12px; margin-top: 30px; flex-wrap: wrap; }
.button { display: inline-block; padding: 10px 16px; border-radius: 10px; font-weight: 600; font-size: 15px; }
.button:hover { text-decoration: none; filter: brightness(1.05); }
.button.primary { background: #fff; color: var(--navy); }
.button.ghost { border: 1px solid rgba(255, 255, 255, .35); color: #fff; }
section { padding: 44px 0 20px; }
h2 { font-size: 24px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 22px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 20px; }
.card { background: #fff; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; color: inherit;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 8px 24px rgba(15, 23, 42, .06); }
a.card { display: flex; flex-direction: column; transition: transform .15s ease, box-shadow .15s ease; }
a.card:hover { transform: translateY(-3px); text-decoration: none;
  box-shadow: 0 2px 4px rgba(15, 23, 42, .06), 0 18px 36px rgba(15, 23, 42, .12); }
.thumb { aspect-ratio: 16 / 10; background: #fff; border-bottom: 1px solid var(--line); overflow: hidden; }
.thumb img { width: 100%; height: 100%; object-fit: cover; object-position: top center; display: block; }
.card .body { padding: 14px 16px 18px; }
.tag { display: inline-block; font-size: 12px; font-weight: 600; color: var(--accent); background: #eef3ff;
  border-radius: 999px; padding: 2px 10px; }
.tag.new { color: #0f766e; background: #e6f6f4; margin-left: 6px; }
.card h3 { margin: 8px 0 4px; font-size: 17px; line-height: 1.3; }
.card p { margin: 0; color: var(--muted); font-size: 14px; }
.page-head { padding: 30px 0 4px; }
.page-head h1 { font-size: clamp(24px, 3vw, 32px); margin: 10px 0 8px; line-height: 1.2; }
.caption { color: #334155; max-width: 920px; margin: 0; }
.tips { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 18px; }
.tips span { font-size: 13px; color: #334155; background: #fff; border: 1px solid var(--line); border-radius: 999px;
  padding: 4px 11px; }
.plot { padding: 10px; }
.pager { display: flex; justify-content: space-between; gap: 14px; margin: 24px 0 8px; }
.pager a { flex: 1; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px;
  font-weight: 600; color: var(--ink); }
.pager a:hover { border-color: #b6c2d2; text-decoration: none; }
.pager a.next { text-align: right; }
.pager small { display: block; color: var(--muted); font-weight: 500; }
footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; padding: 22px 0 40px; margin-top: 30px; }
.molecules { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.mol-head { display: flex; align-items: baseline; gap: 10px; padding: 14px 16px 2px; flex-wrap: wrap; }
.letter { font-weight: 700; font-size: 18px; }
.formula { font-weight: 600; font-size: 18px; }
.rid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); }
.viewer { position: relative; height: 330px; margin: 0 8px; }
.controls { display: flex; gap: 6px; padding: 8px 16px; flex-wrap: wrap; align-items: center; }
.controls button { font: inherit; font-size: 13px; border: 1px solid var(--line); background: #fff; color: var(--ink);
  border-radius: 999px; padding: 4px 12px; cursor: pointer; }
.controls button.on { background: var(--navy); color: #fff; border-color: var(--navy); }
.controls .play { margin-left: auto; }
.errors { display: grid; grid-template-columns: max-content 1fr 1fr; gap: 8px 16px; padding: 10px 16px 14px;
  font-size: 14px; align-items: center; border-top: 1px solid var(--line); margin-top: 4px; }
.errors .head { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.who { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.who i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.meter b { font-variant-numeric: tabular-nums; }
.meter div { height: 6px; background: #eef1f5; border-radius: 999px; margin-top: 4px; overflow: hidden; }
.meter div span { display: block; height: 100%; border-radius: 999px; }
.dft { padding: 0 16px 14px; color: var(--muted); font-size: 13px; }
.contacts { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 16px 10px; min-height: 30px; }
.contacts span { font-size: 13px; color: #7a4d12; background: #fff4e5; border: 1px solid #f3d3a5; border-radius: 999px;
  padding: 2px 10px; font-variant-numeric: tabular-nums; }
.legend-row { display: flex; gap: 18px; flex-wrap: wrap; align-items: center; margin: 0 0 18px; color: #334155; font-size: 14px; }
.legend-row i { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: -1px;
  border: 1px solid #4f535a; }
.legend-row i.dash { width: 26px; height: 0; border: 0; border-top: 3px dotted #e08a1e; border-radius: 0; vertical-align: 3px; }
.legend-row .controls { padding: 0; margin-left: auto; }
@media (max-width: 900px) { .molecules { grid-template-columns: 1fr; } }
@media (max-width: 760px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero .wrap { padding: 44px 16px 40px; }
  .wrap { padding: 0 16px; }
  .brand { font-size: 14px; }
}
'''

HEAD = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>
'''
FOOT = '''<footer><div class="wrap">Built by <a href="{repo}/blob/main/paper/interactive.py">paper/interactive.py</a> from the
processed results in <a href="{repo}/tree/main/data">data/</a>. Code, data, and the scripts for the static figures are in the
<a href="{repo}">GitHub repository</a>.</div></footer>
<script>
document.fonts && document.fonts.ready.then(function () {{
  if (window.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(function (p) {{ Plotly.Plots.resize(p); }});
}});
</script>
</body>
</html>
'''

FIGURE5_SCRIPT = '''
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.5/3Dmol-min.js"></script>
<script>
const reactions = JSON.parse(document.getElementById('reactions').textContent);
const STYLE = {C: ['#555B66', .36], H: ['#EEF0F3', .24], N: ['#4267C8', .35], O: ['#D44E4B', .35]};
const FRAMES = ['reactant', 'ts', 'product'];
const players = [];
function at(p) { return {x: p[0], y: p[1], z: p[2]}; }
function draw(entry) {
  const v = entry.viewer, rx = entry.reaction, f = rx.frames[entry.frame];
  v.removeAllModels(); v.removeAllShapes(); v.removeAllLabels();
  const atoms = f.xyz.map((p, i) => ({elem: rx.symbols[i], x: p[0], y: p[1], z: p[2], serial: i + 1, bonds: [], bondOrder: []}));
  f.bonds.forEach(([a, b]) => { atoms[a].bonds.push(b); atoms[a].bondOrder.push(1); atoms[b].bonds.push(a); atoms[b].bondOrder.push(1); });
  const model = v.addModel();
  model.addAtoms(atoms);
  Object.entries(STYLE).forEach(([elem, [color, radius]]) =>
    model.setStyle({elem: elem}, {sphere: {radius: radius, color: color}, stick: {radius: .11, color: '#9AA1AB'}}));
  entry.contacts.textContent = '';
  rx.changing_contacts.forEach(([a, b]) => {
    const p = f.xyz[a], q = f.xyz[b];
    const open = f.dashed.some(([x, y]) => x === a && y === b);
    if (open) v.addCylinder({start: at(p), end: at(q), radius: .09, color: '#E08A1E', dashed: true, dashLength: .17, gapLength: .12});
    const item = document.createElement('span');
    item.textContent = rx.symbols[a] + (a + 1) + '\u2013' + rx.symbols[b] + (b + 1) + ' ' +
      Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]).toFixed(2) + ' \u00c5' + (open ? '' : ', bonded');
    entry.contacts.appendChild(item);
  });
  v.setHoverable({}, true, function (atom, viewer) {
    if (!atom.label) atom.label = viewer.addLabel(atom.elem + atom.serial, {position: atom, fontSize: 12,
      backgroundColor: '#0f172a', fontColor: '#fff', backgroundOpacity: .85, inFront: true});
  }, function (atom, viewer) { if (atom.label) { viewer.removeLabel(atom.label); delete atom.label; } });
  v.render();
}
function show(entry, frame) {
  entry.frame = frame;
  entry.buttons.forEach(b => b.classList.toggle('on', b.dataset.frame === frame));
  draw(entry);
}
function toggle(entry) {
  if (entry.timer) { clearInterval(entry.timer); entry.timer = null; entry.play.textContent = 'Play'; return; }
  entry.play.textContent = 'Pause';
  entry.timer = setInterval(() => show(entry, FRAMES[(FRAMES.indexOf(entry.frame) + 1) % 3]), 1100);
}
document.querySelectorAll('.molecule').forEach((card, i) => {
  const entry = {reaction: reactions[i], frame: 'ts', timer: null};
  entry.viewer = $3Dmol.createViewer(card.querySelector('.viewer'), {backgroundColor: 'white', antialias: true});
  entry.buttons = Array.from(card.querySelectorAll('button[data-frame]'));
  entry.buttons.forEach(b => b.addEventListener('click', () => show(entry, b.dataset.frame)));
  entry.play = card.querySelector('.play');
  entry.contacts = card.querySelector('.contacts');
  entry.play.addEventListener('click', () => toggle(entry));
  show(entry, 'ts');
  entry.viewer.zoomTo();
  entry.viewer.zoom(1.15);
  entry.viewer.render();
  players.push(entry);
});
document.getElementById('play-all').addEventListener('click', function () {
  const running = players.some(e => e.timer);
  players.forEach(e => { if (!!e.timer === running) toggle(e); });
  this.textContent = running ? 'Play all' : 'Pause all';
});
</script>
'''


def topbar(current):
    links = ''.join(f'<a href="{stem}.html"{" class=current" if stem == current else ""}>{label}</a>'
                    for stem, label, *_ in PAGES)
    return (f'<div class="topbar"><div class="wrap"><a class="brand" href="index.html">DORTS-9K <span>MLIP</span> '
            f'benchmark</a><nav class="nav">{links}<a href="{REPO}">GitHub</a></nav></div></div>\n')


def page(stem, label, title, caption, legend, body):
    i = [p[0] for p in PAGES].index(stem)
    previous, following = PAGES[i - 1] if i else None, PAGES[i + 1] if i + 1 < len(PAGES) else None
    pager = '<div class="pager">'
    pager += (f'<a href="{previous[0]}.html"><small>Previous</small>{previous[1]}. {previous[2]}</a>' if previous
              else '<a href="index.html"><small>Back to</small>All figures</a>')
    pager += (f'<a class="next" href="{following[0]}.html"><small>Next</small>{following[1]}. {following[2]}</a>'
              if following else '<a class="next" href="index.html"><small>Back to</small>All figures</a>')
    pager += '</div>'
    tips = ''.join(f'<span>{t}</span>' for t in TIPS[legend])
    return (topbar(stem) + f'<main class="wrap"><div class="page-head"><span class="tag">{label}</span><h1>{title}</h1>'
            f'<p class="caption">{caption}</p></div><div class="tips">{tips}</div>{body}{pager}</main>\n')


def figure5_body(reactions):
    peak = {k: max(r['errors'][m][k] for r in reactions for m in MODEL_COLORS)
            for k in ['ts_force_rmse_ev_per_a', 'barrier_error_kcal_mol']}
    names = {'uma_m_omol': 'UMA-M OMol', 'aimnet2_rxn': 'AIMNet2-RXN'}
    cards = []
    for r in reactions:
        formula = ''.join(f'<sub>{c}</sub>' if c.isdigit() else c for c in r['formula'])
        rows = ''
        for model, color in MODEL_COLORS.items():
            cells = ''
            for key in ['ts_force_rmse_ev_per_a', 'barrier_error_kcal_mol']:
                value = r['errors'][model][key]
                digits = 2 if key == 'barrier_error_kcal_mol' and value >= 10 else 3  # as printed in the paper
                cells += (f'<div class="meter"><b>{value:.{digits}f}</b><div><span style="width:{100 * value / peak[key]:.1f}%;'
                          f'background:{color}"></span></div></div>')
            rows += f'<div class="who"><i style="background:{color}"></i>{names[model]}</div>{cells}'
        cards.append(
            f'<div class="card molecule"><div class="mol-head"><span class="letter">{r["panel"]}</span>'
            f'<span class="formula">{formula}</span><span class="tag">{r["reaction_class"]}</span>'
            f'<span class="rid">{r["reaction_id"]}</span></div><div class="viewer"></div>'
            '<div class="controls"><button data-frame="reactant">Reactant</button><button data-frame="ts">Transition state'
            '</button><button data-frame="product">Product</button><button class="play">Play</button></div>'
            '<div class="contacts"></div>'
            f'<div class="errors"><span></span><span class="head">TS force RMSE ({FORCE_UNIT})</span>'
            f'<span class="head">Barrier error ({ENERGY_UNIT})</span>{rows}</div>'
            f'<div class="dft">DFT barrier {r["dft_barrier_kcal_mol"]:.1f} kcal/mol, measured from the '
            f'{r["reference_endpoint"]}</div></div>')
    legend = ''.join(f'<span><i style="background:{ATOMS[s][0]}"></i>{s}</span>' for s in ['C', 'H', 'N', 'O'])
    return (f'<div class="legend-row">{legend}<span><i class="dash"></i>Changing contact</span>'
            '<span class="controls"><button id="play-all">Play all</button></span></div>'
            f'<div class="molecules">{"".join(cards)}</div>'
            f'<script type="application/json" id="reactions">{json.dumps(reactions, separators=(",", ":"))}</script>')


def index_body(stats):
    tiles = ''.join(f'<div class="stat"><b>{value}</b><span>{text}</span></div>' for value, text in stats)
    cards = ''
    for stem, label, title, summary, *_ in PAGES:
        badge = '<span class="tag new">3D</span>' if stem == 'figure5' else ''
        cards += (f'<a class="card" href="{stem}.html"><div class="thumb"><img src="thumbs/{stem}.png" alt="" '
                  f'loading="lazy"></div><div class="body"><span class="tag">{label}</span>{badge}<h3>{title}</h3>'
                  f'<p>{summary}</p></div></a>')
    return (topbar(None) + f'<header class="hero">{hero_curves()}<div class="shade"></div><div class="wrap">'
            '<div class="eyebrow">Interactive figures</div>'
            '<h1>Pretrained interatomic potentials along molecular reaction pathways</h1>'
            '<p class="lede">Seventeen pretrained machine-learning interatomic potentials compared with ωB97M-V/def2-TZVP '
            'energies and forces on the DORTS-9K reaction dataset, from reactants through transition states to products.</p>'
            f'<div class="stats">{tiles}</div><div class="cta"><a class="button primary" href="figure1.html">Explore the '
            f'figures</a><a class="button ghost" href="{REPO}">Code and data on GitHub</a></div></div></header>'
            '<main class="wrap"><section><h2>Figures</h2><p class="sub">Every figure is drawn from the processed results '
            f'in the repository. Open one to hover, zoom, and filter checkpoints.</p><div class="grid">{cards}</div>'
            '</section></main>\n')


def thumbnail(fig, stem, width=1000):
    fig = go.Figure(fig)
    fig.update_layout(showlegend=False, width=width)
    (SITE / 'thumbs').mkdir(exist_ok=True)
    fig.write_image(SITE / 'thumbs' / f'{stem}.png', scale=.6)


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / 'style.css').write_text(CSS.lstrip(), encoding='utf-8')
    config = {'responsive': True, 'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'},
              'modeBarButtonsToRemove': ['select2d', 'lasso2d']}
    builders = {'figure1': figure1, 'figure2': figure2, 'figure3': lambda: profile('full'),
                'figure4': lambda: ecdf('full'), 'table2': table2, 'figureS1': lambda: profile('chno'),
                'figureS2': lambda: ecdf('chno'), 'figureS3': tails}
    reactions = json.loads((DATA / 'figure5_reactions.json').read_text())
    for stem, label, title, summary, caption, legend in PAGES:
        if stem == 'figure5':
            body, script = figure5_body(reactions), FIGURE5_SCRIPT
            thumbnail(molecule(reactions[0]), stem, width=960)
        else:
            fig = builders[stem]()
            body = '<div class="card plot">' + fig.to_html(full_html=False, include_plotlyjs='cdn', config=config) + '</div>'
            script = ''
            thumbnail(fig, stem)
        html = (HEAD.format(title=f'{label}. {title}', description=summary)
                + page(stem, label, title, caption, legend, body) + script + FOOT.format(repo=REPO))
        (SITE / f'{stem}.html').write_text(html, encoding='utf-8')
        print('Wrote', SITE / f'{stem}.html')
    full = ts_errors('full')
    uma = full[full.model.eq('uma_m_omol')].barrier_error_kcal_mol.dropna()
    counts = read('global_statistics.csv').query("cohort == 'full' and metric == 'force'").iloc[0]
    stats = [(f'{counts.n_configurations:,}', 'configurations with DFT energies and forces'),
             (f'{counts.n_reactions:,}', 'reactions'), (f'{len(CHNO_MODELS)}', 'pretrained checkpoints'),
             (f'{100 * (uma <= 1).mean():.1f}%', 'of UMA-M OMol barriers within 1 kcal/mol')]
    html = (HEAD.format(title='DORTS-9K MLIP benchmark', description='Interactive figures of the DORTS-9K MLIP benchmark')
            + index_body(stats) + FOOT.format(repo=REPO))
    (SITE / 'index.html').write_text(html, encoding='utf-8')
    print('Wrote', SITE / 'index.html')


if __name__ == '__main__':
    main()
