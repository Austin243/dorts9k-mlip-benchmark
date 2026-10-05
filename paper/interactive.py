"""Interactive versions of the data figures, written as a small website to results/site.

Every page shows one figure of the paper with hover readouts and zoom. GitHub Pages publishes the
site from the main branch through .github/workflows/pages.yml.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import CHNO_MODELS, CLASSES, COHORT_MODELS, COLORS, FULL_MODELS, LABELS, ROOT, read, tail_statistics, ts_errors
from figure2 import CONNECTOR_COLOR, FRONT_COLOR, LEGEND_ORDER, MARKERS, SIZE_PATHS, prepare_table
from figure4 import PATTERN
from figureS3 import FAMILY_COLORS, family

SITE = ROOT / 'results' / 'site'
REPO = 'https://github.com/Austin243/dorts9k-mlip-benchmark'
FONT = 'Arial, Helvetica, sans-serif'
SYMBOLS = {'o': 'circle', 's': 'square', 'D': 'diamond', '^': 'triangle-up', 'v': 'triangle-down', 'P': 'cross'}
DASHES = {'-': 'solid', (0, (3.2, 1.6)): 'dash', (0, (.05, 1.9)): 'dot',
          (0, (5, 1.5, 1, 1.5)): 'dashdot', (0, (6.5, 2)): 'longdash'}
LINE_DASH = {m: DASHES[PATTERN[m]] for m in FULL_MODELS} | {'ani1xnr': 'dot', 'aimnet2_rxn': 'solid'}
ERROR_SCALE = ['#B8DED8', '#E5F0E5', '#FFF0C2', '#E79A67', '#9E413B']
FORCE_UNIT, ENERGY_UNIT = 'eV/Å', 'kcal/mol'
HINTS = {True: 'Hover for values. Click a legend entry to hide it, double-click it to show it alone, and drag to zoom.',
         False: 'Hover for values and drag to zoom. Double-click the plot to reset the view.'}
LEGEND_PAGES = {'figure2', 'figure4', 'figureS2', 'figureS3'}
PAGES = [
    ('figure1', 'Figure 1', 'Global error distributions',
     'Force RMSE (a) and absolute per-atom relative-energy error (b) of fifteen checkpoints over all DORTS-9K '
     'configurations. Boxes span the 25th to 75th percentiles and whiskers the 5th to 95th percentiles.'),
    ('figure2', 'Figure 2', 'Accuracy and evaluation time',
     'Median force RMSE (a) and median absolute per-atom relative-energy error (b) against the mean energy-and-force '
     'evaluation time per configuration on one A100 GPU. The solid line is the Pareto frontier, and dotted lines join '
     'model sizes within a family.'),
    ('figure3', 'Figure 3', 'Errors along the reaction profile',
     'Median force RMSE and median absolute total energy error at the five direct anchors of the 8,474 reactions that '
     'have all five. Energies are measured from the reference endpoint, and the last column is the error in the '
     'reaction energy. Outlined cells mark the smallest value in each column.'),
    ('figure4', 'Figure 4', 'Transition-state errors by reaction class',
     'Cumulative distributions of the absolute total barrier error (a to d) and the transition-state force RMSE '
     '(e to h) of fifteen checkpoints on the full dataset. Hovering over a curve names the reaction at each step.'),
    ('figureS1', 'Figure S1', 'Errors along the reaction profile on the CHNO subset',
     'The same comparison as Figure 3 for seventeen checkpoints on the 3,706 CHNO reactions with all five anchors.'),
    ('figureS2', 'Figure S2', 'Transition-state errors by reaction class on the CHNO subset',
     'The same comparison as Figure 4 for seventeen checkpoints on the CHNO reactions.'),
    ('figureS3', 'Figure S3', 'Error tails on the CHNO subset',
     'Median, P95, P99, and maximum transition-state force RMSE (a) and absolute total barrier error (b) of seventeen '
     'checkpoints on the CHNO subset, ordered by the median force error.'),
]


def style(fig, height):
    fig.update_layout(template='seaborn', height=height, font=dict(family=FONT, size=13, color='#242424'),
                      margin=dict(l=70, r=20, t=60, b=60), hoverlabel=dict(font_family=FONT),
                      legend=dict(groupclick='togglegroup', font=dict(size=12)))
    fig.update_annotations(font_size=14)
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
                                 fillcolor=COLORS[model], line=dict(color='#414141', width=1), width=.6,
                                 hoverinfo='skip', showlegend=False), row=1, col=col)
        unit = FORCE_UNIT if metric == 'force' else 'meV/atom'
        fig.add_trace(go.Scatter(
            x=table['median'], y=[LABELS[m] for m in FULL_MODELS], mode='markers', showlegend=False,
            marker=dict(size=18, opacity=0), customdata=table[['q05', 'q25', 'q75', 'q95']].to_numpy(),
            hovertemplate=f'<b>%{{y}}</b><br>median %{{x:.3g}} {unit}<br>P25 to P75 %{{customdata[1]:.3g}} to '
                          f'%{{customdata[2]:.3g}}<br>P5 to P95 %{{customdata[0]:.3g}} to %{{customdata[3]:.3g}}'
                          '<extra></extra>'), row=1, col=col)
        fig.update_xaxes(type='log', row=1, col=col)
    fig.update_yaxes(autorange='reversed')
    return style(fig, 640)


def figure2():
    table = prepare_table()
    indexed = table.set_index('model')
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.1,
                        subplot_titles=['(a) Force', '(b) Relative energy'])
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
                                 line=dict(color=FRONT_COLOR, width=2), hoverinfo='skip', name='Pareto frontier',
                                 legendgroup='front', showlegend=col == 1), row=1, col=col)
        for model in LEGEND_ORDER:
            r = indexed.loc[model]
            fig.add_trace(go.Scatter(
                x=[r.milliseconds_per_configuration], y=[r[metric]], mode='markers', name=LABELS[model],
                legendgroup=model, showlegend=col == 1,
                marker=dict(symbol=SYMBOLS[MARKERS[model]], size=13, color=COLORS[model],
                            line=dict(color=FRONT_COLOR if r['pareto_' + metric] else 'white', width=1.5)),
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
            zmax=np.log10(limits[1]), xgap=2, ygap=2,
            customdata=np.dstack([wide[k].to_numpy() for k in ['median', 'mean', 'q25', 'q75']]),
            hovertemplate=f'<b>%{{y}}</b><br>%{{x}}<br>median %{{customdata[0]:.3g}} {unit}<br>mean '
                          f'%{{customdata[1]:.3g}}<br>P25 to P75 %{{customdata[2]:.3g}} to %{{customdata[3]:.3g}}'
                          '<extra></extra>',
            colorbar=dict(orientation='h', x=.27 if col == 1 else .77, y=-.12, len=.42, thickness=12,
                          tickvals=np.log10(ticks), ticktext=[f'{t:g}' for t in ticks],
                          title=dict(text=unit, side='right'))), row=1, col=col)
        for j in range(len(roles)):
            i = int(np.argmin(median[:, j]))
            fig.add_shape(type='rect', x0=j - .5, x1=j + .5, y0=i - .5, y1=i + .5, fillcolor='rgba(0,0,0,0)', line=dict(color='#123F4A', width=2.5),
                          row=1, col=col)
        fig.update_xaxes(side='top', showgrid=False, row=1, col=col)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    style(fig, 160 + 34 * len(models))
    fig.update_layout(margin=dict(t=120, b=90))
    fig.update_annotations(yshift=48)
    return fig


def ecdf(cohort):
    table = ts_errors(cohort)
    models = COHORT_MODELS[cohort]
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
                    line=dict(color=COLORS[model], width=1.6, dash=LINE_DASH[model]), name=LABELS[model],
                    legendgroup=model, showlegend=row == 1 and col == 1,
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
    markers = [('median', 'Median', 'circle'), ('p95', 'P95', 'square-open'),
               ('p99', 'P99', 'triangle-up-open'), ('maximum', 'Maximum', 'x')]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.04,
                        subplot_titles=[f'(a) TS force RMSE ({FORCE_UNIT})', f'(b) Absolute barrier error ({ENERGY_UNIT})'])
    for col, (field, unit) in enumerate(metrics, 1):
        for model, label in zip(order, labels):
            values = [summary[model, field][k] for k, _, _ in markers]
            fig.add_trace(go.Scatter(x=values, y=[label] * 4, mode='lines', hoverinfo='skip', showlegend=False,
                                     line=dict(color=FAMILY_COLORS[family(model)], width=1.5), opacity=.55),
                          row=1, col=col)
        colors = [FAMILY_COLORS[family(m)] for m in order]
        for key, name, symbol in markers:
            fig.add_trace(go.Scatter(
                x=[summary[m, field][key] for m in order], y=labels, mode='markers', name=name, legendgroup=key,
                showlegend=False, marker=dict(symbol=symbol, size=10, color=colors, line=dict(width=1.5, color=colors)),
                hovertemplate=f'<b>%{{y}}</b><br>{name} %{{x:.3g}} {unit}<extra></extra>'), row=1, col=col)
            if col == 1:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name, legendgroup=key,
                                         marker=dict(symbol=symbol, size=10, color='#242424', line=dict(width=1.5))),
                              row=1, col=col)
        fig.update_xaxes(type='log', row=1, col=col)
    fig.update_yaxes(autorange='reversed')
    return style(fig, 720)


PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ margin: 0; background: #ffffff; color: #242424; font-family: Arial, Helvetica, sans-serif; line-height: 1.5; }}
header, main, footer {{ max-width: 1280px; margin: 0 auto; padding: 0 16px; }}
header {{ padding-top: 16px; border-bottom: 1px solid #e3e3e3; }}
header a.site {{ font-weight: bold; color: #242424; text-decoration: none; font-size: 18px; }}
nav {{ display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 8px 0 12px; }}
nav a {{ color: #2f6f9f; text-decoration: none; }}
nav a.current {{ color: #242424; font-weight: bold; }}
h1 {{ font-size: 24px; margin: 20px 0 8px; }}
p {{ max-width: 900px; }}
p.hint {{ color: #5c5c5c; font-size: 14px; }}
footer {{ border-top: 1px solid #e3e3e3; margin-top: 24px; padding-bottom: 24px; color: #5c5c5c; font-size: 14px; }}
footer a, main a {{ color: #2f6f9f; }}
</style>
</head>
<body>
<header><a class="site" href="index.html">DORTS-9K MLIP benchmark</a><nav>{nav}</nav></header>
<main>
{body}
</main>
<footer><p>Built by <a href="{repo}/blob/main/paper/interactive.py">paper/interactive.py</a> from the processed results in
<a href="{repo}/tree/main/data">data/</a>. Code and data are in the <a href="{repo}">GitHub repository</a>.</p></footer>
</body>
</html>
'''


def nav(current):
    return ''.join(f'<a href="{stem}.html"{" class=current" if stem == current else ""}>{number}</a>'
                   for stem, number, _, _ in PAGES)


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    builders = {'figure1': figure1, 'figure2': figure2, 'figure3': lambda: profile('full'),
                'figure4': lambda: ecdf('full'), 'figureS1': lambda: profile('chno'),
                'figureS2': lambda: ecdf('chno'), 'figureS3': tails}
    config = {'responsive': True, 'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'}}
    for stem, number, title, caption in PAGES:
        plot = builders[stem]().to_html(full_html=False, include_plotlyjs='cdn', config=config)
        body = f'<h1>{number}. {title}</h1>\n<p>{caption}</p>\n<p class="hint">{HINTS[stem in LEGEND_PAGES]}</p>\n{plot}'
        (SITE / f'{stem}.html').write_text(PAGE.format(title=f'{number}. {title}', nav=nav(stem), body=body, repo=REPO),
                                          encoding='utf-8')
        print('Wrote', SITE / f'{stem}.html')
    items = ''.join(f'<li><a href="{stem}.html">{number}. {title}</a></li>' for stem, number, title, _ in PAGES)
    body = ('<h1>Interactive figures</h1>\n<p>Interactive versions of the data figures in our benchmark of pretrained '
            'machine-learning interatomic potentials on the DORTS-9K reaction dataset. Fifteen checkpoints are compared '
            'with the ωB97M-V/def2-TZVP reference on all 909,077 configurations, and seventeen on the CHNO subset.</p>\n'
            f'<ul>{items}</ul>\n<p>Figure 5 shows VESTA renderings of six reactions and appears only in the paper.</p>')
    (SITE / 'index.html').write_text(PAGE.format(title='DORTS-9K MLIP benchmark', nav=nav(None), body=body, repo=REPO),
                                     encoding='utf-8')
    print('Wrote', SITE / 'index.html')


if __name__ == '__main__':
    main()
