'''
One dense figure of a MAPFAST-GNN hyperparameter sweep (models/train_gnn.py), from the results.json of every configuration:

	row 1	validation speedup per epoch of every configuration, one panel per layer kind (the best one bold), and what
			the best configuration picks on the validation set at each epoch (does it collapse onto one solver?)
	row 2	best validation speedup by width and depth, one heatmap per layer kind, and speedup against model size
	row 3	validation speedup by the value of each hyperparameter, and validation against test speedup

Speedup is the total runtime of always using the default solver (the one with the lowest total training runtime, e.g.
bcp2) divided by the configuration's; the dashed line at 1 is that solver, which every configuration has to beat.

	python tools/plot_sweep.py experiments/gnn_sweep/movingai_10-500-agents_300s/eight-solvers

writes sweep.png in that folder (train_gnn.py select also draws it).
'''
import argparse
import glob
import json
import os

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from plots import GRID, NEUTRAL, SEQUENTIAL, SERIES_1, SERIES_2, SERIES_3, SERIES_4, SERIES_5, SERIES_6, SERIES_7, SERIES_8, \
	SURFACE, TEXT, TEXT_2, plt, save

SERIES = [SERIES_1, SERIES_2, SERIES_3, SERIES_4, SERIES_5, SERIES_6, SERIES_7, SERIES_8]
HYPERPARAMETERS = ('kind', 'hidden', 'layers', 'lr', 'loss')

def baseline_line(ax, baseline, label=True):
	ax.axhline(1.0, color=TEXT_2, linewidth=1, linestyle='--')
	if label:
		ax.annotate('always {}'.format(baseline['solver']), (0, 1.0), xycoords=('axes fraction', 'data'),
					xytext=(4, -10), textcoords='offset points', color=TEXT_2, fontsize=8)

def plot_curves(ax, results, kind, color, baseline, ylim):
	'''
	Validation speedup per epoch of every configuration of one layer kind, the darker the wider; the best one bold
	'''
	runs = [r for r in results if r['config']['kind'] == kind]
	widths = sorted(set(r['config']['hidden'] for r in results))
	shade = {w: SEQUENTIAL[min(len(SEQUENTIAL) - 1, 2 + 2 * i)] for i, w in enumerate(widths)}
	for r in runs:
		speedup = [h['valid_speedup'] for h in r['history']]
		ax.plot(speedup, color=shade[r['config']['hidden']], linewidth=1, alpha=0.8,
				linestyle='-' if r['config']['loss'] == 'ce' else ':')
	if runs:
		best = max(runs, key=lambda r: (r['valid_speedup'], r['valid_accuracy']))
		ax.plot([h['valid_speedup'] for h in best['history']], color=color, linewidth=2.5)
		ax.annotate('{}  {:.3f}'.format(best['name'], best['valid_speedup']), (1, 0), xycoords='axes fraction',
					xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom', fontsize=8, color=TEXT)
	baseline_line(ax, baseline)
	ax.set_ylim(*ylim)
	ax.set_title('{}: validation speedup per epoch'.format(kind.upper()), fontsize=10)
	ax.set_xlabel('Epoch')
	ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

def plot_picks(ax, best):
	'''
	Share of the validation set the best configuration gives to each solver at each epoch (stacked)
	'''
	shares = np.array([h['valid_pick_share'] for h in best['history']], dtype=float)
	shares /= shares.sum(axis=1, keepdims=True)
	epochs = np.arange(len(shares))
	used = [i for i in range(shares.shape[1]) if shares[:, i].max() > 0]
	ax.stackplot(epochs, shares[:, used].T, colors=[SERIES[i % len(SERIES)] for i in used],
				 labels=[best['solvers'][i] for i in used], edgecolor=SURFACE, linewidth=0.5)
	ax.set_xlim(0, max(len(shares) - 1, 1))
	ax.set_ylim(0, 1)
	ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
	ax.set_title('Best configuration: its validation picks', fontsize=10)
	ax.set_xlabel('Epoch')
	ax.legend(loc='lower left', fontsize=7, ncol=2, frameon=True, facecolor=SURFACE, edgecolor=GRID)

def plot_heatmap(ax, results, kind, values_min, values_max):
	'''
	Best validation speedup of each width and depth (over learning rates and losses) of one layer kind
	'''
	widths = sorted(set(r['config']['hidden'] for r in results))
	depths = sorted(set(r['config']['layers'] for r in results))
	grid = np.full((len(depths), len(widths)), np.nan)
	for r in results:
		if r['config']['kind'] == kind:
			i, j = depths.index(r['config']['layers']), widths.index(r['config']['hidden'])
			grid[i, j] = np.nanmax([grid[i, j], r['valid_speedup']])
	cmap = LinearSegmentedColormap.from_list('sequential', SEQUENTIAL)
	cmap.set_bad(SURFACE)
	ax.imshow(grid, cmap=cmap, vmin=values_min, vmax=values_max, aspect='auto', origin='lower')
	for i in range(len(depths)):
		for j in range(len(widths)):
			if not np.isnan(grid[i, j]):
				dark = (grid[i, j] - values_min) / max(values_max - values_min, 1e-9) > 0.6
				ax.text(j, i, '{:.3f}'.format(grid[i, j]), ha='center', va='center', fontsize=8, color=SURFACE if dark else TEXT)
	ax.set_xticks(range(len(widths)))
	ax.set_xticklabels(widths)
	ax.set_yticks(range(len(depths)))
	ax.set_yticklabels(depths)
	ax.set_xlabel('Width (hidden features)')
	ax.set_ylabel('Depth (layers)')
	ax.grid(False)
	ax.set_title('{}: best validation speedup'.format(kind.upper()), fontsize=10)

def plot_size(ax, results, colors, baseline):
	'''
	Validation speedup against the number of parameters, by layer kind
	'''
	for kind, color in colors.items():
		runs = [r for r in results if r['config']['kind'] == kind]
		ax.scatter([r['parameters'] for r in runs], [r['valid_speedup'] for r in runs], s=22, color=color,
				   edgecolor=SURFACE, linewidth=0.5, label=kind)
	baseline_line(ax, baseline)
	ax.set_xscale('log')
	ax.set_xlabel('Parameters')
	ax.set_title('Validation speedup by model size', fontsize=10)
	ax.legend(loc='lower right', fontsize=8)

def plot_marginals(fig, cell, results, colors, baseline):
	'''
	Validation speedup of every configuration, grouped by the value of each hyperparameter (one small panel each)
	'''
	inner = cell.subgridspec(1, len(HYPERPARAMETERS), wspace=0.08)
	axes = []
	for k, name in enumerate(HYPERPARAMETERS):
		ax = fig.add_subplot(inner[0, k], sharey=axes[0] if axes else None)
		values = sorted(set(r['config'][name] for r in results), key=lambda v: (isinstance(v, str), v))
		for i, v in enumerate(values):
			runs = [r for r in results if r['config'][name] == v]
			jitter = (np.random.default_rng(i).random(len(runs)) - 0.5) * 0.4
			ax.scatter(i + jitter, [r['valid_speedup'] for r in runs], s=10,
					   color=[colors[r['config']['kind']] for r in runs], alpha=0.8, linewidth=0)
			ax.plot([i - 0.3, i + 0.3], [np.median([r['valid_speedup'] for r in runs])] * 2, color=TEXT, linewidth=2)
		baseline_line(ax, baseline, label=k == 0)
		ax.set_xticks(range(len(values)))
		ax.set_xticklabels(['{:g}'.format(v) if isinstance(v, float) else v for v in values], fontsize=8)
		ax.set_xlim(-0.6, len(values) - 0.4)
		ax.set_xlabel(name)
		ax.grid(axis='x', visible=False)
		if k:
			ax.tick_params(labelleft=False)
		else:
			ax.set_ylabel('Validation speedup')
		axes.append(ax)
	axes[0].set_title('Validation speedup by hyperparameter (bar: median)', fontsize=10, loc='left')

def plot_generalisation(ax, results, colors):
	'''
	Test against validation speedup: the selection by validation speedup is only sound if they agree
	'''
	for kind, color in colors.items():
		runs = [r for r in results if r['config']['kind'] == kind]
		ax.scatter([r['valid_speedup'] for r in runs], [r['test_speedup'] for r in runs], s=18, color=color,
				   edgecolor=SURFACE, linewidth=0.5)
	low = min(min(r['valid_speedup'], r['test_speedup']) for r in results)
	high = max(max(r['valid_speedup'], r['test_speedup']) for r in results)
	pad = (high - low) * 0.05 + 1e-3
	ax.plot([low - pad, high + pad], [low - pad, high + pad], color=TEXT_2, linewidth=1, linestyle='--')
	best = results[0]
	ax.scatter([best['valid_speedup']], [best['test_speedup']], s=80, facecolor='none', edgecolor=TEXT, linewidth=1.5)
	ax.annotate('selected', (best['valid_speedup'], best['test_speedup']), xytext=(6, -10), textcoords='offset points',
				fontsize=8, color=TEXT)
	ax.set_xlabel('Validation speedup')
	ax.set_ylabel('Test speedup')
	ax.set_title('Test against validation speedup', fontsize=10)

def plot_sweep(results, out):
	'''
	Draws the sweep figure of `results`, the results.json of every configuration ranked best first
	'''
	results = [r for r in results if r.get('history')]
	if not results:
		print('no per-epoch history in the results, so', out, 'is not drawn')
		return
	baseline = results[0]['baseline']
	kinds = sorted(set(r['config']['kind'] for r in results))
	colors = {k: SERIES[i] for i, k in enumerate(kinds)}
	speedups = [h['valid_speedup'] for r in results for h in r['history']] + [1.0]
	pad = (max(speedups) - min(speedups)) * 0.05 + 0.005
	ylim = (min(speedups) - pad, max(speedups) + pad)
	best_values = [r['valid_speedup'] for r in results]

	fig = plt.figure(figsize=(5 * max(4, len(kinds) + 1), 14))
	columns = len(kinds) + 1
	grid = fig.add_gridspec(3, columns, hspace=0.42, wspace=0.28, top=0.92, bottom=0.05)
	for i, kind in enumerate(kinds):
		plot_curves(fig.add_subplot(grid[0, i]), results, kind, colors[kind], baseline, ylim)
		plot_heatmap(fig.add_subplot(grid[1, i]), results, kind, min(best_values), max(best_values))
	plot_picks(fig.add_subplot(grid[0, columns - 1]), results[0])
	plot_size(fig.add_subplot(grid[1, columns - 1]), results, colors, baseline)
	plot_marginals(fig, grid[2, :columns - 1], results, colors, baseline)
	plot_generalisation(fig.add_subplot(grid[2, columns - 1]), results, colors)
	best = results[0]
	fig.suptitle('MAPFAST-GNN sweep: {} configurations, best {} (speedup over always {}: validation {:.3f}, test {:.3f})'.format(
		len(results), best['name'], baseline['solver'], best['valid_speedup'], best['test_speedup']),
		x=0.01, ha='left', fontsize=13, fontweight='bold', color=TEXT)
	fig.text(0.01, 0.955, 'Thin lines: every configuration of the kind, darker = wider, dotted = regret loss; '
			 'dashed: always using the default solver (speedup 1)', fontsize=9, color=TEXT_2)
	save(fig, out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Draw the figure of a GNN hyperparameter sweep')
	parser.add_argument('out', help='Sweep folder, with <configuration>/results.json')
	args = parser.parse_args()
	results = []
	for path in glob.glob(os.path.join(args.out, '*', 'results.json')):
		with open(path) as f:
			results.append(json.load(f))
	results.sort(key=lambda r: (r['valid_speedup'], r['valid_accuracy']), reverse=True)
	plot_sweep(results, os.path.join(args.out, 'sweep.png'))
