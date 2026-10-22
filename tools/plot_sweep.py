'''
The figure of a MAPFAST-GNN hyperparameter sweep (models/train_gnn.py), from the results.json of every configuration:
the validation speedup per epoch of every configuration (the best one bold), and test against validation speedup.
Speedup is the total runtime of always using the default solver (e.g. bcp2) divided by the configuration's; the dashed
line at 1 is that solver.

	python tools/plot_sweep.py experiments/gnn_sweep/movingai_10-500-agents_300s/eight-solvers

writes sweep.png in that folder (train_gnn.py select also draws it).
'''
import argparse
import glob
import json
import os

import matplotlib

from plots import SERIES_1, SERIES_2, SERIES_3, SURFACE, TEXT, TEXT_2, plt, save

def plot_sweep(results, out):
	'''
	Draws the sweep figure of `results`, the results.json of every configuration ranked best first
	'''
	colors = dict(zip(sorted(set(r['config']['kind'] for r in results)), (SERIES_1, SERIES_2, SERIES_3)))
	best = results[0]
	fig, (curves, generalisation) = plt.subplots(1, 2, figsize=(14, 5))
	for r in results:
		curves.plot([h['valid_speedup'] for h in r['history']], color=colors[r['config']['kind']], linewidth=1, alpha=0.4)
	curves.plot([h['valid_speedup'] for h in best['history']], color=colors[best['config']['kind']], linewidth=2.5,
				label='best: ' + best['name'])
	for kind, color in colors.items():
		runs = [r for r in results if r['config']['kind'] == kind]
		generalisation.scatter([r['valid_speedup'] for r in runs], [r['test_speedup'] for r in runs], s=20, color=color,
							   edgecolor=SURFACE, linewidth=0.5, label=kind)
	for ax in (curves, generalisation):
		ax.axhline(1.0, color=TEXT_2, linewidth=1, linestyle='--')
	generalisation.scatter([best['valid_speedup']], [best['test_speedup']], s=80, facecolor='none', edgecolor=TEXT)
	curves.set(title='Validation speedup per epoch', xlabel='Epoch', ylabel='Speedup over always {}'.format(best['baseline']['solver']))
	generalisation.set(title='Test against validation speedup (circled: selected)', xlabel='Validation speedup', ylabel='Test speedup')
	curves.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	curves.legend(loc='lower right')
	generalisation.legend(loc='lower right')
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
