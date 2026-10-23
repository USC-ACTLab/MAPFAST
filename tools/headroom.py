'''
Where the headroom of solver selection lies on a dataset: how much less total runtime the oracle (always the fastest
solver) needs than always using the best single solver (the lowest total runtime, e.g. bcp2), broken down by instance,
by the winning solver, by agent count, by map and by the size of the win, with a bootstrap confidence interval, and
what each two-solver portfolio (the best single solver and one other) would save. Unsolved runs cost the dataset's
time limit, as in plots.py. Writes to --out (default experiments/headroom/<dataset path below datasets/>/<split>/):

	headroom.md		the tables
	headroom.png		how concentrated the savings are, and the savings by agent count, winning solver and size of the win
	scatter.png		the best single solver's time against the fastest other solver's, per instance

	python tools/headroom.py datasets/movingai_10-500-agents_300s/eight-solvers --split test

--split train, valid or test keeps those instances of main.py's split (the models' test set is `test`); all (default)
keeps every instance. Run it in the models' Python environment (pytorch.venvsource).
'''
import argparse
import json
import os
import re
import sys

import numpy as np

from benchmark_dataset import map_name
from plots import SEQUENTIAL, SERIES_1, SURFACE, TEXT, TEXT_2, matplotlib, plt, save

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from utils import read_config

WIN_BINS = (0, 0.01, 0.1, 1, 10, 60, np.inf)

def table(header, rows):
	'''
	Returns: List of the lines of a markdown table
	'''
	return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]

def breakdown(keys, saving, reference):
	'''
	Returns: List of (key, instances, minutes of the reference solver, minutes saved) for each distinct key, sorted
	'''
	rows = []
	for key in sorted(set(keys)):
		mask = keys == key
		rows.append((key, int(mask.sum()), reference[mask].sum() / 60, saving[mask].sum() / 60))
	return rows

def analyse(names, times, solvers, timeout, out):
	n = len(names)
	default = int(times.sum(0).argmin())
	reference = times[:, default]
	oracle = times.min(1)
	saving = reference - oracle
	total = saving.sum() / 60
	d = solvers[default]
	# the fastest other solver of each instance, and the winner (the default on a tie)
	others = np.delete(np.arange(len(solvers)), default)
	best_other = others[times[:, others].argmin(1)]
	winner = np.where(saving > 0, best_other, default)
	wins = saving > 0

	rng = np.random.default_rng(0)
	samples = rng.integers(0, n, size=(2000, n))
	shares = saving[samples].sum(1) / reference[samples].sum(1)
	ranked = np.sort(saving)[::-1].cumsum()
	holding = [int(np.searchsorted(ranked, q * ranked[-1]) + 1) for q in (0.5, 0.8, 0.9)]
	lines = ['# Headroom over always {}'.format(d), '',
			 '{} instances; unsolved runs count {:g} s. The oracle always picks the fastest solver.'.format(n, timeout), '']
	lines += table(('Always ' + d + ' (min)', 'Oracle (min)', 'Savings (min)', 'Savings (%)', '95% bootstrap CI (%)',
					'Instances with a saving', 'Instances holding 50% / 80% / 90% of it'),
				   [('{:.1f}'.format(reference.sum() / 60), '{:.1f}'.format(oracle.sum() / 60), '{:.1f}'.format(total),
					 '{:.1%}'.format(total * 60 / reference.sum()),
					 '{:.1%}–{:.1%}'.format(*np.percentile(shares, [2.5, 97.5])), int(wins.sum()),
					 ' / '.join(map(str, holding)))])

	failed = reference >= timeout
	lines += ['', '## Whether {} solves it'.format(d), '']
	lines += table(('{} ...'.format(d), 'Instances', 'Another solver faster', 'Savings (min)', 'Share'),
				   [(label, int(mask.sum()), int((mask & wins).sum()), '{:.1f}'.format(saving[mask].sum() / 60),
					 '{:.0%}'.format(saving[mask].sum() / 60 / total)) for label, mask in
					(('solves it', ~failed), ('times out', failed))])

	lines += ['', '## Winning solver', '',
			  'Instances on which each solver beats {d}, what it saves there, and what the two-solver portfolio '
			  '{{{d}, solver}} would save with a perfect choice between the two.'.format(d=d), '']
	rows = []
	for s in others:
		mask = winner == s
		pair = (reference - np.minimum(reference, times[:, s])).sum() / 60
		rows.append((solvers[s], int(mask.sum()), saving[mask].sum() / 60,
					 np.median(saving[mask]) if mask.any() else 0, pair))
	rows.sort(key=lambda r: -r[2])
	lines += table(('Solver', 'Wins', 'Savings (min)', 'Share', 'Median win (s)', '{' + d + ', solver} saves (min)'),
				   [(s, w, '{:.1f}'.format(m), '{:.0%}'.format(m / total), '{:.2f}'.format(med), '{:.1f}'.format(p))
					for s, w, m, med, p in rows])

	agents = np.array([int(re.search(r'_agents(\d+)\.yaml$', name).group(1)) for name in names])
	by_agents = breakdown(agents, saving, reference)
	lines += ['', '## Agent count', '']
	lines += table(('Agents', 'Instances', 'Always ' + d + ' (min)', 'Savings (min)', 'Savings (% of ' + d + ')', 'Share'),
				   [(k, c, '{:.1f}'.format(r), '{:.1f}'.format(m), '{:.1%}'.format(m / r), '{:.0%}'.format(m / total))
					for k, c, r, m in by_agents])

	by_map = sorted(breakdown(np.array([map_name(name) for name in names]), saving, reference), key=lambda r: -r[3])
	lines += ['', '## Maps (the 15 with the most savings)', '']
	lines += table(('Map', 'Instances', 'Always ' + d + ' (min)', 'Savings (min)', 'Savings (% of ' + d + ')', 'Share'),
				   [(k, c, '{:.1f}'.format(r), '{:.1f}'.format(m), '{:.1%}'.format(m / r), '{:.0%}'.format(m / total))
					for k, c, r, m in by_map[:15]])

	size = np.digitize(saving, WIN_BINS[1:-1], right=True)
	labels = ['≤ 0.01 s', '0.01–0.1 s', '0.1–1 s', '1–10 s', '10–60 s', '> 60 s']
	by_size = [(labels[b], int((wins & (size == b)).sum()), saving[wins & (size == b)].sum() / 60) for b in range(len(labels))]
	lines += ['', '## Size of the win', '']
	lines += table(('Saving per instance', 'Instances', 'Savings (min)', 'Share'),
				   [(l, c, '{:.1f}'.format(m), '{:.0%}'.format(m / total)) for l, c, m in by_size])

	os.makedirs(out, exist_ok=True)
	with open(os.path.join(out, 'headroom.md'), 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print('\n'.join(lines))

	fig, axes = plt.subplots(2, 2, figsize=(13, 9))
	curve = axes[0, 0]
	curve.plot(np.arange(1, len(ranked) + 1), ranked / 60, color=SERIES_1)
	for q, k in zip((0.5, 0.8, 0.9), holding):
		curve.plot(k, ranked[k - 1] / 60, 'o', color=SERIES_1, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
		curve.annotate('{:.0%} in {} instances'.format(q, k), (k, ranked[k - 1] / 60), xytext=(8, -12),
					   textcoords='offset points', color=TEXT)
	curve.set_xscale('log')
	curve.set_xlim(right=max(int(wins.sum()), 2))
	curve.set_ylim(0, total * 1.05)
	curve.set(xlabel='Instances, largest saving first (log scale)', ylabel='Cumulative savings (min)',
			  title='How concentrated the {:.1f} min of savings are'.format(total))

	bars = axes[0, 1]
	bars.bar([str(k) for k, *_ in by_agents], [m for *_, m in by_agents], 0.7, color=SERIES_1)
	bars.tick_params(axis='x', rotation=90)
	bars.grid(axis='x', visible=False)
	bars.set(xlabel='Agents', ylabel='Savings (min)', title='Savings by agent count')

	solver_bars = axes[1, 0]
	order = rows[::-1]
	solver_bars.barh([r[0] for r in order], [r[2] for r in order], 0.7, color=SERIES_1)
	for i, r in enumerate(order):
		solver_bars.annotate('{} wins'.format(r[1]), (r[2], i), xytext=(4, 0), textcoords='offset points',
							 va='center', color=TEXT_2)
	solver_bars.grid(axis='y', visible=False)
	solver_bars.set_xlim(right=max(r[2] for r in rows) * 1.25 or 1)
	solver_bars.set(xlabel='Savings (min)', title='Savings by the solver that beats {}'.format(d))

	size_bars = axes[1, 1]
	size_bars.bar(['{}\n{} inst.'.format(l, c) for l, c, _ in by_size], [m for _, _, m in by_size], 0.7, color=SERIES_1)
	size_bars.grid(axis='x', visible=False)
	size_bars.set(xlabel='Saving per instance', ylabel='Savings (min)', title='Savings by the size of the win')
	save(fig, os.path.join(out, 'headroom.png'))

	fig, ax = plt.subplots(figsize=(7.5, 6.5))
	floor = 1e-3
	x, y = np.clip(reference, floor, None), np.clip(times[np.arange(n), best_other], floor, None)
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list('sequential', SEQUENTIAL[1:])
	points = ax.scatter(x, y, c=agents, cmap=cmap, s=10, linewidths=0)
	ax.plot([floor, timeout], [floor, timeout], color=TEXT_2, linestyle='--', linewidth=1)
	ax.set(xscale='log', yscale='log', xlabel='Always {} (s; {:g} = unsolved)'.format(d, timeout),
		   ylabel='Fastest other solver (s)', title='{} against the fastest other solver, per instance'.format(d))
	ax.annotate('another solver faster: {} instances, {:.1f} min saved'.format(int(wins.sum()), total), (0.98, 0.03),
				xycoords='axes fraction', ha='right', color=TEXT)
	ax.annotate('{} fastest: {} instances'.format(d, n - int(wins.sum())), (0.02, 0.88), xycoords='axes fraction',
				va='top', color=TEXT, bbox=dict(boxstyle='square,pad=0.2', facecolor=SURFACE, edgecolor='none'))
	bar = fig.colorbar(points, ax=ax, fraction=0.04)
	bar.set_label('Agents', color=TEXT_2)
	bar.outline.set_visible(False)
	save(fig, os.path.join(out, 'scatter.png'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Where the oracle saves time over the best single solver')
	parser.add_argument('dataset', help='Dataset folder, e.g. datasets/movingai_10-500-agents_300s/eight-solvers')
	parser.add_argument('--split', choices=['all', 'train', 'valid', 'test'], default='all',
						help='Instances of main.py\'s split to analyse (default: all)')
	parser.add_argument('--out', default=None, help='Output folder (default: experiments/headroom/<dataset>/<split>)')
	args = parser.parse_args()

	dataset = os.path.normpath(args.dataset)
	config = read_config(dataset)
	with open(config['yaml_details']) as f:
		yaml_details = json.load(f)
	names = sorted(yaml_details)
	if args.split != 'all':
		from train_gnn import split
		train, valid, test, _, _ = split(dataset)
		names = {'train': train, 'valid': valid, 'test': test}[args.split]
	solvers = sorted(config['mapping'], key=config['mapping'].get)
	times = np.array([[yaml_details[name][s] for s in solvers] for name in names], dtype=np.float64)
	times = np.where(times == -1, config['timeout'], times)
	rel = os.path.relpath(dataset, 'datasets')
	analyse(np.array(names), times, solvers, config['timeout'],
			args.out or os.path.join('experiments', 'headroom', rel, args.split))
