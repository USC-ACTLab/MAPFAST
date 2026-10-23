'''
Graphs of a MAPFAST training run and of its test predictions, and a table of the accuracy, coverage and total runtime
of the model, of every solver and of the oracle:

	loss.png				training loss per step and per epoch
	runtime.png			total runtime of the model, of every solver and of the oracle
	confusion.png			predicted against actual fastest solver

	python tools/plots.py -C datasets/movingai_pilot/nine-solvers --log logs/MAPFASTv2_1234.out

With --run, several trained models are compared on the same graphs instead (see plot_comparison), all
evaluated under the same time limit (see apply_time_limit):

	python tools/plots.py --run MAPFAST datasets/mapfast_paper/four-solvers logs/MAPFAST_1234.out --run MAPFASTv2 datasets/movingai_pilot/nine-solvers logs/MAPFASTv2_1235.out -o logs/MAPFAST_vs_MAPFASTv2_1236

The graphs are written next to the log, to logs/<name>_<jobid>_plots/ for --log logs/<name>_<jobid>.out
(logs/<dataset>_plots/ without --log). Without --log, loss.png is skipped; without the
prediction_output of the config (written by main.py -T 0), only loss.png is drawn.
'''
import argparse
import json
import os
import re
import sys

# the model code (utils.py, MAPFAST.py) is in ../models
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Reference palette of the dataviz method: eight categorical slots, neutral ink and surfaces
SERIES_1 = '#2a78d6'
SERIES_2 = '#eb6834'
SERIES_3 = '#1baf7a'
SERIES_4 = '#eda100'
SERIES_5 = '#e87ba4'
SERIES_6 = '#008300'
SERIES_7 = '#6250d6'
SERIES_8 = '#e34948'
NEUTRAL = '#b5b3ad'
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e6e5e1'
SEQUENTIAL = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
	'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
	'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
	'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
	'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'text.color': TEXT,
	'legend.frameon': False, 'font.size': 10, 'lines.linewidth': 2,
})

STEP_LINE = re.compile(r'Epoch (\d+) \| Step (\d+)/(\d+) \| avg loss ([\d.]+) \| batch loss ([\d.]+)')
EPOCH_LINE = re.compile(r'Epoch (\d+) complete \| mean loss ([\d.]+)')

def read_training_log(log_file):
	'''
	Reads the progress lines that main.py prints while training.

	Returns: Json object with
		'steps' -> list of (global step, running mean loss of the epoch, batch loss)
		'epochs' -> list of (global step at the end of the epoch, mean loss of the epoch)
	'''
	log = {'steps': [], 'epochs': []}
	steps_per_epoch = None
	with open(log_file) as f:
		for line in f:
			m = STEP_LINE.search(line)
			if m:
				epoch, step, steps_per_epoch = int(m.group(1)), int(m.group(2)), int(m.group(3))
				log['steps'].append((epoch * steps_per_epoch + step, float(m.group(4)), float(m.group(5))))
				continue
			m = EPOCH_LINE.search(line)
			if m and steps_per_epoch:
				log['epochs'].append(((int(m.group(1)) + 1) * steps_per_epoch, float(m.group(2))))
	return log

def plot_loss(log, out):
	'''
	Line chart of the training loss: the running mean of each epoch and the mean of each finished epoch.
	'''
	fig, ax = plt.subplots(figsize=(8, 4.5))
	steps = np.array(log['steps'])
	ax.plot(steps[:, 0], steps[:, 1], color=SERIES_1, label='Training (running mean of the epoch)')
	if log['epochs']:
		epochs = np.array(log['epochs'])
		ax.plot(epochs[:, 0], epochs[:, 1], 'o', color=SERIES_1, markersize=8, markeredgecolor=SURFACE,
				markeredgewidth=2, label='Training (epoch mean)')
		last = epochs[-1]
		ax.annotate('{:.3f}'.format(last[1]), last, xytext=(6, 6), textcoords='offset points', color=TEXT_2)
	ax.set_title('Training loss')
	ax.set_xlabel('Step')
	ax.set_ylabel('Loss')
	ax.set_ylim(bottom=0)
	ax.legend(loc='upper right')
	save(fig, out)

def selections(yaml_details, predictions, mapping):
	'''
	The solver each selector picks for each test instance: one entry per solver of mapping (always that
	solver), 'Model' (the prediction) and 'Oracle' (the fastest solver).

	Returns: Tuple of
		1. List of the yaml_details keys of the test instances, one per prediction
		2. Json object of selector name -> list of the solver it picks for each instance
	'''
	inv_mapping = {v: k for k, v in mapping.items()}
	keys = [k[:k.rindex('_')] for k in predictions]
	picks = {s: [s] * len(keys) for s in mapping}
	picks['Model'] = [inv_mapping[p['best'][0]] for p in predictions.values()]
	picks['Oracle'] = [yaml_details[k]['SOLVER'] for k in keys]
	return keys, picks

def scores(yaml_details, keys, picks, timeout):
	'''
	Returns: Json object of selector name -> (accuracy, coverage, total runtime in minutes), where an
	unsolved instance costs `timeout` seconds
	'''
	result = {}
	for name, chosen in picks.items():
		fastest = [yaml_details[k]['SOLVER'] == s for k, s in zip(keys, chosen)]
		times = [yaml_details[k][s] for k, s in zip(keys, chosen)]
		result[name] = (np.mean(fastest), np.mean([t != -1 for t in times]),
						sum(timeout if t == -1 else t for t in times) / 60)
	return result

def print_scores(result, timeout):
	'''
	Prints the accuracy (picks the fastest solver), coverage (picks a solver that solves the instance) and total runtime
	of every selector of `result` (see scores), fastest first
	'''
	print('{:<24} {:>9} {:>9} {:>14}'.format('', 'accuracy', 'coverage', 'runtime (min)'))
	for name in sorted(result, key=lambda n: result[n][2]):
		print('{:<24} {:>9.1%} {:>9.1%} {:>14.1f}'.format(name, *result[name]))
	print('(an unsolved instance counts {:g} s)'.format(timeout))

def plot_runtime(result, timeout, out):
	'''
	Horizontal bars of the total runtime of always using each solver, of the model's picks and of the
	oracle (always the fastest solver), sorted from fastest to slowest.
	'''
	names = sorted(result, key=lambda n: result[n][2], reverse=True)
	colors = [SERIES_1 if n == 'Model' else TEXT_2 if n == 'Oracle' else NEUTRAL for n in names]
	fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.5))
	ax.barh(names, [result[n][2] for n in names], 0.7, color=colors)
	for i, n in enumerate(names):
		ax.annotate('{:.1f}'.format(result[n][2]), (result[n][2], i), xytext=(4, 0), textcoords='offset points',
					va='center', color=TEXT if n in ('Model', 'Oracle') else TEXT_2)
	for label in ax.get_yticklabels():
		if label.get_text() in ('Model', 'Oracle'):
			label.set_fontweight('bold')
			label.set_color(TEXT)
	ax.grid(axis='y', visible=False)
	ax.set_xlim(right=max(r[2] for r in result.values()) * 1.12)
	ax.set_xlabel('Total runtime on the test set (minutes; an unsolved instance counts {:g} s)'.format(timeout))
	ax.set_title('Total runtime: model against single solvers and the oracle')
	save(fig, out)

def plot_confusion(keys, picks, mapping, out):
	'''
	Heatmap of the actual fastest solver (rows) against the model's pick (columns), shaded by the share of
	each row and labelled with the counts.
	'''
	solvers = list(mapping)
	index = {s: i for i, s in enumerate(solvers)}
	counts = np.zeros((len(solvers), len(solvers)), dtype=int)
	for actual, predicted in zip(picks['Oracle'], picks['Model']):
		counts[index[actual], index[predicted]] += 1
	shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list('sequential', SEQUENTIAL)
	size = 0.6 * len(solvers) + 2.5
	fig, ax = plt.subplots(figsize=(size + 1, size))
	image = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
	for i in range(len(solvers)):
		for j in range(len(solvers)):
			if counts[i, j]:
				ax.text(j, i, counts[i, j], ha='center', va='center', fontsize=9,
						color=SURFACE if shares[i, j] > 0.55 else TEXT)
	ax.set_xticks(range(len(solvers)))
	ax.set_xticklabels(solvers, rotation=45, ha='right')
	ax.set_yticks(range(len(solvers)))
	ax.set_yticklabels(['{} ({})'.format(s, n) for s, n in zip(solvers, counts.sum(axis=1))])
	ax.set_xlabel('Model\'s pick')
	ax.set_ylabel('Actual fastest solver (test instances)')
	ax.grid(False)
	ax.spines[:].set_visible(False)
	bar = fig.colorbar(image, ax=ax, fraction=0.04, format=matplotlib.ticker.PercentFormatter(1))
	bar.set_label('Share of the row', color=TEXT_2)
	bar.outline.set_visible(False)
	ax.set_title('Predicted against actual fastest solver')
	save(fig, out)

def load_run(config_path):
	'''
	Loads the dataset and test predictions of a dataset folder (or its config.json).

	Returns: Json object with yaml_details, mapping, timeout, keys, picks and result
	(see selections and scores), or None if main.py -T 0 has not written the predictions yet
	'''
	from utils import read_config
	config = read_config(config_path)
	if not os.path.exists(config['prediction_output']):
		print('no predictions at', config['prediction_output'], '- run main.py -T 0 first')
		return None
	run = {'mapping': config['mapping'], 'timeout': config['timeout']}
	for name in ('yaml_details', 'prediction_output'):
		with open(config[name]) as f:
			run[name] = json.load(f)
	run['keys'], run['picks'] = selections(run['yaml_details'], run['prediction_output'], run['mapping'])
	run['result'] = scores(run['yaml_details'], run['keys'], run['picks'], run['timeout'])
	return run

def apply_time_limit(run, limit):
	'''
	Evaluates a run as if its benchmark had used a time limit of `limit` seconds: runs slower than that
	count as unsolved (costing `limit` seconds), and test instances that no solver solves within it are
	left out, as MAPF Benchmarking leaves out instances that no solver solved.

	Returns: Copy of run (see load_run) with yaml_details, keys, picks, result and timeout updated,
	and 'dropped' the number of test instances left out
	'''
	yaml_details = {}
	for key, record in run['yaml_details'].items():
		yaml_details[key] = {s: (-1 if s != 'SOLVER' and t > limit else t) for s, t in record.items()}
	keep = [i for i, k in enumerate(run['keys']) if yaml_details[k][yaml_details[k]['SOLVER']] != -1]
	keys = [run['keys'][i] for i in keep]
	picks = {name: [chosen[i] for i in keep] for name, chosen in run['picks'].items()}
	return {**run, 'yaml_details': yaml_details, 'keys': keys, 'picks': picks, 'timeout': limit,
			'result': scores(yaml_details, keys, picks, limit), 'dropped': len(run['keys']) - len(keys)}

def plot_comparison(runs, output, limit=None):
	'''
	Graphs that put several trained models on the same axes, for `runs` a list of (name, color, run, log),
	with run as returned by load_run (or None) and log as returned by read_training_log (or None):

		compare_loss.png			mean training loss of each epoch
		compare_total_runtime.png	total runtime of each model, of the oracle of each portfolio and of every solver (see plot_total_runtime)
		compare_runtime.png		total runtime of each model relative to its oracle and to its fastest single solver
		compare_accuracy.png		accuracy against coverage of each model and of every solver (see plot_accuracy)

	`limit` is the common time limit the runs were evaluated under (see apply_time_limit).
	'''
	logged = [(name, color, log) for name, color, _, log in runs if log and log['epochs']]
	if logged:
		fig, ax = plt.subplots(figsize=(8, 4.5))
		for name, color, log in logged:
			epochs = np.array(log['epochs'])
			x = np.arange(len(epochs))
			ax.plot(x, epochs[:, 1], 'o-', color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
			ax.annotate('{:.3f}'.format(epochs[-1, 1]), (x[-1], epochs[-1, 1]), xytext=(6, 6), textcoords='offset points', color=TEXT_2)
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Mean training loss')
		ax.set_ylim(bottom=0)
		ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
		ax.set_title('Training loss')
		ax.legend(loc='upper right')
		save(fig, os.path.join(output, 'compare_loss.png'))

	tested = [(name, color, run) for name, color, run, _ in runs if run]
	if tested:
		plot_total_runtime(tested, limit, os.path.join(output, 'compare_total_runtime.png'))
		plot_relative_runtime(tested, os.path.join(output, 'compare_runtime.png'))
		plot_accuracy(tested, os.path.join(output, 'compare_accuracy.png'))

def plot_total_runtime(tested, limit, out):
	'''
	Horizontal bars, as in runtime.png, of the total runtime of each model (in its color), of the oracle of
	each solver portfolio (named after the first model with that portfolio) and of always using each solver,
	sorted from fastest to slowest, for `tested` a list of (name, color, run). The single solvers' runtimes
	come from the first run whose portfolio has the solver, so they only match the models' when the runs
	share a test set.
	'''
	if len(set(tuple(run['keys']) for _, _, run in tested)) > 1:
		print('warning: the runs have different test sets, so', out, 'compares runtimes over different instances')
	bars = {}
	emphasis = {}
	portfolios = []
	solvers = {}
	for name, color, run in tested:
		result = run['result']
		bars[name], emphasis[name] = result['Model'][2], color
		portfolio = sorted(run['mapping'])
		if portfolio not in portfolios:
			portfolios.append(portfolio)
			bars[name + ' oracle'], emphasis[name + ' oracle'] = result['Oracle'][2], TEXT_2
		for s in run['mapping']:
			solvers.setdefault(s, result[s][2])
	for s, minutes in solvers.items():
		bars.setdefault(s, minutes)

	names = sorted(bars, key=bars.get, reverse=True)
	fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.5))
	ax.barh(names, [bars[n] for n in names], 0.7, color=[emphasis.get(n, NEUTRAL) for n in names])
	for i, n in enumerate(names):
		ax.annotate('{:.1f}'.format(bars[n]), (bars[n], i), xytext=(4, 0), textcoords='offset points',
					va='center', color=TEXT if n in emphasis else TEXT_2)
	for label in ax.get_yticklabels():
		if label.get_text() in emphasis:
			label.set_fontweight('bold')
			label.set_color(TEXT)
	ax.grid(axis='y', visible=False)
	ax.set_xlim(right=max(bars.values()) * 1.12)
	ax.set_xlabel('Total runtime on the test set (minutes; an unsolved instance counts {:g} s)'.format(limit))
	ax.set_title('Total runtime: models against single solvers and the oracles')
	save(fig, out)

def plot_relative_runtime(tested, out):
	'''
	Total runtime of each model (in its color) on its own test set, divided by that of its portfolio's oracle (left;
	never below 1) and of always using its portfolio's fastest single solver (right; below 1 beats it), for `tested` a
	list of (name, color, run), the model closest to the oracle on top.
	'''
	rows = []
	for name, color, run in tested:
		result = run['result']
		best = min(run['mapping'], key=lambda s: result[s][2])
		rows.append((name, color, result['Model'][2] / result['Oracle'][2], result['Model'][2] / result[best][2], best))
	rows.sort(key=lambda r: r[2], reverse=True)
	bests = sorted(set(r[4] for r in rows))

	values = [v for r in rows for v in r[2:4]]
	low, high = min(1, min(values)), max(1, max(values))
	span = max(high - low, 0.05)
	fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 0.45 * len(rows) + 1.8))
	titles = ('Relative to its oracle', 'Relative to always its fastest single solver ({})'.format(', '.join(bests)))
	for ax, index, title in zip(axes, (2, 3), titles):
		ax.axvline(1, color=TEXT_2, linestyle='--', linewidth=1)
		for i, r in enumerate(rows):
			ax.plot(r[index], i, 'o', color=r[1], markersize=10, markeredgecolor=SURFACE, markeredgewidth=2)
			label = '{:.2f}×'.format(r[index]) + (' {}'.format(r[4]) if index == 3 and len(bests) > 1 else '')
			ax.annotate(label, (r[index], i), xytext=(9, 0), textcoords='offset points', va='center', color=TEXT,
						bbox=dict(boxstyle='square,pad=0.1', facecolor=SURFACE, edgecolor='none'))
		ax.set_xlabel('Model\'s total runtime ÷ reference (lower is better)')
		ax.set_title(title)
	axes[0].set_yticks(range(len(rows)), [r[0] for r in rows], fontweight='bold', color=TEXT)
	axes[0].set_ylim(-0.6, len(rows) - 0.4)
	axes[0].set_xlim(low - 0.08 * span, high + 0.3 * span)
	save(fig, out)

def plot_accuracy(tested, out):
	'''
	Scatter of the accuracy (picks the fastest solver) against the coverage (picks a solver that solves the instance)
	of each model (in its color) and of always using each solver (gray rings), for `tested` a list of (name, color, run):
	every selector on the left, and on the right a zoom on the models and the solvers that cover about as much. A
	model is scored on its own portfolio and a solver on the largest portfolio that has it, so a solver's accuracy
	drops as its portfolio grows (the oracles, 100% on both, are left out).
	'''
	solvers = {}
	for _, _, run in sorted(tested, key=lambda t: -len(t[2]['mapping'])):
		for s in run['mapping']:
			solvers.setdefault(s, run['result'][s][1::-1])
	models = [(name, color, run['result']['Model'][1::-1]) for name, color, run in tested]
	near = [p for _, _, p in models]
	near += [p for p in solvers.values() if p[0] >= min(x for x, _ in near) - 0.01]
	(x0, y0), (x1, y1) = np.min(near, axis=0), np.max(near, axis=0)
	pad_x, pad_y = max(0.15 * (x1 - x0), 0.002), max(0.15 * (y1 - y0), 0.01)
	box = (x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y)

	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
	for ax in axes:
		for name, color, (x, y) in models:
			ax.plot(x, y, 'o', color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
		# rings on top, so that a model that picks like a single solver shows inside its ring
		for x, y in solvers.values():
			ax.plot(x, y, 'o', markersize=13, markerfacecolor='none', markeredgecolor=NEUTRAL, markeredgewidth=2)
		for axis in (ax.xaxis, ax.yaxis):
			# no tick above 100%, which the zoom's margin can reach
			axis.set_major_formatter(lambda v, _: '' if v > 1.0001 else '{:.4g}%'.format(100 * v))
		ax.set_xlabel('Coverage (picks a solver that solves the instance)')
	axes[0].set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), ylabel='Accuracy (picks the fastest solver)')
	axes[0].add_patch(matplotlib.patches.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False,
												   edgecolor=TEXT_2, linestyle='--', linewidth=1))
	axes[1].set(xlim=box[:2], ylim=box[2:])
	axes[0].set_title('Accuracy against coverage: models and single solvers')
	axes[1].set_title('Zoom on the models (dashed box)')
	fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), handles=axes[1].get_legend_handles_labels()[0])
	fig.tight_layout(rect=(0, 0.08, 1, 1))
	inside = lambda x, y: box[0] <= x <= box[1] and box[2] <= y <= box[3]
	label_points(axes[0], [(x, y, s, TEXT_2) for s, (x, y) in solvers.items() if not inside(x, y)])
	label_points(axes[1], [(x, y, s, TEXT_2) for s, (x, y) in solvers.items() if inside(x, y)]
				 + [(x, y, name, TEXT) for name, _, (x, y) in models])
	fig.savefig(out, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print('wrote', out)

def label_points(ax, points):
	'''
	Labels each (x, y, text, color) of `points` beside its point (to the left near the right edge), moved up or down
	as little as keeps it inside the axes and off the labels placed before it, with a line to its point if moved
	'''
	renderer = ax.figure.canvas.get_renderer()
	placed = []
	for x, y, text, color in sorted(points, key=lambda p: -p[1]):
		left = ax.transLimits.transform((x, y))[0] > 0.85
		offset = -10 if left else 10
		style = dict(textcoords='offset points', va='center', ha='right' if left else 'left', color=color, fontsize=9)
		label = ax.annotate(text, (x, y), xytext=(offset, 0), **style)
		extent = label.get_window_extent(renderer)
		fits = lambda box: ax.bbox.y0 <= box.y0 and box.y1 <= ax.bbox.y1 and not any(box.overlaps(o) for o in placed)
		shift = next((d for d in sorted(range(-400, 401, 2), key=abs) if fits(extent.translated(0, d))), 0)
		if shift:
			label.remove()
			ax.annotate(text, (x, y), xytext=(offset, shift * 72 / ax.figure.dpi), **style,
						arrowprops=dict(arrowstyle='-', color=NEUTRAL, linewidth=0.8, shrinkA=1, shrinkB=5))
		placed.append(extent.translated(0, shift))

def save(fig, out):
	fig.tight_layout()
	fig.savefig(out, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print('wrote', out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='datasets/mapfast_paper/four-solvers', help='Give the dataset folder, or the location of its config.json file')
	parser.add_argument('--log', default=None, help='Slurm log of the training run (logs/<name>_<jobid>.out), for the loss graph')
	parser.add_argument('--run', nargs=3, action='append', metavar=('NAME', 'DATASET', 'LOG'),
						help='Compare several models on the same graphs instead: a name, its dataset folder and its training log. Repeat for each model.')
	parser.add_argument('--timeout', type=float, default=None,
						help='With --run: time limit in seconds to evaluate every model under (default: the smallest timeout of the runs\' configs)')
	parser.add_argument('-o', '--output', default=None, help='Folder to write the graphs to (default: logs/<name>_<jobid>_plots for --log logs/<name>_<jobid>.out, else logs/<dataset>_plots; logs/<name>_vs_<name>_plots with --run)')
	args = parser.parse_args()

	if args.run:
		output = args.output or os.path.join('logs', '_vs_'.join(name for name, _, _ in args.run) + '_plots')
		os.makedirs(output, exist_ok=True)
		colors = [SERIES_1, SERIES_2, SERIES_3, SERIES_4, SERIES_5, SERIES_6, SERIES_7, SERIES_8]
		if len(args.run) > len(colors):
			raise SystemExit('at most {} runs can be compared'.format(len(colors)))
		runs = []
		for (name, dataset, log_file), color in zip(args.run, colors):
			log = read_training_log(log_file) if os.path.exists(log_file) else None
			runs.append((name, color, load_run(dataset), log))
		# evaluate every model under the same time limit: the strictest of their benchmarks
		tested = [r[2] for r in runs if r[2]]
		limit = args.timeout or (min(r['timeout'] for r in tested) if tested else None)
		for i, (name, color, run, log) in enumerate(runs):
			if run:
				run = apply_time_limit(run, limit)
				runs[i] = (name, color, run, log)
				print('{}: {} test instances under a {:g} s time limit ({} left out: no solver solves them within it)'.format(
					name, len(run['keys']), limit, run['dropped']))
				print_scores({'Model': run['result']['Model']}, limit)
		plot_comparison(runs, output, limit)
		sys.exit(0)

	if args.output:
		output = args.output
	elif args.log:
		output = os.path.splitext(args.log)[0] + '_plots'
	else:
		dataset_dir = args.config if os.path.isdir(args.config) else os.path.dirname(args.config)
		output = os.path.join('logs', os.path.basename(os.path.normpath(dataset_dir)) + '_plots')
	os.makedirs(output, exist_ok=True)

	if args.log:
		log = read_training_log(args.log)
		if log['steps']:
			plot_loss(log, os.path.join(output, 'loss.png'))
		else:
			print('no training progress lines in', args.log)

	run = load_run(args.config)
	if run is None:
		sys.exit(0)
	print_scores(run['result'], run['timeout'])
	plot_runtime(run['result'], run['timeout'], os.path.join(output, 'runtime.png'))
	plot_confusion(run['keys'], run['picks'], run['mapping'], os.path.join(output, 'confusion.png'))
