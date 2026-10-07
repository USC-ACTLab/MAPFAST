'''
Graphs of a MAPFAST training run and of its test predictions:

	loss.png				training loss per step and per epoch (and validation loss, if logged)
	accuracy_coverage.png	accuracy and coverage of the model and of every solver
	runtime.png			total runtime of the model, of every solver and of the oracle
	confusion.png			predicted against actual fastest solver
	accuracy_by_agents.png	accuracy of the model and of the best solver against the number of agents

	python plots.py -C datasets/nine_solvers --log logs/MAPFASTv2_1234.out

With --run, several trained models are compared on the same graphs instead (see plot_comparison), all
evaluated under the same time limit (see apply_time_limit):

	python plots.py --run MAPFAST datasets/three_solvers logs/MAPFAST_1234.out --run MAPFASTv2 datasets/nine_solvers logs/MAPFASTv2_1235.out -o logs/MAPFAST_vs_MAPFASTv2_1236

The graphs are written next to the log, to logs/<name>_<jobid>_plots/ for --log logs/<name>_<jobid>.out
(logs/<dataset>_plots/ without --log). Without --log, loss.png is skipped; without the
prediction_output of the config (written by main.py -T 0), only loss.png is drawn.
The accuracy, coverage and runtime are computed as in analysis.py.
'''
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Reference palette of the dataviz method: two categorical slots, neutral ink and surfaces
SERIES_1 = '#2a78d6'
SERIES_2 = '#eb6834'
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
VALID_LINE = re.compile(r'Num Batches (\d+) / (\d+) \|.*\| Valid_Losses \[([^\]]*)\]')

def read_training_log(log_file):
	'''
	Reads the progress lines that main.py prints while training.

	Returns: Json object with
		'steps' -> list of (global step, running mean loss of the epoch, batch loss)
		'epochs' -> list of (global step at the end of the epoch, mean loss of the epoch)
		'valid' -> list of (global step, validation loss)
	'''
	log = {'steps': [], 'epochs': [], 'valid': []}
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
				continue
			m = VALID_LINE.search(line)
			if m and log['steps']:
				losses = [float(x) for x in m.group(3).split(',') if x.strip()]
				if losses:
					log['valid'].append((log['steps'][-1][0], sum(losses)))
	return log

def plot_loss(log, out):
	'''
	Line chart of the training loss: the running mean of each epoch, the mean of each finished epoch
	and, if main.py logged any, the validation loss.
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
	if log['valid']:
		valid = np.array(log['valid'])
		ax.plot(valid[:, 0], valid[:, 1], 'o-', color=SERIES_2, markersize=8, markeredgecolor=SURFACE,
				markeredgewidth=2, label='Validation')
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

def plot_accuracy_coverage(result, out):
	'''
	Grouped horizontal bars of the accuracy (picks the fastest solver) and coverage (picks a solver that
	solves the instance) of the model and of every solver, sorted by accuracy.
	'''
	names = sorted((n for n in result if n != 'Oracle'), key=lambda n: result[n][0])
	y = np.arange(len(names))
	h = 0.38
	fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
	ax.barh(y + h / 2, [result[n][0] for n in names], h - 0.04, color=SERIES_1, label='Accuracy (picks the fastest)')
	ax.barh(y - h / 2, [result[n][1] for n in names], h - 0.04, color=SERIES_2, label='Coverage (picks a solver that solves it)')
	model = names.index('Model')
	for dy, value in ((h / 2, result['Model'][0]), (-h / 2, result['Model'][1])):
		ax.annotate('{:.0%}'.format(value), (value, model + dy), xytext=(4, 0), textcoords='offset points',
					va='center', color=TEXT)
	ax.set_yticks(y)
	ax.set_yticklabels(names)
	for label in ax.get_yticklabels():
		if label.get_text() == 'Model':
			label.set_fontweight('bold')
			label.set_color(TEXT)
	ax.set_xlim(0, 1.08)
	ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
	ax.grid(axis='y', visible=False)
	ax.set_title('Accuracy and coverage on the test set')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08 - 0.3 / len(names)), ncol=2)
	save(fig, out)

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

# agent-count ranges, so that datasets with different agent counts share one x axis
AGENT_BINS = [(1, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, None)]

def agent_bin_label(low, high):
	return '>{}'.format(low - 1) if high is None else '\u2264{}'.format(high) if low == 1 else '{}\u2013{}'.format(low, high)

def accuracy_by_agents(yaml_details, map_details, keys, chosen):
	'''
	Returns: List of (bin index, accuracy, number of instances) for every agent-count range of AGENT_BINS
	with test instances, where `chosen` is the solver picked for each of `keys`
	'''
	agents = np.array([map_details[k]['no_agents'] for k in keys])
	correct = np.array([yaml_details[k]['SOLVER'] == s for k, s in zip(keys, chosen)])
	rows = []
	for i, (low, high) in enumerate(AGENT_BINS):
		inside = (agents >= low) & (agents <= (high if high is not None else agents.max()))
		if inside.any():
			rows.append((i, correct[inside].mean(), int(inside.sum())))
	return rows

def plot_accuracy_lines(lines, title, out):
	'''
	Line chart of accuracy against the agent-count ranges of AGENT_BINS, one line per (label, color, rows)
	of `lines`, with rows as returned by accuracy_by_agents.
	'''
	fig, ax = plt.subplots(figsize=(8, 4.5))
	used = sorted(set(r[0] for _, _, rows in lines for r in rows))
	for label, color, rows in lines:
		ax.plot([r[0] for r in rows], [r[1] for r in rows], 'o-', color=color, markersize=8,
				markeredgecolor=SURFACE, markeredgewidth=2, label=label)
	ax.set_xticks(used)
	ax.set_xticklabels([agent_bin_label(*AGENT_BINS[i]) for i in used])
	ax.set_ylim(0, 1.05)
	ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
	ax.set_xlabel('Agents')
	ax.set_ylabel('Accuracy (picks the fastest)')
	ax.set_title(title)
	ax.legend(loc='lower left')
	save(fig, out)

def plot_accuracy_by_agents(yaml_details, map_details, keys, picks, mapping, out):
	'''
	Line chart of the accuracy of the model and of the single solver with the best overall accuracy,
	against the number of agents of the test instances.
	'''
	best = max(mapping, key=lambda s: sum(yaml_details[k]['SOLVER'] == s for k in keys))
	plot_accuracy_lines([
		('Model', SERIES_1, accuracy_by_agents(yaml_details, map_details, keys, picks['Model'])),
		('Always {} (best single solver)'.format(best), SERIES_2, accuracy_by_agents(yaml_details, map_details, keys, picks[best])),
	], 'Accuracy by number of agents', out)

def load_run(config_path):
	'''
	Loads the dataset and test predictions of a dataset folder (or its config.json).

	Returns: Json object with yaml_details, map_details, mapping, timeout, keys, picks and result
	(see selections and scores), or None if main.py -T 0 has not written the predictions yet
	'''
	from utils import read_config
	config = read_config(config_path)['Analysis']
	if not os.path.exists(config['prediction_output']):
		print('no predictions at', config['prediction_output'], '- run main.py -T 0 first')
		return None
	run = {'mapping': config['mapping'], 'timeout': config.get('timeout', 300)}
	for name in ('yaml_details', 'map_details', 'prediction_output'):
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
		compare_accuracy.png		accuracy and coverage of each model on its own test set
		compare_runtime.png		total runtime of each model relative to its oracle and to its best single solver
		compare_accuracy_by_agents.png	accuracy of each model against the number of agents

	The models may have different solver portfolios and test sets, so runtimes are compared as ratios.
	`limit` is the common time limit the runs were evaluated under (see apply_time_limit), shown in the titles.
	'''
	under = '' if limit is None else ' ({:g} s time limit for both)'.format(limit)
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
	if not tested:
		return

	metrics = ['Accuracy\n(picks the fastest)', 'Coverage\n(picks a solver that solves it)']
	fig, ax = plt.subplots(figsize=(8, 4.5))
	width = 0.8 / len(tested)
	for i, (name, color, run) in enumerate(tested):
		values = run['result']['Model'][:2]
		x = np.arange(len(metrics)) + (i - (len(tested) - 1) / 2) * width
		ax.bar(x, values, width - 0.04, color=color, label=name)
		for xi, v in zip(x, values):
			ax.annotate('{:.0%}'.format(v), (xi, v), xytext=(0, 4), textcoords='offset points', ha='center', color=TEXT)
	ax.set_xticks(range(len(metrics)))
	ax.set_xticklabels(metrics)
	ax.set_ylim(0, 1.1)
	ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
	ax.grid(axis='x', visible=False)
	ax.set_title('Accuracy and coverage on each model\'s test set' + under)
	ax.legend(loc='upper left')
	save(fig, os.path.join(output, 'compare_accuracy.png'))

	metrics = ['Relative to the oracle\n(always the fastest solver)', 'Relative to the best single solver']
	fig, ax = plt.subplots(figsize=(8, 4.5))
	for i, (name, color, run) in enumerate(tested):
		result = run['result']
		model = result['Model'][2]
		best = min(run['mapping'], key=lambda s: result[s][2])
		values = [model / result['Oracle'][2], model / result[best][2]]
		x = np.arange(len(metrics)) + (i - (len(tested) - 1) / 2) * width
		ax.bar(x, values, width - 0.04, color=color, label='{} (best single solver: {})'.format(name, best))
		for xi, v in zip(x, values):
			ax.annotate('{:.2f}\u00d7'.format(v), (xi, v), xytext=(0, 4), textcoords='offset points', ha='center', color=TEXT)
	ax.axhline(1, color=TEXT_2, linewidth=1, linestyle='--')
	ax.set_xticks(range(len(metrics)))
	ax.set_xticklabels(metrics)
	ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
	ax.grid(axis='x', visible=False)
	ax.set_ylabel('Model\'s total runtime \u00f7 reference')
	ax.set_title('Total runtime on each model\'s test set, lower is better' + under)
	ax.legend(loc='upper right')
	save(fig, os.path.join(output, 'compare_runtime.png'))

	plot_accuracy_lines([(name, color, accuracy_by_agents(run['yaml_details'], run['map_details'], run['keys'], run['picks']['Model']))
						 for name, color, run in tested], 'Accuracy by number of agents' + under, os.path.join(output, 'compare_accuracy_by_agents.png'))

def save(fig, out):
	fig.tight_layout()
	fig.savefig(out, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print('wrote', out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='datasets/three_solvers', help='Give the dataset folder, or the location of its config.json file')
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
		colors = [SERIES_1, SERIES_2]
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
	keys, picks, result, mapping, timeout = run['keys'], run['picks'], run['result'], run['mapping'], run['timeout']
	plot_accuracy_coverage(result, os.path.join(output, 'accuracy_coverage.png'))
	plot_runtime(result, timeout, os.path.join(output, 'runtime.png'))
	plot_confusion(keys, picks, mapping, os.path.join(output, 'confusion.png'))
	plot_accuracy_by_agents(run['yaml_details'], run['map_details'], keys, picks, mapping, os.path.join(output, 'accuracy_by_agents.png'))
