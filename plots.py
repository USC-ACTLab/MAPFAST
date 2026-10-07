'''
Graphs of a MAPFAST training run and of its test predictions:

	loss.png				training loss per step and per epoch (and validation loss, if logged)
	accuracy_coverage.png	accuracy and coverage of the model and of every solver
	runtime.png			total runtime of the model, of every solver and of the oracle
	confusion.png			predicted against actual fastest solver
	accuracy_by_agents.png	accuracy of the model and of the best solver against the number of agents

	python plots.py -C datasets/nine_solvers --log logs/1234.out

Each graph is written to <dataset>/plots/. Without --log, loss.png is skipped; without the
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

def plot_accuracy_by_agents(yaml_details, map_details, keys, picks, mapping, out):
	'''
	Line chart of the accuracy of the model and of the single solver with the best overall accuracy,
	against the number of agents of the test instances.
	'''
	agents = np.array([map_details[k]['no_agents'] for k in keys])
	best = max(mapping, key=lambda s: sum(yaml_details[k]['SOLVER'] == s for k in keys))
	counts = sorted(set(agents))
	fig, ax = plt.subplots(figsize=(8, 4.5))
	for name, color in (('Model', SERIES_1), (best, SERIES_2)):
		correct = np.array([yaml_details[k]['SOLVER'] == s for k, s in zip(keys, picks[name])])
		accuracy = [correct[agents == n].mean() for n in counts]
		label = 'Model' if name == 'Model' else 'Always {} (best single solver)'.format(name)
		ax.plot(counts, accuracy, 'o-', color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
	ax.set_xscale('log')
	ax.set_xticks(counts)
	ax.set_xticklabels(['{}\n(n={})'.format(n, (agents == n).sum()) for n in counts])
	ax.minorticks_off()
	ax.set_ylim(0, 1.05)
	ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
	ax.set_xlabel('Agents (test instances)')
	ax.set_ylabel('Accuracy (picks the fastest)')
	ax.set_title('Accuracy by number of agents')
	ax.legend(loc='lower left')
	save(fig, out)

def save(fig, out):
	fig.tight_layout()
	fig.savefig(out, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print('wrote', out)

if __name__ == '__main__':
	from utils import read_config

	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='datasets/three_solvers', help='Give the dataset folder, or the location of its config.json file')
	parser.add_argument('--log', default=None, help='Slurm log of the training run (logs/<jobid>.out), for the loss graph')
	parser.add_argument('-o', '--output', default=None, help='Folder to write the graphs to (default: <dataset>/plots)')
	args = parser.parse_args()

	config = read_config(args.config)['Analysis']
	dataset_dir = args.config if os.path.isdir(args.config) else os.path.dirname(args.config)
	output = args.output or os.path.join(dataset_dir, 'plots')
	os.makedirs(output, exist_ok=True)

	if args.log:
		log = read_training_log(args.log)
		if log['steps']:
			plot_loss(log, os.path.join(output, 'loss.png'))
		else:
			print('no training progress lines in', args.log)

	if not os.path.exists(config['prediction_output']):
		print('no predictions at', config['prediction_output'], '- run main.py -T 0 first')
		sys.exit(0)
	with open(config['yaml_details']) as f:
		yaml_details = json.load(f)
	with open(config['map_details']) as f:
		map_details = json.load(f)
	with open(config['prediction_output']) as f:
		predictions = json.load(f)
	mapping = config['mapping']
	timeout = config.get('timeout', 300)

	keys, picks = selections(yaml_details, predictions, mapping)
	result = scores(yaml_details, keys, picks, timeout)
	plot_accuracy_coverage(result, os.path.join(output, 'accuracy_coverage.png'))
	plot_runtime(result, timeout, os.path.join(output, 'runtime.png'))
	plot_confusion(keys, picks, mapping, os.path.join(output, 'confusion.png'))
	plot_accuracy_by_agents(yaml_details, map_details, keys, picks, mapping, os.path.join(output, 'accuracy_by_agents.png'))
