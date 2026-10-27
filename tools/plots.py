'''
Score table and loss/runtime/confusion graphs of one model (-C DATASET --log LOG), or a comparison of several
(--run NAME DATASET LOG ...) under a common time limit.
'''
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from movingai import MAP_FAMILIES, agent_count, map_family

# dataviz palette
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
	main.py's progress lines, or a MAPFAST-GNN configuration's results.json:
	{'steps': [(step, running mean, batch loss)], 'epochs': [(step, mean loss)]}.
	'''
	if log_file.endswith('.json'):
		with open(log_file) as f:
			return {'steps': [], 'epochs': [(h['epoch'], h['train_loss']) for h in json.load(f)['history']]}
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
	'''Returns (test keys, {selector: picks}) for every single solver, 'Model' and 'Oracle'.'''
	inv_mapping = {v: k for k, v in mapping.items()}
	keys = [k[:k.rindex('_')] for k in predictions]
	picks = {s: [s] * len(keys) for s in mapping}
	picks['Model'] = [inv_mapping[p['best'][0]] for p in predictions.values()]
	picks['Oracle'] = [yaml_details[k]['SOLVER'] for k in keys]
	return keys, picks

def scores(yaml_details, keys, picks, timeout):
	'''{selector: (accuracy, coverage, total minutes)}; unsolved costs `timeout`.'''
	result = {}
	for name, chosen in picks.items():
		fastest = [yaml_details[k]['SOLVER'] == s for k, s in zip(keys, chosen)]
		times = [yaml_details[k][s] for k, s in zip(keys, chosen)]
		result[name] = (np.mean(fastest), np.mean([t != -1 for t in times]),
						sum(timeout if t == -1 else t for t in times) / 60)
	return result

def print_scores(result, timeout):
	print('{:<24} {:>9} {:>9} {:>14}'.format('', 'accuracy', 'coverage', 'runtime (min)'))
	for name in sorted(result, key=lambda n: result[n][2]):
		print('{:<24} {:>9.1%} {:>9.1%} {:>14.1f}'.format(name, *result[name]))
	print('(an unsolved instance counts {:g} s)'.format(timeout))

def plot_bars(bars, emphasis, timeout, title, out):
	'''Total runtime bars, slowest on top; `emphasis` maps the highlighted bars to their colors.'''
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
	ax.set_xlabel('Total runtime on the test set (minutes; an unsolved instance counts {:g} s)'.format(timeout))
	ax.set_title(title)
	save(fig, out)

def plot_runtime(result, timeout, out):
	plot_bars({n: r[2] for n, r in result.items()}, {'Model': SERIES_1, 'Oracle': TEXT_2}, timeout,
			  'Total runtime: model against single solvers and the oracle', out)

# shades of correct (blue) and wrong (orange) picks, by share of the row
CORRECT = matplotlib.colors.LinearSegmentedColormap.from_list('correct', SEQUENTIAL)
WRONG = matplotlib.colors.LinearSegmentedColormap.from_list('wrong', [SURFACE, '#f9d3c2', SERIES_2, '#a8441a'])

def plot_confusion(keys, picks, mapping, out, title='Predicted against actual fastest solver'):
	solvers = list(mapping)
	index = {s: i for i, s in enumerate(solvers)}
	counts = np.zeros((len(solvers), len(solvers)), dtype=int)
	for actual, predicted in zip(picks['Oracle'], picks['Model']):
		counts[index[actual], index[predicted]] += 1
	shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
	correct = np.eye(len(solvers), dtype=bool)

	size = 0.6 * len(solvers) + 2.5
	fig, ax = plt.subplots(figsize=(size + 1, size + 0.4))
	# any nonzero share stays visibly tinted
	shade = 0.15 + 0.85 * shares
	colors = np.where(correct[..., None], CORRECT(shade), WRONG(shade))
	colors[counts == 0] = matplotlib.colors.to_rgba(SURFACE)
	ax.imshow(colors)
	for i in range(len(solvers)):
		for j in range(len(solvers)):
			if counts[i, j]:
				ax.text(j, i, counts[i, j], ha='center', va='center', fontsize=9,
						color=SURFACE if shares[i, j] > 0.55 else TEXT, fontweight='bold' if i == j else 'normal')
		# outline the correct cell, so correctness does not rest on color alone
		if counts[i].any():
			ax.add_patch(matplotlib.patches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=1.5))
	ax.set_xticks(range(len(solvers)))
	ax.set_xticklabels(solvers, rotation=45, ha='right')
	ax.set_yticks(range(len(solvers)))
	ax.set_yticklabels(['{} ({}, {:.0%} right)'.format(s, n, c / n) if n else '{} (0)'.format(s)
						for s, n, c in zip(solvers, counts.sum(axis=1), counts.diagonal())])
	ax.set_xlabel('Model\'s pick')
	ax.set_ylabel('Actual fastest solver (test instances)')
	ax.grid(False)
	ax.spines[:].set_visible(False)
	ax.legend(handles=[matplotlib.patches.Patch(facecolor=CORRECT(0.75), edgecolor=TEXT, linewidth=1.5, label='Correct pick'),
					   matplotlib.patches.Patch(facecolor=WRONG(0.75), label='Wrong pick')],
			  title='Darker: larger\nshare of the row', title_fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1),
			  alignment='left')
	ax.set_title('{} ({:.1%} right overall)'.format(title, counts.trace() / max(counts.sum(), 1)))
	save(fig, out)

def load_run(config_path):
	'''A dataset's predictions and scores, or None before main.py -T 0.'''
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
	'''Rescores a run under a shorter time limit, dropping instances no solver solves within it.'''
	yaml_details = {}
	for key, record in run['yaml_details'].items():
		yaml_details[key] = {s: (-1 if s != 'SOLVER' and t > limit else t) for s, t in record.items()}
	keep = [i for i, k in enumerate(run['keys']) if yaml_details[k][yaml_details[k]['SOLVER']] != -1]
	keys = [run['keys'][i] for i in keep]
	picks = {name: [chosen[i] for i in keep] for name, chosen in run['picks'].items()}
	return {**run, 'yaml_details': yaml_details, 'keys': keys, 'picks': picks, 'timeout': limit,
			'result': scores(yaml_details, keys, picks, limit), 'dropped': len(run['keys']) - len(keys)}

def plot_comparison(runs, output, limit=None):
	'''compare_loss.png, compare_total_runtime.png, compare_solved.png and confusion_<name>.png for `runs`: (name, color, run or None, log or None).'''
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
		plot_solved(tested, limit, os.path.join(output, 'compare_solved.png'))
	for name, _, run in tested:
		plot_confusion(run['keys'], run['picks'], run['mapping'], os.path.join(output, 'confusion_{}.png'.format(name)),
					   name + ': predicted against actual fastest solver')

def plot_total_runtime(tested, limit, out):
	'''Total runtime of each model, each portfolio's oracle and each single solver (from the first run that has it).'''
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
			oracle = 'Oracle ({} solvers)'.format(len(portfolio))
			bars[oracle], emphasis[oracle] = result['Oracle'][2], TEXT_2
		for s in run['mapping']:
			solvers.setdefault(s, result[s][2])
	for s, minutes in solvers.items():
		bars.setdefault(s, minutes)

	plot_bars(bars, emphasis, limit, 'Total runtime: models against single solvers and the oracles', out)

def plot_solved(tested, limit, out, bins=7):
	'''Share of the test instances each model, bcp2 and lazycbs solves against agent count (in bins of equal size), by map family.'''
	lines = [(name, color, '-', run, run['picks']['Model']) for name, color, run in tested]
	for s, color in (('bcp2', TEXT), ('lazycbs', NEUTRAL)):
		run = next((run for _, _, run in tested if s in run['mapping']), None)
		if run:
			lines.append((s, color, '--', run, run['picks'][s]))
	agents = np.array([agent_count(k) for k in tested[0][2]['keys']])
	counts, sizes = np.unique(agents, return_counts=True)
	group = (sizes.cumsum() - sizes) * bins // sizes.sum()
	ranges = [(counts[group == g].min(), counts[group == g].max()) for g in np.unique(group)]
	families = ['all maps'] + [f for f in MAP_FAMILIES if f in {map_family(k) for k in tested[0][2]['keys']}]

	columns = min(4, len(families))
	rows = -(-len(families) // columns)
	fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.2 * rows), sharey=True, squeeze=False)
	for name, color, style, run, picks in lines:
		family = np.array([map_family(k) for k in run['keys']])
		bin_of = np.searchsorted([hi for _, hi in ranges], [agent_count(k) for k in run['keys']])
		solved = np.array([run['yaml_details'][k][s] != -1 for k, s in zip(run['keys'], picks)])
		for ax, f in zip(axes.flat, families):
			mask = (family == f) | (f == 'all maps')
			share = [solved[mask & (bin_of == b)].mean() * 100 if (mask & (bin_of == b)).any() else np.nan
					 for b in range(len(ranges))]
			ax.plot(share, style, marker='o', markersize=4, color=color, linewidth=2 if style == '-' else 1.5, label=name)
	family = np.array([map_family(k) for k in tested[0][2]['keys']])
	for ax, f in zip(axes.flat, families):
		ax.set_title('{} ({} instances)'.format(f, len(family) if f == 'all maps' else (family == f).sum()), fontsize=11)
		ax.set_xticks(range(len(ranges)))
		ax.set_xticklabels(['{}–{}'.format(lo, hi) if lo < hi else str(lo) for lo, hi in ranges], rotation=45)
		ax.set_xlabel('Agents')
		ax.set_ylim(top=100.5)
		ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
	for ax in axes[:, 0]:
		ax.set_ylabel('Test instances solved within {:g} s'.format(limit))
	for ax in axes.flat[len(families):]:
		ax.set_visible(False)
	fig.tight_layout()
	fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', ncol=min(len(lines), 5),
			   bbox_to_anchor=(0.5, 0))
	save(fig, out)

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
						help='Compare several models on the same graphs instead: a name, its dataset folder and its training log (from main.py, or a GNN configuration\'s results.json; none to skip). Repeat for each model.')
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
		# the strictest time limit of the runs
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
