'''
Train (`train DATASET --config-index I --out DIR`) and select (`select DIR --dataset-out DATASET2`) MAPFAST-GNN over GRID.
Fallback: keep the default solver unless the top output beats it by a threshold; epoch, threshold and configuration
are chosen by validation speedup. --folds K: choose on pooled out-of-fold outputs, test picks average the K networks.
'''
import argparse
import glob
import itertools
import json
import os
import pickle
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from gnn import SolverGNN, collate, graph_size
from utils import read_config

# edges: 'ends' links agents to their start and goal, 'paths' also to their path cells
GRID = {
	'kind': ['sage'],
	'hidden': [128],
	'layers': [4],
	'lr': [1e-3],
	'edges': ['ends', 'paths'],
	'loss': ['ce', 'runtime'],
}
THRESHOLDS = (0., 0.25, 0.5, 1., 2.)

def configurations():
	return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

def config_name(config):
	return '{kind}_h{hidden}_l{layers}_lr{lr:g}_{edges}_{loss}'.format(**config)

def split(dataset):
	'''main.py's split. Returns (train, valid, test, config, yaml_details).'''
	from MAPFAST import MAPFAST
	config = read_config(dataset)
	with open(config['yaml_details']) as f:
		yaml_details = json.load(f)
	random.seed(42)
	train, test, valid = MAPFAST('cpu', yaml_details, None, config['mapping']).get_train_valid_test_list()
	strip = lambda names: [n[:n.rindex('_')] for n in names]
	return strip(train), strip(valid), strip(test), config, yaml_details

def solver_times(yaml_details, names, solvers, timeout):
	times = np.array([[yaml_details[n][s] for s in solvers] for n in names], dtype=np.float64)
	return np.where(times == -1, timeout, times)

def apply_threshold(outputs, default, threshold):
	return np.where(outputs.max(1) - outputs[:, default] >= threshold, outputs.argmax(1), default)

def batches(names, data, batch_size, max_nodes):
	runs, nodes = [[]], 0
	for name in names:
		size = data['size'](name)
		if runs[-1] and (len(runs[-1]) == batch_size or nodes + size > max_nodes):
			runs.append([])
			nodes = 0
		runs[-1].append(name)
		nodes += size
	return runs

def predict(net, names, data, args):
	'''Returns (outputs, seconds per instance).'''
	net.eval()
	outputs = []
	start = time.perf_counter()
	with torch.no_grad():
		for run in batches(names, data, args.batch_size, args.max_nodes):
			outputs.append(net(*data['collate'](run), len(run)).cpu().numpy())
	return np.concatenate(outputs), (time.perf_counter() - start) / max(len(names), 1)

def score(picks, names, data):
	times = data['times'](names)
	runtime = times[np.arange(len(names)), picks].sum() / 60
	return {'runtime': float(runtime), 'speedup': float(times[:, data['default']].sum() / 60 / runtime),
			'accuracy': float((picks == data['labels'](names)).mean())}

def best_threshold(outputs, names, data):
	scored = [(t, score(apply_threshold(outputs, data['default'], t), names, data)) for t in data['thresholds']]
	return min(scored, key=lambda s: (s[1]['runtime'], -s[1]['accuracy']))

def folds(train_names, valid_names, k):
	'''(train, valid) per fold; k = 1 is main.py's split.'''
	if k == 1:
		return [(train_names, valid_names)]
	pool = sorted(train_names + valid_names)
	random.Random(42).shuffle(pool)
	return [([n for j, n in enumerate(pool) if j % k != i], pool[i::k]) for i in range(k)]

def fit(train_names, valid_names, test_names, data, graphs, config, args, device):
	'''Returns per-epoch valid/test outputs and loss, inference seconds and parameter count.'''
	timeout = data['timeout']
	globs = np.stack([graphs['instances'][n]['glob'] for n in train_names]).astype(np.float64)
	global_stats = (globs.mean(0).astype(np.float32), (globs.std(0) + 1e-6).astype(np.float32))
	data = {**data, 'collate': lambda names: collate(names, graphs, device, global_stats, config['edges'] == 'paths')}

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	net = SolverGNN(len(graphs['node_features']), len(graphs['global_features']), len(data['solvers']),
					config['kind'], config['hidden'], config['layers'], args.dropout).to(device)
	optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=args.weight_decay)
	order = list(train_names)
	random.shuffle(order)
	runs = batches(order, data, args.batch_size, args.max_nodes)
	schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['lr'], total_steps=args.epochs * len(runs))

	result = {'valid': [], 'test': [], 'loss': []}
	for epoch in range(args.epochs):
		net.train()
		random.shuffle(runs)
		total = 0.
		for names in runs:
			outputs = net(*data['collate'](names), len(names))
			if config['loss'] == 'runtime':
				times = torch.from_numpy(data['times'](names)).float().to(device)
				# outputs are -log(1 + predicted seconds), so argmax is the fastest
				error = -outputs - torch.log1p(times)
				# a timeout is only a lower bound
				error = torch.where(times >= timeout, error.clamp(max=0), error)
				loss = (error ** 2).mean()
			else:
				loss = F.cross_entropy(outputs, torch.from_numpy(data['labels'](names)).to(device))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			schedule.step()
			total += loss.item() * len(names)
		valid_outputs, _ = predict(net, valid_names, data, args)
		test_outputs, seconds = predict(net, test_names, data, args)
		result['valid'].append(valid_outputs)
		result['test'].append(test_outputs)
		result['loss'].append(total / len(order))
		threshold, valid = best_threshold(valid_outputs, valid_names, data)
		print('Epoch {} | loss {:.4f} | validation: speedup {:.4f} over always {} | accuracy {:.4f} | threshold {:g}'.format(
			epoch, total / len(order), valid['speedup'], data['solvers'][data['default']], valid['accuracy'], threshold), flush=True)
	return {**{k: np.array(v) for k, v in result.items()}, 'seconds': seconds,
			'parameters': sum(p.numel() for p in net.parameters())}

def train(args):
	config = configurations()[args.config_index]
	name = config_name(config)
	out = os.path.join(args.out, name)
	os.makedirs(out, exist_ok=True)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Configuration {}: {} on {}'.format(args.config_index, name, device), flush=True)

	train_names, valid_names, test_names, dataset_config, yaml_details = split(args.dataset)
	graph_file = args.graphs or os.path.join(args.dataset, 'graphs.pkl')
	with open(graph_file, 'rb') as f:
		graphs = pickle.load(f)
	missing = [n for n in train_names + valid_names + test_names if n not in graphs['instances']]
	if missing:
		raise SystemExit('{} instances have no graph in {}, e.g. {}'.format(len(missing), graph_file, missing[0]))
	if args.limit:
		train_names, valid_names, test_names = train_names[:args.limit], valid_names[:args.limit // 8], test_names[:args.limit // 8]
	mapping = dataset_config['mapping']
	solvers = sorted(mapping, key=mapping.get)
	timeout = dataset_config['timeout']

	time_rows = dict(zip(yaml_details, solver_times(yaml_details, list(yaml_details), solvers, timeout)))
	data = {
		'solvers': solvers,
		'timeout': timeout,
		'size': lambda name: graph_size(name, graphs),
		'times': lambda names: np.stack([time_rows[n] for n in names]),
		'labels': lambda names: np.array([mapping[yaml_details[n]['SOLVER']] for n in names]),
	}
	parts = folds(train_names, valid_names, args.folds)
	valid_names = [n for _, valid in parts for n in valid]
	# default: the single solver with the lowest training runtime
	data['default'] = int(data['times']([n for train, _ in parts for n in train]).sum(0).argmin())
	data['thresholds'] = (0.,) if args.no_fallback else THRESHOLDS
	baseline = {'solver': solvers[data['default']], **score(np.full(len(valid_names), data['default']), valid_names, data)}
	oracle = score(data['labels'](valid_names), valid_names, data)

	fits = []
	for i, (fold_train, fold_valid) in enumerate(parts):
		path = os.path.join(out, 'fold{}of{}.npz'.format(i + 1, len(parts)))
		if os.path.exists(path) and np.load(path)['loss'].shape == (args.epochs,):
			print('Fold {}/{}: done before ({})'.format(i + 1, len(parts), path), flush=True)
		else:
			print('Fold {}/{}: training on {}, validating on {} instances'.format(
				i + 1, len(parts), len(fold_train), len(fold_valid)), flush=True)
			# atomic, so a killed job leaves no truncated fold to resume from
			np.savez(path + '.tmp.npz', **fit(fold_train, fold_valid, test_names, data, graphs, config, args, device))
			os.replace(path + '.tmp.npz', path)
		fits.append(dict(np.load(path)))

	valid_outputs = np.concatenate([f['valid'] for f in fits], axis=1)
	history = []
	for epoch in range(args.epochs):
		threshold, valid = best_threshold(valid_outputs[epoch], valid_names, data)
		history.append({'epoch': epoch, 'train_loss': float(np.mean([f['loss'][epoch] for f in fits])),
						'threshold': threshold, **{'valid_' + k: v for k, v in valid.items()}})
	best = {k: v for k, v in max(history, key=lambda h: (h['valid_speedup'], h['valid_accuracy'])).items()
			if k != 'train_loss'}
	best['valid_speedup_folds'] = [score(apply_threshold(f['valid'][best['epoch']], data['default'], best['threshold']),
										 valid, data)['speedup'] for f, (_, valid) in zip(fits, parts)]
	picks = apply_threshold(np.mean([f['test'][best['epoch']] for f in fits], axis=0), data['default'], best['threshold'])
	test = score(picks, test_names, data)
	predictions = {n + '_0': {'best': [int(p), mapping[yaml_details[n]['SOLVER']]]} for n, p in zip(test_names, picks)}
	with open(os.path.join(out, 'predictions.json'), 'w') as f:
		json.dump(predictions, f)
	results = {'name': name, 'config': config, **best, **{'test_' + k: v for k, v in test.items()},
			   'inference_ms_per_instance': 1000 * float(sum(f['seconds'] for f in fits)),
			   'parameters': int(fits[0]['parameters']), 'solvers': solvers, 'baseline': baseline, 'oracle': oracle,
			   'history': history, 'folds': len(parts), 'train': len(parts[0][0]), 'valid': len(valid_names),
			   'test': len(test_names)}
	with open(os.path.join(out, 'results.json'), 'w') as f:
		json.dump(results, f, indent='\t')
	print('Best epoch {epoch} (threshold {threshold:g}) | validation speedup {valid_speedup:.4f} | test speedup '
		  '{test_speedup:.4f}, accuracy {test_accuracy:.4f} | {inference_ms_per_instance:.2f} ms per instance'.format(**results), flush=True)

def select(args):
	results = []
	for path in sorted(glob.glob(os.path.join(args.out, '*', 'results.json'))):
		with open(path) as f:
			results.append(json.load(f))
	suffix, planned = '', configurations()
	if args.where:
		key, value = args.where.split('=')
		results = [r for r in results if str(r['config'][key]) == value]
		planned = [c for c in planned if str(c[key]) == value]
		suffix = '_' + value
	if not results:
		raise SystemExit('no results.json in {}{}'.format(args.out, ' with ' + args.where if args.where else ''))
	results.sort(key=lambda r: (r['valid_speedup'], r['valid_accuracy']), reverse=True)
	baseline = results[0]['baseline']
	lines = ['# MAPFAST-GNN hyperparameter sweep' + (' ({})'.format(args.where) if args.where else ''), '',
			 '{} of {} configurations finished, ranked by validation speedup over always using {} (the test columns are '
			 'only reported; with k-fold cross-validation, the folds\' range is in brackets). Validation: always {} '
			 '{:.1f} min, perfect selector {:.1f} min.'.format(
				 len(results), len(planned), baseline['solver'], baseline['solver'], baseline['runtime'],
				 results[0]['oracle']['runtime']), '',
			 '| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |',
			 '|---|---|---|---|---|---|---|---|---|']
	for i, r in enumerate(results):
		folds = r.get('valid_speedup_folds', [])
		spread = ' ({:.3f}–{:.3f})'.format(min(folds), max(folds)) if len(folds) > 1 else ''
		lines.append('| {} | `{}` | {} | {:g} | {:.4f}{} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |'.format(
			i + 1, r['name'], r['epoch'], r['threshold'], r['valid_speedup'], spread, r['valid_accuracy'],
			r['test_speedup'], r['test_accuracy'], r['inference_ms_per_instance']))
	with open(os.path.join(args.out, 'summary{}.md'.format(suffix)), 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print('\n'.join(lines[2:6 + min(10, len(results))]))
	best = results[0]
	print('Best: {} (validation speedup {:.4f}, test speedup {:.4f})'.format(best['name'], best['valid_speedup'], best['test_speedup']))
	if args.dataset_out:
		target = read_config(args.dataset_out)['prediction_output']
		shutil.copy(os.path.join(args.out, best['name'], 'predictions.json'), target)
		print('wrote', target)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and select MAPFAST-GNN')
	commands = parser.add_subparsers(dest='command', required=True)
	commands.add_parser('list', help='Print the number of hyperparameter configurations')
	p = commands.add_parser('train', help='Train one hyperparameter configuration')
	p.add_argument('dataset', help='Dataset folder, e.g. datasets/movingai_10-500-agents_300s/eight-solvers')
	p.add_argument('--config-index', type=int, required=True, help='Configuration to train (0 to list - 1)')
	p.add_argument('--out', required=True, help='Sweep folder; the configuration writes to <out>/<configuration>/')
	p.add_argument('--graphs', default=None, help='Graph file (default: <dataset>/graphs.pkl)')
	p.add_argument('--epochs', type=int, default=40)
	p.add_argument('--batch-size', type=int, default=32)
	p.add_argument('--max-nodes', type=int, default=100000, help='Most nodes in a batch (default 100000)')
	p.add_argument('--dropout', type=float, default=0.1)
	p.add_argument('--weight-decay', type=float, default=1e-4)
	p.add_argument('--limit', type=int, default=None, help='Only train on the first n training instances (a test)')
	p.add_argument('--no-fallback', action='store_true', help='Always pick the top output (threshold 0)')
	p.add_argument('--folds', type=int, default=1, help='Folds of k-fold cross-validation (default 1: main.py\'s '
				   'validation set)')
	p = commands.add_parser('select', help='Rank the trained configurations and keep the best')
	p.add_argument('out', help='Sweep folder')
	p.add_argument('--dataset-out', default=None, help='Dataset folder whose prediction_output gets the best test predictions')
	p.add_argument('--where', default=None, help='Only rank the configurations with KEY=VALUE, e.g. edges=paths; writes '
				   'summary_<VALUE>.md')
	args = parser.parse_args()

	if args.command == 'list':
		print(len(configurations()))
	elif args.command == 'train':
		train(args)
	else:
		select(args)
