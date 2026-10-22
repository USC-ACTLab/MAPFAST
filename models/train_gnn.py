'''
Train and select MAPFAST-GNN, the graph neural network solver selector (gnn.py), on a dataset's graphs
(tools/build_graphs.py), over a grid of hyperparameters. The instances are split into training, validation and test
sets exactly as main.py splits them for the CNN, so every model is tested on the same instances.

	python models/train_gnn.py list                                   number of hyperparameter configurations
	python models/train_gnn.py train DATASET --config-index I --out DIR   train configuration I (one Slurm array task)
	python models/train_gnn.py select DIR --dataset-out DATASET2          rank the trained configurations and write
	                                                                      the best one's test predictions for DATASET2

Unlike MAPFAST and MAPFASTv2 (cross-entropy on the fastest solver), it is judged on the runtime it costs:
	losses			'ce': cross-entropy on the fastest solver, the outputs being logits
					'runtime': the outputs are minus each solver's predicted log(1 + seconds), fitted by squared error;
					a run that timed out only counts the dataset's limit T as a lower bound (no loss above log(1 + T))
	fallback rule	it leaves the default solver (the single solver with the lowest total training runtime, e.g. bcp2)
					only when its top pick's output beats the default's by at least a threshold (log-odds for 'ce',
					log of the predicted speedup for 'runtime'), chosen on the validation set among THRESHOLDS;
					threshold 0 is its plain argmax
	selection		the epoch, the threshold and the configuration with the lowest validation runtime, reported as
					the speedup over always using the default solver (above 1: it beats it)

A batch holds up to --batch-size instances and --max-nodes nodes (a graph has a node per free cell of its map).

`select --where KEY=VALUE` ranks only the configurations with that setting, e.g. edges=paths, to compare settings.
`train` writes DIR/<configuration>/: results.json (validation and test speedup, accuracy and runtime, per-epoch
history, inference time), predictions.json (test picks, in the format of main.py -T 0) and model.pt. `select` ranks the
configurations by validation speedup (the test set is only reported), writes DIR/summary.md and DIR/sweep.png
(tools/plot_sweep.py) and copies the best predictions to DATASET2's prediction_output, so that plots.py and
compare.sh (NAME=@DATASET2) treat MAPFAST-GNN like the other models. scripts/sweep_gnn.sh runs all three on Slurm.
'''
import argparse
import glob
import itertools
import json
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from gnn import SolverGNN, collate, graph_size
from utils import read_config

# (hidden: width of the node embeddings; layers: message-passing depth, i.e. how many cells away information reaches;
# edges: 'ends' joins each agent to its start and goal cells, 'paths' to every cell of its path too)
# (hidden: width of the node embeddings; layers: message-passing depth, i.e. how many cells away information reaches)
GRID = {
	'kind': ['sage', 'gin'],
	'hidden': [128],
	'layers': [4, 8],
	'lr': [1e-3],
	'edges': ['ends', 'paths'],
	'loss': ['ce', 'runtime'],
}
# margins by which the top pick's output must beat the default solver's to replace it (see the fallback rule above)
THRESHOLDS = (0., 0.25, 0.5, 1., 2.)

def configurations():
	'''
	Returns: List of every hyperparameter configuration of GRID, as dictionaries
	'''
	return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

def config_name(config):
	return '{kind}_h{hidden}_l{layers}_lr{lr:g}_{edges}_{loss}'.format(**config)

def split(dataset):
	'''
	The instances of the training, validation and test sets of main.py for this dataset (the same seed and split).

	Returns: Tuple of three lists of yaml_details keys, the dataset's config and its yaml_details
	'''
	from MAPFAST import MAPFAST
	config = read_config(dataset)
	with open(config['yaml_details']) as f:
		yaml_details = json.load(f)
	test_details = None
	if config['test_details']:
		with open(config['test_details']) as f:
			test_details = json.load(f)
	random.seed(42)
	mapfast = MAPFAST('cpu', yaml_details, None, None, None, config['mapping'], test_details, 1)
	train, test, valid = mapfast.get_train_valid_test_list()
	strip = lambda names: [n[:n.rindex('_')] for n in names]
	return strip(train), strip(valid), strip(test), config, yaml_details

def solver_times(yaml_details, names, solvers, timeout):
	'''
	Returns: Numpy array (names, solvers) of each solver's time in seconds, `timeout` if it did not solve the instance
	'''
	times = np.array([[yaml_details[n][s] for s in solvers] for n in names], dtype=np.float64)
	return np.where(times == -1, timeout, times)

def apply_threshold(outputs, default, threshold):
	'''
	Returns: The picks of the fallback rule: the top solver if its output beats `default`'s by at least `threshold`,
	else `default`
	'''
	return np.where(outputs.max(1) - outputs[:, default] >= threshold, outputs.argmax(1), default)

def batches(names, data, batch_size, max_nodes):
	'''
	Returns: List of consecutive runs of `names` of at most batch_size instances and max_nodes nodes (but at least one)
	'''
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
	'''
	Returns: Tuple of the outputs (numpy array (names, solvers)) and the seconds per instance
	'''
	net.eval()
	outputs = []
	start = time.perf_counter()
	with torch.no_grad():
		for run in batches(names, data, args.batch_size, args.max_nodes):
			outputs.append(net(*data['collate'](run), len(run)).cpu().numpy())
	return np.concatenate(outputs), (time.perf_counter() - start) / max(len(names), 1)

def score(picks, names, data):
	'''
	Returns: Dictionary of the picks' total runtime (minutes), speedup over always using the default solver, and
	accuracy (the pick is the dataset's fastest solver, as plots.py counts it)
	'''
	times = data['times'](names)
	runtime = times[np.arange(len(names)), picks].sum() / 60
	return {'runtime': float(runtime), 'speedup': float(times[:, data['default']].sum() / 60 / runtime),
			'accuracy': float((picks == data['labels'](names)).mean())}

def best_threshold(outputs, names, data):
	'''
	Returns: Tuple of the threshold with the lowest runtime on `names` and its score (see score)
	'''
	scored = [(t, score(apply_threshold(outputs, data['default'], t), names, data)) for t in THRESHOLDS]
	return min(scored, key=lambda s: (s[1]['runtime'], -s[1]['accuracy']))

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

	# standardise the global features with the training set's statistics (the node features are small already)
	globs = np.stack([graphs['instances'][n]['glob'] for n in train_names]).astype(np.float64)
	global_stats = (globs.mean(0).astype(np.float32), (globs.std(0) + 1e-6).astype(np.float32))
	time_rows = dict(zip(yaml_details, solver_times(yaml_details, list(yaml_details), solvers, timeout)))
	data = {
		'collate': lambda names: collate(names, graphs, device, global_stats, config['edges'] == 'paths'),
		'size': lambda name: graph_size(name, graphs),
		'times': lambda names: np.stack([time_rows[n] for n in names]),
		'labels': lambda names: np.array([mapping[yaml_details[n]['SOLVER']] for n in names]),
	}
	# the default solver: the single solver with the lowest total training runtime
	data['default'] = int(data['times'](train_names).sum(0).argmin())
	baseline = {'solver': solvers[data['default']], **score(np.full(len(valid_names), data['default']), valid_names, data)}
	oracle = score(data['labels'](valid_names), valid_names, data)

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	net = SolverGNN(len(graphs['node_features']), len(graphs['global_features']), len(solvers),
					config['kind'], config['hidden'], config['layers'], args.dropout).to(device)
	optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=args.weight_decay)
	# fixed batches of a random order, visited in a new order every epoch
	order = list(train_names)
	random.shuffle(order)
	runs = batches(order, data, args.batch_size, args.max_nodes)
	schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['lr'], total_steps=args.epochs * len(runs))

	best = None
	history = []
	for epoch in range(args.epochs):
		net.train()
		random.shuffle(runs)
		total = 0.
		for names in runs:
			outputs = net(*data['collate'](names), len(names))
			if config['loss'] == 'runtime':
				times = torch.from_numpy(data['times'](names)).float().to(device)
				error = -outputs - torch.log1p(times)
				# a timeout only bounds the runtime from below
				error = torch.where(times >= timeout, error.clamp(max=0), error)
				loss = (error ** 2).mean()
			else:
				loss = F.cross_entropy(outputs, torch.from_numpy(data['labels'](names)).to(device))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			schedule.step()
			total += loss.item() * len(names)
		outputs, _ = predict(net, valid_names, data, args)
		threshold, valid = best_threshold(outputs, valid_names, data)
		history.append({'epoch': epoch, 'train_loss': total / len(order), 'threshold': threshold,
						**{'valid_' + k: v for k, v in valid.items()}})
		print('Epoch {} | loss {:.4f} | validation: speedup {:.4f} over always {} | accuracy {:.4f} | threshold {:g}'.format(
			epoch, total / len(order), valid['speedup'], solvers[data['default']], valid['accuracy'], threshold), flush=True)
		if best is None or (valid['speedup'], valid['accuracy']) > (best['valid_speedup'], best['valid_accuracy']):
			best = {'epoch': epoch, 'threshold': threshold, **{'valid_' + k: v for k, v in valid.items()}}
			torch.save({'state': net.state_dict(), 'config': config, 'mapping': mapping,
						'global_stats': global_stats, 'default': data['default'],
						'threshold': threshold}, os.path.join(out, 'model.pt'))

	net.load_state_dict(torch.load(os.path.join(out, 'model.pt'), weights_only=False)['state'])
	outputs, seconds = predict(net, test_names, data, args)
	picks = apply_threshold(outputs, data['default'], best['threshold'])
	test = score(picks, test_names, data)
	# the picks are indices of solvers, which is in mapping order (derive_dataset.py numbers the solvers 0, 1, ...)
	predictions = {n + '_0': {'best': [int(p), mapping[yaml_details[n]['SOLVER']]]} for n, p in zip(test_names, picks)}
	with open(os.path.join(out, 'predictions.json'), 'w') as f:
		json.dump(predictions, f)
	results = {'name': name, 'config': config, **best, **{'test_' + k: v for k, v in test.items()},
			   'inference_ms_per_instance': 1000 * seconds, 'parameters': sum(p.numel() for p in net.parameters()),
			   'solvers': solvers, 'baseline': baseline, 'oracle': oracle, 'history': history,
			   'train': len(train_names), 'valid': len(valid_names), 'test': len(test_names)}
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
			 'only reported). Validation: always {} {:.1f} min, perfect selector {:.1f} min.'.format(
				 len(results), len(planned), baseline['solver'], baseline['solver'], baseline['runtime'],
				 results[0]['oracle']['runtime']), '',
			 '| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |',
			 '|---|---|---|---|---|---|---|---|---|']
	for i, r in enumerate(results):
		lines.append('| {} | `{}` | {} | {:g} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |'.format(
			i + 1, r['name'], r['epoch'], r['threshold'], r['valid_speedup'], r['valid_accuracy'], r['test_speedup'],
			r['test_accuracy'], r['inference_ms_per_instance']))
	with open(os.path.join(args.out, 'summary{}.md'.format(suffix)), 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print('\n'.join(lines[2:6 + min(10, len(results))]))
	best = results[0]
	print('Best: {} (validation speedup {:.4f}, test speedup {:.4f})'.format(best['name'], best['valid_speedup'], best['test_speedup']))
	if args.dataset_out:
		target = read_config(args.dataset_out)['prediction_output']
		shutil.copy(os.path.join(args.out, best['name'], 'predictions.json'), target)
		print('wrote', target)
	if not args.no_plot:
		sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
		from plot_sweep import plot_sweep
		plot_sweep(results, os.path.join(args.out, 'sweep{}.png'.format(suffix)))

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
	p = commands.add_parser('select', help='Rank the trained configurations and keep the best')
	p.add_argument('out', help='Sweep folder')
	p.add_argument('--dataset-out', default=None, help='Dataset folder whose prediction_output gets the best test predictions')
	p.add_argument('--where', default=None, help='Only rank the configurations with KEY=VALUE, e.g. edges=paths; writes '
				   'summary_<VALUE>.md and sweep_<VALUE>.png')
	p.add_argument('--no-plot', action='store_true', help='Do not draw <out>/sweep.png (see tools/plot_sweep.py)')
	args = parser.parse_args()

	if args.command == 'list':
		print(len(configurations()))
	elif args.command == 'train':
		train(args)
	else:
		select(args)
