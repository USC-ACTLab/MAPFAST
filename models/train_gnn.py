'''
Train and select MAPFAST-GNN, the graph neural network solver selector (gnn.py), on a dataset's graphs
(tools/build_graphs.py), over a grid of hyperparameters. The instances are split into training, validation and test
sets exactly as main.py splits them for the CNN, so every model is tested on the same instances.

	python models/train_gnn.py list                                   number of hyperparameter configurations
	python models/train_gnn.py train DATASET --config-index I --out DIR   train configuration I (one Slurm array task)
	python models/train_gnn.py select DIR --dataset-out DATASET2          rank the trained configurations and write
	                                                                      the best one's test predictions for DATASET2

Unlike MAPFAST and MAPFASTv2 (cross-entropy on the fastest solver), it is trained and judged on the runtime it costs:
	regret loss		the expected seconds a pick loses to the fastest solver, (softmax . (t - t_min)) / T, where an
					unsolved run counts the dataset's time limit T: a pick is charged what it costs, so a 0.04 s loss
					barely matters and a timeout the most (the 'ce' configurations train on cross-entropy instead)
	fallback rule	it leaves the default solver (the single solver with the lowest total training runtime, e.g. bcp2)
					only when its top pick has a probability of at least a threshold, chosen on the validation set
					among THRESHOLDS; threshold 0 is its plain argmax, and it can always fall back to the default
	selection		the epoch, the threshold and the configuration with the lowest validation runtime, reported as
					the speedup over always using the default solver (above 1: it beats it)

`train` writes DIR/<configuration>/: results.json (validation and test speedup, accuracy and runtime, per-epoch
history, inference time), predictions.json (test picks, in the format of main.py -T 0) and model.pt. `select` ranks the
configurations by validation speedup (the test set is only reported), writes DIR/summary.md and DIR/sweep.png
(tools/plot_sweep.py) and copies the best predictions to DATASET2's prediction_output, so that analysis.py, plots.py
and compare.sh (NAME=@DATASET2) treat MAPFAST-GNN like the other models. scripts/sweep_gnn.sh runs all three on Slurm.
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

from gnn import SolverGNN, collate
from utils import read_config

# the hyperparameter grid; every other setting is fixed by the command line
# (hidden: width of the node embeddings; layers: message-passing depth, i.e. how many blocks away information reaches)
GRID = {
	'kind': ['gcn', 'sage', 'gin'],
	'hidden': [32, 64, 128, 256],
	'layers': [2, 4, 8],
	'lr': [1e-3, 3e-4],
	'loss': ['ce', 'regret'],
}
# probabilities the top pick needs to replace the default solver (see the fallback rule above)
THRESHOLDS = (0., 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

def configurations():
	'''
	Returns: List of every hyperparameter configuration of GRID, as dictionaries
	'''
	return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]

def config_name(config):
	return '{kind}_h{hidden}_l{layers}_lr{lr:g}_{loss}'.format(**config)

def split(dataset):
	'''
	The instances of the training, validation and test sets of main.py for this dataset (the same seed and split).

	Returns: Tuple of three lists of yaml_details keys, the dataset's config and its yaml_details
	'''
	from MAPFAST import MAPFAST
	config = read_config(dataset)
	training = config['Training']
	with open(training['yaml_details']) as f:
		yaml_details = json.load(f)
	test_details = None
	if training['test_details']:
		with open(training['test_details']) as f:
			test_details = json.load(f)
	random.seed(42)
	mapfast = MAPFAST('cpu', yaml_details, None, None, None, training['mapping'], test_details, 1)
	train, test, valid = mapfast.get_train_valid_test_list()
	strip = lambda names: [n[:n.rindex('_')] for n in names]
	return strip(train), strip(valid), strip(test), config, yaml_details

def solver_times(yaml_details, names, solvers, timeout):
	'''
	Returns: Numpy array (names, solvers) of each solver's time in seconds, `timeout` if it did not solve the instance
	'''
	times = np.array([[yaml_details[n][s] for s in solvers] for n in names], dtype=np.float64)
	return np.where(times == -1, timeout, times)

def apply_threshold(probabilities, default, threshold):
	'''
	Returns: The picks of the fallback rule: the top solver if its probability is at least `threshold`, else `default`
	'''
	picks = probabilities.argmax(1)
	return np.where(probabilities.max(1) >= threshold, picks, default)

def predict(net, names, data, batch_size):
	'''
	Returns: Tuple of the solver probabilities (numpy array (names, solvers)) and the seconds per instance
	'''
	net.eval()
	probabilities = []
	start = time.perf_counter()
	with torch.no_grad():
		for i in range(0, len(names), batch_size):
			x, edges, batch, glob = data['collate'](names[i:i + batch_size])
			probabilities.append(F.softmax(net(x, edges, batch, glob, len(names[i:i + batch_size])), 1).cpu().numpy())
	return np.concatenate(probabilities), (time.perf_counter() - start) / max(len(names), 1)

def score(picks, names, data):
	'''
	Returns: Dictionary of the picks' total runtime (minutes), speedup over always using the default solver, and
	accuracy (the pick is the dataset's fastest solver, as analysis.py counts it)
	'''
	times = data['times'](names)
	runtime = times[np.arange(len(names)), picks].sum() / 60
	return {'runtime': float(runtime), 'speedup': float(times[:, data['default']].sum() / 60 / runtime),
			'accuracy': float((picks == data['labels'](names)).mean())}

def best_threshold(probabilities, names, data):
	'''
	Returns: Tuple of the threshold with the lowest runtime on `names` and its score (see score)
	'''
	scored = [(t, score(apply_threshold(probabilities, data['default'], t), names, data)) for t in THRESHOLDS]
	return min(scored, key=lambda s: (s[1]['runtime'], -s[1]['accuracy']))

def train(args):
	config = configurations()[args.config_index]
	name = config_name(config)
	out = os.path.join(args.out, name)
	os.makedirs(out, exist_ok=True)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Configuration {}: {} on {}'.format(args.config_index, name, device), flush=True)

	train_names, valid_names, test_names, dataset_config, yaml_details = split(args.dataset)
	graph_file = args.graphs or os.path.join(args.dataset, 'graphs_s{}.pkl'.format(args.side))
	with open(graph_file, 'rb') as f:
		graphs = pickle.load(f)
	missing = [n for n in train_names + valid_names + test_names if n not in graphs['instances']]
	if missing:
		raise SystemExit('{} instances have no graph in {}, e.g. {}'.format(len(missing), graph_file, missing[0]))
	if args.limit:
		train_names, valid_names, test_names = train_names[:args.limit], valid_names[:args.limit // 8], test_names[:args.limit // 8]
	mapping = dataset_config['Training']['mapping']
	solvers = sorted(mapping, key=mapping.get)
	timeout = dataset_config['Analysis'].get('timeout', 300)

	# standardise the features with the training set's statistics
	sample = [graphs['instances'][n] for n in train_names[::max(1, len(train_names) // 2000)]]
	nodes = np.concatenate([g['x'] for g in sample]).astype(np.float64)
	node_stats = (nodes.mean(0).astype(np.float32), (nodes.std(0) + 1e-6).astype(np.float32))
	globs = np.stack([g['glob'] for g in sample]).astype(np.float64)
	global_stats = (globs.mean(0).astype(np.float32), (globs.std(0) + 1e-6).astype(np.float32))
	time_rows = dict(zip(yaml_details, solver_times(yaml_details, list(yaml_details), solvers, timeout)))
	data = {
		'collate': lambda names: collate([graphs['instances'][n] for n in names], graphs['maps'], graphs['side'], device, node_stats, global_stats),
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
	steps = args.epochs * ((len(train_names) + args.batch_size - 1) // args.batch_size)
	schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config['lr'], total_steps=steps)

	best = None
	history = []
	order = list(train_names)
	for epoch in range(args.epochs):
		net.train()
		random.shuffle(order)
		total = 0.
		for i in range(0, len(order), args.batch_size):
			names = order[i:i + args.batch_size]
			x, edges, batch, glob = data['collate'](names)
			logits = net(x, edges, batch, glob, len(names))
			if config['loss'] == 'regret':
				times = torch.from_numpy(data['times'](names)).float().to(device)
				lost = times - times.min(1, keepdim=True).values
				loss = (F.softmax(logits, 1) * lost).sum(1).mean() / timeout
			else:
				loss = F.cross_entropy(logits, torch.from_numpy(data['labels'](names)).to(device))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			schedule.step()
			total += loss.item() * len(names)
		probabilities, _ = predict(net, valid_names, data, args.batch_size)
		threshold, valid = best_threshold(probabilities, valid_names, data)
		picks = apply_threshold(probabilities, data['default'], threshold)
		history.append({'epoch': epoch, 'train_loss': total / len(order), 'threshold': threshold,
						**{'valid_' + k: v for k, v in valid.items()},
						'valid_pick_share': np.bincount(picks, minlength=len(solvers)).tolist()})
		print('Epoch {} | loss {:.4f} | validation: speedup {:.4f} over always {} | accuracy {:.4f} | threshold {:g}'.format(
			epoch, total / len(order), valid['speedup'], solvers[data['default']], valid['accuracy'], threshold), flush=True)
		if best is None or (valid['speedup'], valid['accuracy']) > (best['valid_speedup'], best['valid_accuracy']):
			best = {'epoch': epoch, 'threshold': threshold, **{'valid_' + k: v for k, v in valid.items()}}
			torch.save({'state': net.state_dict(), 'config': config, 'mapping': mapping, 'node_stats': node_stats,
						'global_stats': global_stats, 'side': graphs['side'], 'default': data['default'],
						'threshold': threshold}, os.path.join(out, 'model.pt'))

	net.load_state_dict(torch.load(os.path.join(out, 'model.pt'), weights_only=False)['state'])
	probabilities, seconds = predict(net, test_names, data, args.batch_size)
	picks = apply_threshold(probabilities, data['default'], best['threshold'])
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
	if not results:
		raise SystemExit('no results.json in ' + args.out)
	results.sort(key=lambda r: (r['valid_speedup'], r['valid_accuracy']), reverse=True)
	baseline = results[0]['baseline']
	lines = ['# MAPFAST-GNN hyperparameter sweep', '',
			 '{} of {} configurations finished, ranked by validation speedup over always using {} (the test columns are '
			 'only reported). Validation: always {} {:.1f} min, perfect selector {:.1f} min.'.format(
				 len(results), len(configurations()), baseline['solver'], baseline['solver'], baseline['runtime'],
				 results[0]['oracle']['runtime']), '',
			 '| Rank | Configuration | Epoch | Threshold | Val. speedup | Val. accuracy | Test speedup | Test accuracy | ms / instance |',
			 '|---|---|---|---|---|---|---|---|---|']
	for i, r in enumerate(results):
		lines.append('| {} | `{}` | {} | {:g} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |'.format(
			i + 1, r['name'], r['epoch'], r['threshold'], r['valid_speedup'], r['valid_accuracy'], r['test_speedup'],
			r['test_accuracy'], r['inference_ms_per_instance']))
	with open(os.path.join(args.out, 'summary.md'), 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print('\n'.join(lines[2:6 + min(10, len(results))]))
	best = results[0]
	print('Best: {} (validation speedup {:.4f}, test speedup {:.4f})'.format(best['name'], best['valid_speedup'], best['test_speedup']))
	if args.dataset_out:
		target = read_config(args.dataset_out)['Testing']['prediction_output']
		shutil.copy(os.path.join(args.out, best['name'], 'predictions.json'), target)
		print('wrote', target)
	if not args.no_plot:
		sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
		from plot_sweep import plot_sweep
		plot_sweep(results, os.path.join(args.out, 'sweep.png'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train and select MAPFAST-GNN')
	commands = parser.add_subparsers(dest='command', required=True)
	commands.add_parser('list', help='Print the number of hyperparameter configurations')
	p = commands.add_parser('train', help='Train one hyperparameter configuration')
	p.add_argument('dataset', help='Dataset folder, e.g. datasets/movingai_10-500-agents_300s/eight-solvers')
	p.add_argument('--config-index', type=int, required=True, help='Configuration to train (0 to list - 1)')
	p.add_argument('--out', required=True, help='Sweep folder; the configuration writes to <out>/<configuration>/')
	p.add_argument('--graphs', default=None, help='Graph file (default: <dataset>/graphs_s<side>.pkl)')
	p.add_argument('--side', type=int, default=64)
	p.add_argument('--epochs', type=int, default=40)
	p.add_argument('--batch-size', type=int, default=32)
	p.add_argument('--dropout', type=float, default=0.1)
	p.add_argument('--weight-decay', type=float, default=1e-4)
	p.add_argument('--limit', type=int, default=None, help='Only train on the first n training instances (a test)')
	p = commands.add_parser('select', help='Rank the trained configurations and keep the best')
	p.add_argument('out', help='Sweep folder')
	p.add_argument('--dataset-out', default=None, help='Dataset folder whose prediction_output gets the best test predictions')
	p.add_argument('--no-plot', action='store_true', help='Do not draw <out>/sweep.png (see tools/plot_sweep.py)')
	args = parser.parse_args()

	if args.command == 'list':
		print(len(configurations()))
	elif args.command == 'train':
		train(args)
	else:
		select(args)
