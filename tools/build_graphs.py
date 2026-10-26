'''
MAPFAST-GNN graphs (models/gnn.py) of a dataset's instances, from the MovingAI files, into <dataset>/graphs.pkl:
maps (cell edges), scenarios (each agent's start, goal and shortest path) and instances (agent count, global features).
Node features are computed in gnn.collate; none depends on position, so predictions are flip/rotation invariant.

	python tools/build_graphs.py datasets/<benchmark>/<portfolio> --benchmarks ~/MAPF-Benchmarking/benchmarks
'''
import argparse
import math
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

from movingai import agent_count, load_map, load_scenario, map_name, scen_name, shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from utils import read_config, read_json

NODE_FEATURES = ('agent', 'starts', 'goals', 'paths', 'length', 'detour', 'overlap')
GLOBAL_FEATURES = ('log_agents', 'log_free', 'agents_per_free', 'mean_path', 'max_path', 'overlap_fraction', 'log_cells')

def cell_graph(obstacles):
	'''Returns (node index per cell, -1 for obstacles; 4-connected edges in both directions).'''
	free = ~obstacles
	node = np.full(obstacles.shape, -1, dtype=np.int64)
	node[free] = np.arange(free.sum())
	pairs = [(node[1:, :], node[:-1, :]), (node[:, 1:], node[:, :-1])]
	sources = np.concatenate([a[(a >= 0) & (b >= 0)] for a, b in pairs])
	targets = np.concatenate([b[(a >= 0) & (b >= 0)] for a, b in pairs])
	return node, np.stack([np.concatenate([sources, targets]), np.concatenate([targets, sources])]).astype(np.int32)

def build_scenario(task):
	'''Returns (map graph, scenario paths, {instance: {'scenario', 'agents', 'glob'}}) of one scenario file.'''
	map_file, scen_file, instances = task
	obstacles = load_map(map_file)
	height, width = obstacles.shape
	node, edges = cell_graph(obstacles)
	agents = load_scenario(scen_file)
	n_max = max(agent_count(i) for i in instances)
	if n_max > len(agents):
		raise ValueError('{} has {} agents, fewer than {}'.format(scen_file, len(agents), n_max))

	paths, lengths, detour = [], [], []
	for start, goal in agents[:n_max]:
		path = np.array(shortest_path(obstacles, start, goal), dtype=np.int64).reshape(-1, 2)
		paths.append(node[path[:, 0], path[:, 1]])
		lengths.append(max(len(path) - 1, 0))
		detour.append(lengths[-1] / max(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 1))
	scenario = {
		'map': map_name(instances[0]),
		'start': np.array([node[s[0], s[1]] for s, _ in agents[:n_max]], dtype=np.int32),
		'goal': np.array([node[g[0], g[1]] for _, g in agents[:n_max]], dtype=np.int32),
		'offsets': np.concatenate([[0], np.cumsum([len(p) for p in paths])]).astype(np.int64),
		'paths': np.concatenate(paths).astype(np.int32),
		'detour': np.array(detour, dtype=np.float32),
	}

	cells = len(node[node >= 0])
	graphs = {}
	for instance in instances:
		n = agent_count(instance)
		visits = np.bincount(scenario['paths'][:scenario['offsets'][n]], minlength=cells)
		graphs[instance] = {'scenario': scen_name(instance), 'agents': n, 'glob': np.array([
			math.log(n),
			math.log(cells),
			n / cells,
			np.mean(lengths[:n]) / (height + width),
			max(lengths[:n]) / (height + width),
			np.maximum(visits - 1, 0).sum() / max(visits.sum(), 1),
			math.log(height * width),
		], dtype=np.float32)}
	return {'edges': edges, 'cells': cells}, scenario, graphs

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build the GNN graphs of every instance of a dataset from the MovingAI files')
	parser.add_argument('dataset', help='Dataset folder, e.g. datasets/movingai_10-500-agents_300s/eight-solvers')
	parser.add_argument('--benchmarks', default=os.path.expanduser('~/MAPF-Benchmarking/benchmarks'),
						help='MAPF Benchmarking benchmarks/ directory, with <map>/<map>.map and the scenario files')
	parser.add_argument('--limit', type=int, default=None, help='Only build the first n scenario files (a test)')
	parser.add_argument('--out', default=None, help='Output file (default: <dataset>/graphs.pkl)')
	parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='Scenario files to build in parallel (default: all CPUs)')
	args = parser.parse_args()

	yaml_details = read_json(read_config(args.dataset)['yaml_details'])
	benchmarks = os.path.expanduser(args.benchmarks)
	out = args.out or os.path.join(args.dataset, 'graphs.pkl')

	scenarios = {}
	for instance in yaml_details:
		scenarios.setdefault(scen_name(instance), []).append(instance)
	tasks = []
	for scen, instances in sorted(scenarios.items())[:args.limit]:
		m = map_name(instances[0])
		tasks.append((os.path.join(benchmarks, m, m + '.map'), os.path.join(benchmarks, m, scen + '.scen'), instances))
	missing = sorted(set(f for t in tasks for f in t[:2] if not os.path.isfile(f)))
	if missing:
		raise SystemExit('{} MovingAI files not found, e.g. {}'.format(len(missing), missing[0]))

	result = {'node_features': NODE_FEATURES, 'global_features': GLOBAL_FEATURES, 'maps': {}, 'scenarios': {}, 'instances': {}}
	with Pool(args.jobs) as pool:
		for i, (graph, scenario, graphs) in enumerate(pool.imap_unordered(build_scenario, tasks)):
			result['maps'][scenario['map']] = graph
			result['scenarios'][next(iter(graphs.values()))['scenario']] = scenario
			result['instances'].update(graphs)
			if (i + 1) % 25 == 0 or i + 1 == len(tasks):
				print('{}/{} scenario files, {} graphs'.format(i + 1, len(tasks), len(result['instances'])), flush=True)

	with open(out + '.tmp', 'wb') as f:
		pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
	os.replace(out + '.tmp', out)
	cells = [g['cells'] for g in result['maps'].values()]
	print('wrote {}: {} graphs on {} maps, {}-{} cells per map'.format(out, len(result['instances']), len(cells), min(cells), max(cells)))
