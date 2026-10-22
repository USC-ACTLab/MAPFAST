'''
Turn every instance of a dataset into a graph for the GNN selector (models/gnn.py), reading the MovingAI map and
scenario files of MAPF Benchmarking directly: an instance <map>-<type>-<i>_agents<n>.yaml is the first n agents of
benchmarks/<map>/<map>.map.

The graph has a node per free cell of the map, with an edge between 4-connected cells, and a node per agent, with an
edge to its start cell and one to its goal cell, so that which start goes with which goal is kept (and optionally to
every cell of its path, see models/gnn.py). Each agent's
shortest path (4-connected A*, as in benchmark_dataset.py) is traced on the grid. models/gnn.py's collate computes the
node features from these paths, for the instance's first n agents:

	cell		starts		agents starting in the cell
				goals		agents ending in the cell
				paths		log(1 + agents whose path passes the cell)
	agent		length		log(1 + its path length)
				detour		path length / Manhattan distance from start to goal
				overlap		log(1 + visits of other agents' paths to its path's cells)

(a first feature tells agents (1) from cells (0)), and this file keeps the graph's global features

	log agents, log free cells, agents per free cell, mean and longest path length / (height + width),
	fraction of path visits that overlap, log cells of the map

No feature depends on where a cell is, so the GNN's prediction does not change when the map is flipped or rotated.

	python tools/build_graphs.py datasets/movingai_10-500-agents_300s/eight-solvers --benchmarks ~/MAPF-Benchmarking/benchmarks --jobs 16

writes datasets/movingai_10-500-agents_300s/eight-solvers/graphs.pkl:
	maps		map -> {'edges': int32 (2, edges) cell edges in both directions, 'cells': number of free cells}
	scenarios	scenario file -> {'map', 'start', 'goal': int32 cell of each agent, 'offsets': int64 (agents + 1),
				'paths': int32 cells of every agent's path, agent a's being paths[offsets[a]:offsets[a + 1]],
				'detour': float32 per agent}
	instances	instance -> {'scenario', 'agents': n, 'glob': float32 global features}
The graphs only depend on the instances, so datasets derived from the same one (tools/derive_dataset.py) can share it.
'''
import argparse
import json
import math
import os
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np

from benchmark_dataset import load_map, map_name, scen_name, shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from utils import read_config

NODE_FEATURES = ('agent', 'starts', 'goals', 'paths', 'length', 'detour', 'overlap')
GLOBAL_FEATURES = ('log_agents', 'log_free', 'agents_per_free', 'mean_path', 'max_path', 'overlap_fraction', 'log_cells')

def agent_count(instance):
	'''
	Returns: Number of agents of an instance, e.g. 10 for Berlin_1_256-random-1_agents10.yaml
	'''
	return int(re.search(r'_agents(\d+)\.yaml$', instance).group(1))

def load_scenario(scen_file):
	'''
	Returns: List of the ([row, column] start, [row, column] goal) of every agent of a MovingAI scenario file, in file order
	'''
	agents = []
	with open(scen_file) as f:
		if 'version 1' not in f.readline():
			raise ValueError(scen_file + ' is not a version 1 scenario file')
		for line in f:
			fields = line.rstrip('\r\n').split('\t')
			if len(fields) >= 8:
				sx, sy, gx, gy = (int(v) for v in fields[4:8])
				agents.append(([sy, sx], [gy, gx]))
	return agents

def cell_graph(obstacles):
	'''
	Returns: Tuple of the (height, width) array of each free cell's node (-1 for obstacles) and the int32 (2, edges)
	array of the edges between 4-connected free cells, in both directions
	'''
	free = ~obstacles
	node = np.full(obstacles.shape, -1, dtype=np.int64)
	node[free] = np.arange(free.sum())
	pairs = [(node[1:, :], node[:-1, :]), (node[:, 1:], node[:, :-1])]
	sources = np.concatenate([a[(a >= 0) & (b >= 0)] for a, b in pairs])
	targets = np.concatenate([b[(a >= 0) & (b >= 0)] for a, b in pairs])
	return node, np.stack([np.concatenate([sources, targets]), np.concatenate([targets, sources])]).astype(np.int32)

def build_scenario(task):
	'''
	Traces the paths of the agents of one scenario file that its instances use (their first n agents for several n).

	Returns: Tuple of the map name, its graph, the scenario's paths and the dictionary instance -> {'agents', 'glob'}
	'''
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

def read_json(path):
	with open(path) as f:
		return json.load(f)

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
