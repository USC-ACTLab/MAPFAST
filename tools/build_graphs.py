'''
Turn every instance of a dataset into a graph for the GNN selector (models/gnn.py), reading the MovingAI map and
scenario files of MAPF Benchmarking directly: an instance <map>-<type>-<i>_agents<n>.yaml is the first n agents of
benchmarks/<map>/<map>-<type>-<i>.scen on benchmarks/<map>/<map>.map.

The graph is the map's grid, coarsened to at most SIDE x SIDE blocks of k x k cells (k = 1 for a map that fits): one
node per block with a free cell, and an edge between blocks that share a side. Each agent's shortest path (4-connected
A*, as in benchmark_dataset.py) is traced on the full grid, and every node gets

	free		fraction of the block's cells that are free
	starts		log(1 + agents starting in the block)
	goals		log(1 + agents ending in the block)
	paths		log(1 + cells of the block on some agent's path, counted once per agent)
	overlap		log(1 + visits beyond the first of the block's cells), where paths may meet
	shared		fraction of the block's cells on two or more agents' paths

and the graph the global features

	log agents, log free cells, agents per free cell, mean and longest path length / (height + width),
	fraction of path visits that overlap, log cells of the map

No feature depends on where the block is, so the GNN's prediction does not change when the map is flipped or rotated.

	python tools/build_graphs.py datasets/movingai_10-500-agents_300s/four-solvers --benchmarks ~/MAPF-Benchmarking/benchmarks --jobs 16

writes datasets/movingai_10-500-agents_300s/four-solvers/graphs_s64.pkl. The graphs only depend on the instances, so datasets derived from the
same one (tools/derive_dataset.py) can share the file.
'''
import argparse
import json
import math
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np

from benchmark_dataset import load_map, map_name, scen_name, shortest_path

NODE_FEATURES = ('free', 'starts', 'goals', 'paths', 'overlap', 'shared')
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

def coarsen(obstacles, side):
	'''
	Returns: Tuple of
		1. Block size k
		2. Function that sums a (height, width) array over each block, as a (height / k, width / k) array
		3. Numpy array of the [row, column] of each node (block with a free cell), int16
		4. Fraction of free cells of each node's block
	'''
	height, width = obstacles.shape
	k = max(1, math.ceil(max(height, width) / side))
	ph, pw = math.ceil(height / k) * k, math.ceil(width / k) * k

	def block_sum(a):
		padded = np.zeros((ph, pw), dtype=np.float64)
		padded[:height, :width] = a
		return padded.reshape(ph // k, k, pw // k, k).sum(axis=(1, 3))

	free = block_sum(~obstacles) / (k * k)
	coords = np.argwhere(free > 0)
	return k, block_sum, coords.astype(np.int16), free[coords[:, 0], coords[:, 1]]

def build_scenario(task):
	'''
	Builds the graphs of the instances of one scenario file. Its instances are its first n agents for several n, so
	each agent's path is traced once and the counts grow with n.

	Returns: Tuple of the map name, its node coordinates, and the dictionary instance -> {'x': node features (float16),
	'glob': global features (float32)}
	'''
	map_file, scen_file, instances, side = task
	obstacles = load_map(map_file)
	height, width = obstacles.shape
	agents = load_scenario(scen_file)
	k, block_sum, coords, free = coarsen(obstacles, side)
	rows, cols = coords[:, 0], coords[:, 1]
	free_cells = int((~obstacles).sum())

	visits = np.zeros((height, width), dtype=np.int32)
	starts = np.zeros((height, width), dtype=np.int32)
	goals = np.zeros((height, width), dtype=np.int32)
	lengths = []
	graphs = {}
	done = 0
	for instance in sorted(instances, key=agent_count):
		n = agent_count(instance)
		if n > len(agents):
			raise ValueError('{} has {} agents, fewer than {}'.format(scen_file, len(agents), n))
		for start, goal in agents[done:n]:
			path = np.array(shortest_path(obstacles, start, goal), dtype=np.int64).reshape(-1, 2)
			# a cell the path passes twice still counts once for this agent
			cells = np.unique(path, axis=0)
			visits[cells[:, 0], cells[:, 1]] += 1
			starts[start[0], start[1]] += 1
			goals[goal[0], goal[1]] += 1
			lengths.append(max(len(path) - 1, 0))
		done = n

		overlap = np.maximum(visits - 1, 0)
		x = np.stack([
			free,
			np.log1p(block_sum(starts)[rows, cols]),
			np.log1p(block_sum(goals)[rows, cols]),
			np.log1p(block_sum(visits > 0)[rows, cols]),
			np.log1p(block_sum(overlap)[rows, cols]),
			block_sum(visits > 1)[rows, cols] / (k * k),
		], axis=1)
		glob = np.array([
			math.log(n),
			math.log(free_cells),
			n / free_cells,
			np.mean(lengths) / (height + width),
			max(lengths) / (height + width),
			overlap.sum() / max(visits.sum(), 1),
			math.log(height * width),
		])
		graphs[instance] = {'x': x.astype(np.float16), 'glob': glob.astype(np.float32)}
	return map_name(instances[0]), coords, graphs

def read_json(path):
	with open(path) as f:
		return json.load(f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build the GNN graphs of every instance of a dataset from the MovingAI files')
	parser.add_argument('dataset', help='Dataset folder, e.g. datasets/movingai_10-500-agents_300s/eight-solvers')
	parser.add_argument('--benchmarks', default=os.path.expanduser('~/MAPF-Benchmarking/benchmarks'),
						help='MAPF Benchmarking benchmarks/ directory, with <map>/<map>.map and the scenario files')
	parser.add_argument('--side', type=int, default=64, help='Most blocks along each side of the coarsened grid (default 64)')
	parser.add_argument('--limit', type=int, default=None, help='Only build the first n scenario files (a test)')
	parser.add_argument('--out', default=None, help='Output file (default: <dataset>/graphs_s<side>.pkl)')
	parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='Scenario files to build in parallel (default: all CPUs)')
	args = parser.parse_args()

	config = read_json(os.path.join(args.dataset, 'config.json'))['Analysis']
	yaml_details = read_json(os.path.join(args.dataset, config['yaml_details']))
	benchmarks = os.path.expanduser(args.benchmarks)
	out = args.out or os.path.join(args.dataset, 'graphs_s{}.pkl'.format(args.side))

	scenarios = {}
	for instance in yaml_details:
		scenarios.setdefault(scen_name(instance), []).append(instance)
	tasks = []
	for scen, instances in sorted(scenarios.items())[:args.limit]:
		m = map_name(instances[0])
		tasks.append((os.path.join(benchmarks, m, m + '.map'), os.path.join(benchmarks, m, scen + '.scen'), instances, args.side))
	missing = sorted(set(f for t in tasks for f in t[:2] if not os.path.isfile(f)))
	if missing:
		raise SystemExit('{} MovingAI files not found, e.g. {}'.format(len(missing), missing[0]))

	result = {'side': args.side, 'node_features': NODE_FEATURES, 'global_features': GLOBAL_FEATURES, 'maps': {}, 'instances': {}}
	with Pool(args.jobs) as pool:
		for i, (m, coords, graphs) in enumerate(pool.imap_unordered(build_scenario, tasks)):
			result['maps'][m] = coords
			for instance, graph in graphs.items():
				result['instances'][instance] = {'map': m, **graph}
			if (i + 1) % 25 == 0 or i + 1 == len(tasks):
				print('{}/{} scenario files, {} graphs'.format(i + 1, len(tasks), len(result['instances'])), flush=True)

	with open(out + '.tmp', 'wb') as f:
		pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
	os.replace(out + '.tmp', out)
	nodes = [len(c) for c in result['maps'].values()]
	print('wrote {}: {} graphs on {} maps, {}-{} nodes per graph'.format(out, len(result['instances']), len(nodes), min(nodes), max(nodes)))
