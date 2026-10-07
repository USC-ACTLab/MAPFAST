'''
Turn the MAPFAST dataset written by MAPF Benchmarking (benchmark.py, or the
aggregate job of benchmark.sh) into a dataset folder MAPFAST trains on, datasets/<name>/:

	yaml_details.json, agent_details.json, map_details.json
	images/<instance>.png		one shortest path embedding image per instance
	config.json			Training, Testing and Analysis config for main.py and analysis.py, with paths relative to the folder

	python benchmark_dataset.py ~/MAPF-Benchmarking/output/all_6826438 --benchmarks ~/MAPF-Benchmarking/benchmarks --name nine_solvers --jobs 16

The first argument is the prefix of the three <prefix>_{yaml,agent,map}_details.json files.

MAPF Benchmarking writes coordinates as (x, y) = (column, row) and mp_dim as [width, height];
MAPFAST uses [row, column] and [height, width], so both are converted here.

The images use the encoding of MAPFAST's original images (datasets/three_solvers/images): white free cells, red obstacles,
black cells on the shortest path of some agent, green starts and blue goals, at one pixel per cell
(main.py resizes them to 320 x 320).
'''
import argparse
import functools
import heapq
import json
import os
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image

DETAILS = ('yaml_details', 'agent_details', 'map_details')

FREE = (255, 255, 255)
OBSTACLE = (255, 0, 0)
PATH = (0, 0, 0)
START = (0, 255, 0)
GOAL = (0, 0, 255)

def map_name(instance):
	'''
	Name of the MovingAI map of an instance, e.g. Berlin_1_256 for Berlin_1_256-random-1_agents10.yaml
	'''
	return re.sub(r'-(random|even)-\d+_agents\d+\.yaml$', '', instance)

def scen_name(instance):
	'''
	Name of the scenario file of an instance, e.g. Berlin_1_256-random-1 for Berlin_1_256-random-1_agents10.yaml
	'''
	return re.sub(r'_agents\d+\.yaml$', '', instance)

@functools.lru_cache(maxsize=4)
def load_map(map_file):
	'''
	Returns: Boolean numpy array of shape (height, width), True for obstacles
	'''
	with open(map_file) as f:
		lines = f.read().splitlines()
	height = int(lines[1].split()[1])
	width = int(lines[2].split()[1])
	rows = lines[4:4 + height]
	return np.array([[c in '@TO' for c in row[:width]] for row in rows])

def shortest_path(obstacles, start, goal):
	'''
	A* (4-connected, Manhattan heuristic) from start to goal, both [row, column].

	Returns: List of the [row, column] cells of a shortest path, start and goal included, or [] if there is none
	'''
	height, width = obstacles.shape
	blocked = obstacles.ravel()
	s = start[0] * width + start[1]
	g = goal[0] * width + goal[1]
	gr, gc = goal

	parent = {s: s}
	cost = {s: 0}
	heap = [(abs(start[0] - gr) + abs(start[1] - gc), 0, s)]
	while heap:
		_, d, cell = heapq.heappop(heap)
		d = -d
		if cell == g:
			break
		if d > cost[cell]:
			continue
		r, c = divmod(cell, width)
		for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
			if 0 <= nr < height and 0 <= nc < width:
				n = nr * width + nc
				if not blocked[n] and (n not in cost or d + 1 < cost[n]):
					cost[n] = d + 1
					parent[n] = cell
					# ties broken towards the larger g, i.e. the deeper node
					heapq.heappush(heap, (d + 1 + abs(nr - gr) + abs(nc - gc), -(d + 1), n))
	else:
		return []

	path = [g]
	while path[-1] != s:
		path.append(parent[path[-1]])
	return [list(divmod(cell, width)) for cell in reversed(path)]

def render(obstacles, starts, goals, paths):
	'''
	Returns: (height, width, 3) uint8 numpy array of the shortest path embedding
	'''
	img = np.empty(obstacles.shape + (3,), dtype=np.uint8)
	img[:] = FREE
	img[obstacles] = OBSTACLE
	for path in paths:
		for r, c in path:
			img[r, c] = PATH
	for r, c in starts:
		img[r, c] = START
	for r, c in goals:
		img[r, c] = GOAL
	return img

def render_scenario(task):
	'''
	Writes the image of each instance of one scenario file. The instances of a scenario file
	are its first n agents for several n, so each agent's path is computed once.

	Returns: Number of images written
	'''
	map_file, output_dir, instances = task
	obstacles = load_map(map_file)
	paths = {}
	written = 0
	for instance, starts, goals in instances:
		out = os.path.join(output_dir, instance[:-4] + 'png')
		if os.path.exists(out):
			continue
		for start, goal in zip(starts, goals):
			key = (tuple(start), tuple(goal))
			if key not in paths:
				paths[key] = shortest_path(obstacles, start, goal)
		img = render(obstacles, starts, goals, [paths[(tuple(s), tuple(g))] for s, g in zip(starts, goals)])
		Image.fromarray(img).save(out + '.tmp.png')
		os.replace(out + '.tmp.png', out)
		written += 1
	return written

def convert(yaml_details, agent_details, map_details, solvers=None, solved_by=None):
	'''
	Converts the benchmark's records to MAPFAST's conventions, keeping only `solvers` (default: all)
	and the instances that one of them solved, and relabelling SOLVER as the fastest of them.
	With `solved_by`, only the instances that one of those solvers solved are kept, e.g. the solvers of
	another portfolio, so that two datasets built from the same benchmark hold the same instances.

	Returns: Tuple of the converted yaml_details, agent_details and map_details
	'''
	solvers = solvers or [s for s in next(iter(yaml_details.values())) if s != 'SOLVER']
	yd, ad, md = {}, {}, {}
	for instance, record in yaml_details.items():
		times = {s: record[s] for s in solvers}
		solved = {s: t for s, t in times.items() if t != -1}
		if not solved or (solved_by and all(record[s] == -1 for s in solved_by)):
			continue
		yd[instance] = {'SOLVER': min(solved, key=solved.get), **times}
		ad[instance] = {
			'starts': [[y, x] for x, y in agent_details[instance]['starts']],
			'goals': [[y, x] for x, y in agent_details[instance]['goals']],
		}
		width, height = map_details[instance]['mp_dim']
		md[instance] = {**map_details[instance], 'mp_dim': [height, width]}
	return yd, ad, md

def make_config(name, mapping, timeout, augmentation=1):
	'''
	Returns: Json object with the Training, Testing and Analysis sections for this dataset, with paths relative to its folder
	'''
	files = {d: d + '.json' for d in DETAILS}
	common = {
		**files,
		'test_details': '',
		'augmentation': augmentation,
		'batch_size': 16,
		'input_location': 'images/',
		'output_units': {'cl': 1, 'fin': 0, 'pair': 0},
		'model_loc': 'model_weights/',
		'mapping': mapping,
	}
	prediction_output = 'y_pred_true_{}.json'.format(name)
	return {
		'Training': {**common, 'epochs': 10, 'log_interval': 1000, 'model_name': name},
		'Testing': {**common, 'model_name': 'model_{}_epoch_9.pth'.format(name), 'prediction_output': prediction_output},
		'Analysis': {**files, 'prediction_output': prediction_output, 'mapping': mapping, 'timeout': timeout},
	}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build a MAPFAST dataset from the MAPFAST files of MAPF Benchmarking')
	parser.add_argument('prefix', help='Prefix of the <prefix>_{yaml,agent,map}_details.json files, e.g. ~/MAPF-Benchmarking/output/all_6826438')
	parser.add_argument('--benchmarks', required=True, help='MAPF Benchmarking benchmarks/ directory, with <map>/<map>.map for every map')
	parser.add_argument('--name', default=None, help='Name of the dataset (default: the base name of prefix)')
	parser.add_argument('--solvers', nargs='+', default=None, help='Solver portfolio (default: every solver in the files)')
	parser.add_argument('--solved-by', nargs='+', default=None,
						help='Keep only the instances that one of these solvers solved, e.g. the --solvers of another dataset built from the same benchmark, so that both hold the same instances')
	parser.add_argument('--timeout', type=float, default=60, help='Time limit of the benchmark runs in seconds, charged to unsolved runs by analysis.py (default 60, as in benchmark.sh)')
	parser.add_argument('--augmentation', type=int, default=1, choices=range(1, 7),
						help='Copies of each training instance, flipped and rotated (see get_transition in utils.py); 6 uses every transition (default 1: none)')
	parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='Scenario files to render in parallel (default: all CPUs)')
	args = parser.parse_args()

	prefix = os.path.expanduser(args.prefix)
	name = args.name or os.path.basename(prefix)
	details = []
	for d in DETAILS:
		with open('{}_{}.json'.format(prefix, d)) as f:
			details.append(json.load(f))
	yaml_details, agent_details, map_details = convert(*details, args.solvers, args.solved_by)
	solvers = [s for s in next(iter(yaml_details.values())) if s != 'SOLVER']
	mapping = {s: i for i, s in enumerate(solvers)}

	dataset_dir = os.path.join('datasets', name)
	image_dir = os.path.join(dataset_dir, 'images')
	os.makedirs(image_dir, exist_ok=True)
	for d, records in zip(DETAILS, (yaml_details, agent_details, map_details)):
		with open(os.path.join(dataset_dir, d + '.json'), 'w') as f:
			json.dump(records, f)
	with open(os.path.join(dataset_dir, 'config.json'), 'w') as f:
		json.dump(make_config(name, mapping, args.timeout, args.augmentation), f, indent='\t')
	print('{} of {} instances solved by {}'.format(len(yaml_details), len(details[0]), ', '.join(solvers)), flush=True)

	benchmarks = os.path.expanduser(args.benchmarks)
	scenarios = {}
	for instance in yaml_details:
		scenarios.setdefault(scen_name(instance), []).append(instance)
	tasks = []
	for scen, instances in sorted(scenarios.items()):
		m = map_name(instances[0])
		instances.sort(key=lambda i: len(agent_details[i]['starts']))
		tasks.append((os.path.join(benchmarks, m, m + '.map'), image_dir,
					  [(i, agent_details[i]['starts'], agent_details[i]['goals']) for i in instances]))
	missing = sorted(set(t[0] for t in tasks if not os.path.isfile(t[0])))
	if missing:
		raise SystemExit('map files not found: ' + ', '.join(missing))

	written = 0
	with Pool(args.jobs) as pool:
		for i, n in enumerate(pool.imap_unordered(render_scenario, tasks)):
			written += n
			if (i + 1) % 25 == 0 or i + 1 == len(tasks):
				print('{}/{} scenario files rendered, {} images written'.format(i + 1, len(tasks), written), flush=True)

	print('Train with: python main.py -C {}'.format(dataset_dir))
