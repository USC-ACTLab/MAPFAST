'''
MAPF Benchmarking experiment ({yaml,agent,map}_details.json) -> datasets/<name>/ with MAPFAST's shortest-path images
(white free, red obstacle, black path, green start, blue goal; one pixel per cell). Converts (x, y) to [row, column].

	python tools/benchmark_dataset.py ~/MAPF-Benchmarking/experiments/<experiment> --benchmarks ~/MAPF-Benchmarking/benchmarks --name <benchmark>/<portfolio>
'''
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

from movingai import load_map, map_name, scen_name, shortest_path

DETAILS = ('yaml_details', 'agent_details', 'map_details')

FREE = (255, 255, 255)
OBSTACLE = (255, 0, 0)
PATH = (0, 0, 0)
START = (0, 255, 0)
GOAL = (0, 0, 255)

def render(obstacles, starts, goals, paths):
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
	'''Renders one scenario file's instances, sharing paths between them. Returns the number written.'''
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

def keep_solvers(record, solvers):
	'''`record` with only `solvers`, SOLVER the fastest of them; None if none solved it.'''
	times = {s: record[s] for s in solvers}
	solved = {s: t for s, t in times.items() if t != -1}
	return {'SOLVER': min(solved, key=solved.get), **times} if solved else None

def convert(yaml_details, agent_details, map_details, solvers=None, solved_by=None):
	'''
	Keeps `solvers` and the instances one of them (or of `solved_by`, to match another portfolio) solved,
	in [row, column]. Returns (yaml_details, agent_details, map_details).
	'''
	solvers = solvers or [s for s in next(iter(yaml_details.values())) if s != 'SOLVER']
	yd, ad, md = {}, {}, {}
	for instance, record in yaml_details.items():
		kept = keep_solvers(record, solvers)
		if not kept or (solved_by and all(record[s] == -1 for s in solved_by)):
			continue
		yd[instance] = kept
		ad[instance] = {
			'starts': [[y, x] for x, y in agent_details[instance]['starts']],
			'goals': [[y, x] for x, y in agent_details[instance]['goals']],
		}
		width, height = map_details[instance]['mp_dim']
		md[instance] = {**map_details[instance], 'mp_dim': [height, width]}
	return yd, ad, md

def make_config(name, mapping, timeout, augmentation=1):
	tag = name.replace('/', '_')
	return {
		**{d: d + '.json' for d in DETAILS}, 'input_location': 'images/', 'model_loc': 'model_weights/',
		'prediction_output': 'y_pred_true_{}.json'.format(tag), 'model_name': tag, 'mapping': mapping, 'timeout': timeout,
		'augmentation': augmentation, 'batch_size': 16, 'epochs': 10, 'output_units': {'cl': 1, 'fin': 0, 'pair': 0},
	}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Build a MAPFAST dataset from the MAPFAST files of MAPF Benchmarking')
	parser.add_argument('experiment', help='Experiment folder of MAPF Benchmarking, e.g. ~/MAPF-Benchmarking/experiments/movingai_10-500-agents_300s')
	parser.add_argument('--benchmarks', required=True, help='MAPF Benchmarking benchmarks/ directory, with <map>/<map>.map for every map')
	parser.add_argument('--name', default=None, help='Path of the dataset below datasets/, <benchmark>/<portfolio>, e.g. movingai_10-500-agents_300s/four-solvers (default: the experiment\'s name)')
	parser.add_argument('--solvers', nargs='+', default=None, help='Solver portfolio (default: every solver in the files)')
	parser.add_argument('--solved-by', nargs='+', default=None,
						help='Keep only the instances that one of these solvers solved, e.g. the --solvers of another dataset built from the same benchmark, so that both hold the same instances')
	parser.add_argument('--timeout', type=float, default=300, help='Time limit of the benchmark runs in seconds, charged to unsolved runs (default 300)')
	parser.add_argument('--augmentation', type=int, default=1, choices=range(1, 7),
						help='Copies of each training instance, flipped and rotated (see get_transition in utils.py); 6 uses every transition (default 1: none)')
	parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='Scenario files to render in parallel (default: all CPUs)')
	args = parser.parse_args()

	experiment = os.path.expanduser(args.experiment)
	name = args.name or os.path.basename(os.path.normpath(experiment))
	details = []
	for d in DETAILS:
		with open(os.path.join(experiment, d + '.json')) as f:
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

	print('Train with: python models/main.py -C {}'.format(dataset_dir))
