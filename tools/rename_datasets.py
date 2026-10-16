'''
Move the datasets of the old flat names (datasets/full_four_solvers_score_aux, ...) into the folder hierarchy that
benchmark_dataset.py, derive_dataset.py, sweep_gnn.sh and compare.sh now write:

	datasets/<benchmark>/<portfolio>/[<variant>/...]
		benchmark	where the instances come from: an experiment of MAPF Benchmarking (e.g. movingai_10-200-agents_60s),
					movingai_pilot (the first small benchmark) or mapfast_paper (the MAPFAST paper's data)
		portfolio	the solvers to pick from: four-solvers, eight-solvers or nine-solvers
		variant		the same instances with another config (score: the relative speed loss, aux: the auxiliary
					outputs, score-aux) or the predictions of another selector (gnn: MAPFAST-GNN, jev<K>: Jev with K
					examples); a variant can have variants (e.g. four-solvers/score/jev20)

Each dataset is moved with everything in it (images, model weights, predictions, graphs), and the paths in every
config.json are rewritten so that derived datasets still find their source's files. File names inside the folders
(model weights, prediction files) keep their old names.

	python tools/rename_datasets.py            print the plan, change nothing
	python tools/rename_datasets.py --apply    move the datasets

Run it when no job uses the datasets. Datasets whose names it does not recognise are listed and left where they are.
'''
import argparse
import json
import os
import re
import subprocess
import sys

DATASETS = 'datasets'
# the config entries that are paths, relative to the dataset's folder (as utils.read_config reads them)
PATH_KEYS = ('yaml_details', 'agent_details', 'map_details', 'test_details', 'input_location', 'model_loc', 'prediction_output')
FIXED = {
	'three_solvers': 'mapfast_paper/four-solvers',
	'nine_solvers': 'movingai_pilot/nine-solvers',
}
# full_<n>_solvers[_300s][_score][_aux][_gnn | _jev<K>]: the datasets built from round 1's benchmark (and its 300 s extension)
FULL = re.compile(r'^full_(four|eight|nine)_solvers(_300s)?(_score)?(_aux)?(?:_(gnn|jev\d+))?$')

def new_name(old):
	'''
	Returns: The new path of a dataset below datasets/, or None if its name is not recognised
	'''
	if old in FIXED:
		return FIXED[old]
	match = FULL.match(old)
	if not match:
		return None
	portfolio, extended, score, aux, selector = match.groups()
	parts = ['movingai_10-500-agents_300s' if extended else 'movingai_10-200-agents_60s', portfolio + '-solvers']
	variant = '-'.join(v for v, on in (('score', score), ('aux', aux)) if on)
	if variant:
		parts.append(variant)
	if selector:
		parts.append(selector)
	return '/'.join(parts)

def plan():
	'''
	Returns: Tuple of {old folder: new folder} (both below datasets/) and the dataset names that are not recognised
	'''
	moves, unknown = {}, []
	for name in sorted(os.listdir(DATASETS)):
		if not os.path.isfile(os.path.join(DATASETS, name, 'config.json')):
			continue
		new = new_name(name)
		if new is None:
			unknown.append(name)
		elif new != name:
			moves[name] = new
	return moves, unknown

def relocate(path, moves):
	'''
	Returns: Where an absolute path under datasets/ is after the moves
	'''
	root = os.path.abspath(DATASETS)
	for old, new in moves.items():
		old_dir = os.path.join(root, old)
		if path == old_dir or path.startswith(old_dir + os.sep):
			return os.path.join(root, new) + path[len(old_dir):]
	return path

def apply(moves):
	root = os.path.abspath(DATASETS)
	# every config's paths, resolved before anything moves
	configs = {}
	for name in sorted(os.listdir(DATASETS)):
		path = os.path.join(DATASETS, name, 'config.json')
		if os.path.isfile(path):
			with open(path) as f:
				configs[os.path.join(root, name)] = json.load(f)
	for old, new in sorted(moves.items(), key=lambda m: m[1].count('/')):
		target = os.path.join(root, new)
		if os.path.exists(os.path.join(target, 'config.json')):
			sys.exit('{} exists, not overwriting it (nothing after this was moved)'.format(target))
		os.makedirs(os.path.dirname(target), exist_ok=True)
		if os.path.isdir(target):
			# a variant moved here before its source: move the source's files in next to it
			for entry in os.listdir(os.path.join(root, old)):
				os.rename(os.path.join(root, old, entry), os.path.join(target, entry))
			os.rmdir(os.path.join(root, old))
		else:
			os.rename(os.path.join(root, old), target)
		print('moved datasets/{} -> datasets/{}'.format(old, new))
	for old_dir, config in configs.items():
		new_dir = relocate(old_dir, moves)
		for section in config.values():
			for key in PATH_KEYS:
				if section.get(key):
					target = relocate(os.path.normpath(os.path.join(old_dir, section[key])), moves)
					rel = os.path.relpath(target, new_dir)
					section[key] = rel + '/' if section[key].endswith('/') else rel
		with open(os.path.join(new_dir, 'config.json'), 'w') as f:
			json.dump(config, f, indent='\t')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Move the datasets into datasets/<benchmark>/<portfolio>/<variant>/')
	parser.add_argument('--apply', action='store_true', help='move them (default: only print the plan)')
	parser.add_argument('--force', action='store_true', help='move them even though Slurm jobs are queued or running')
	args = parser.parse_args()

	moves, unknown = plan()
	for old, new in sorted(moves.items(), key=lambda m: m[1]):
		print('datasets/{:<40} -> datasets/{}'.format(old, new))
	for name in unknown:
		print('datasets/{:<40}    not recognised, left where it is'.format(name))
	if not moves:
		sys.exit('nothing to move')
	if not args.apply:
		print('Nothing was moved: check the plan, then run again with --apply')
		sys.exit(0)
	try:
		jobs = subprocess.run(['squeue', '--me', '-h'], capture_output=True, text=True, timeout=30).stdout.split('\n')
	except (OSError, subprocess.SubprocessError):
		jobs = []
	if any(line.strip() for line in jobs) and not args.force:
		sys.exit('some of your Slurm jobs are queued or running and may use the datasets; wait for them (or --force)')
	apply(moves)
