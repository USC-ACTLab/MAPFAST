'''
Variant of a dataset without copying its images: fewer solvers (--drop) or a new config (--epochs, --aux).

	python tools/derive_dataset.py datasets/<benchmark>/nine-solvers --name <benchmark>/eight-solvers --drop mapf_prioritized_sipp
'''
import argparse
import json
import os
import sys

from benchmark_dataset import keep_solvers, make_config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from utils import read_config, read_json

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make a dataset folder from another one, with fewer solvers or only a new config')
	parser.add_argument('source', help='Dataset folder to derive from, e.g. datasets/movingai_10-500-agents_300s/nine-solvers')
	parser.add_argument('--name', required=True, help='Path of the new dataset below datasets/, e.g. movingai_10-500-agents_300s/eight-solvers')
	parser.add_argument('--drop', nargs='+', default=[], help='Solvers to leave out of the portfolio')
	parser.add_argument('--epochs', type=int, default=None, help='Training epochs of main.py (default: the source\'s)')
	parser.add_argument('--aux', action='store_true', help='Also train MAPFAST\'s auxiliary outputs (finish prediction and pairwise comparison neurons)')
	args = parser.parse_args()

	source = os.path.normpath(args.source)
	config = read_config(source)
	mapping = config['mapping']
	unknown = [s for s in args.drop if s not in mapping]
	if unknown:
		raise SystemExit('not solvers of {}: {}'.format(source, ', '.join(unknown)))
	solvers = sorted((s for s in mapping if s not in args.drop), key=mapping.get)

	source_yaml = read_json(config['yaml_details'])
	yaml_details = {}
	for instance, record in source_yaml.items():
		kept = keep_solvers(record, solvers)
		if kept:
			yaml_details[instance] = kept

	dataset_dir = os.path.join('datasets', args.name)
	os.makedirs(dataset_dir, exist_ok=True)
	with open(os.path.join(dataset_dir, 'yaml_details.json'), 'w') as f:
		json.dump(yaml_details, f)

	new_config = make_config(args.name, {s: i for i, s in enumerate(solvers)}, config['timeout'], config['augmentation'])
	new_config['epochs'] = args.epochs or config['epochs']
	if args.aux:
		new_config['output_units'] = {'cl': 1, 'fin': 1, 'pair': 1}
	for key in ('agent_details', 'map_details', 'input_location'):
		new_config[key] = os.path.relpath(config[key], dataset_dir) + ('/' if key == 'input_location' else '')
	with open(os.path.join(dataset_dir, 'config.json'), 'w') as f:
		json.dump(new_config, f, indent='\t')

	print('{}: {} of {} instances, solvers {}'.format(dataset_dir, len(yaml_details), len(source_yaml), ', '.join(solvers)))
