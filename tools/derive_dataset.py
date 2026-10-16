'''
Make a dataset folder from another one, with fewer solvers or only a new config, without copying or
re-rendering its images: datasets/<name>/ gets (name a variant of a dataset as a folder in it, e.g.
movingai_10-500-agents_300s/eight-solvers/score; see tools/rename_datasets.py)

	yaml_details.json		the source's, without the dropped solvers and with SOLVER relabelled as the fastest of the others
	config.json			a fresh config (see make_config in benchmark_dataset.py), whose other files are the source's

	python tools/derive_dataset.py datasets/movingai_10-500-agents_300s/nine-solvers --name movingai_10-500-agents_300s/eight-solvers --drop mapf_prioritized_sipp
	python tools/derive_dataset.py datasets/movingai_10-500-agents_300s/four-solvers --name movingai_10-500-agents_300s/four-solvers/aux --aux

The instances that only dropped solvers solved are left out. The training, validation and test split depends only on
the instances, so a dataset that keeps them all is split exactly as its source.
'''
import argparse
import json
import os

from benchmark_dataset import make_config

def read_json(path):
	with open(path) as f:
		return json.load(f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make a dataset folder from another one, with fewer solvers or only a new config')
	parser.add_argument('source', help='Dataset folder to derive from, e.g. datasets/movingai_10-500-agents_300s/nine-solvers')
	parser.add_argument('--name', required=True, help='Path of the new dataset below datasets/, e.g. movingai_10-500-agents_300s/eight-solvers')
	parser.add_argument('--drop', nargs='+', default=[], help='Solvers to leave out of the portfolio')
	parser.add_argument('--loss', choices=['ce', 'score'], default='ce', help='Loss of main.py: cross-entropy (default) or the relative speed loss of round 2')
	parser.add_argument('--aux', action='store_true', help='Also train MAPFAST\'s auxiliary outputs (finish prediction and pairwise comparison neurons)')
	args = parser.parse_args()

	source = os.path.normpath(args.source)
	config = read_json(os.path.join(source, 'config.json'))
	mapping = config['Analysis']['mapping']
	unknown = [s for s in args.drop if s not in mapping]
	if unknown:
		raise SystemExit('not solvers of {}: {}'.format(source, ', '.join(unknown)))
	solvers = sorted((s for s in mapping if s not in args.drop), key=mapping.get)

	yaml_details = {}
	source_yaml = read_json(os.path.join(source, config['Analysis']['yaml_details']))
	for instance, record in source_yaml.items():
		times = {s: record[s] for s in solvers}
		solved = {s: t for s, t in times.items() if t != -1}
		if solved:
			yaml_details[instance] = {'SOLVER': min(solved, key=solved.get), **times}

	dataset_dir = os.path.join('datasets', args.name)
	os.makedirs(dataset_dir, exist_ok=True)
	with open(os.path.join(dataset_dir, 'yaml_details.json'), 'w') as f:
		json.dump(yaml_details, f)

	new_config = make_config(args.name, {s: i for i, s in enumerate(solvers)}, config['Analysis'].get('timeout', 300),
							 config['Training'].get('augmentation', 1))
	new_config['Training']['loss'] = args.loss
	if args.aux:
		for section in new_config.values():
			if 'output_units' in section:
				section['output_units'] = {'cl': 1, 'fin': 1, 'pair': 1}
	# the source's agent and map details and images, relative to the new folder like every path of a config
	for section in new_config.values():
		for key in ('agent_details', 'map_details', 'input_location'):
			if key in section:
				path = os.path.relpath(os.path.join(source, config['Training'][key]), dataset_dir)
				section[key] = path + '/' if key == 'input_location' else path
	with open(os.path.join(dataset_dir, 'config.json'), 'w') as f:
		json.dump(new_config, f, indent='\t')

	print('{}: {} of {} instances, solvers {}'.format(dataset_dir, len(yaml_details), len(source_yaml), ', '.join(solvers)))
