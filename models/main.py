import argparse
import json
import random
import time

import numpy as np
import torch

from MAPFAST import MAPFAST
from utils import read_config, read_json

def load(device, config):
	'''
	Returns: Tuple of the MAPFAST object of a dataset's config and its (train, test, valid) lists
	'''
	mapfast = MAPFAST(device, read_json(config['yaml_details']), read_json(config['agent_details']),
					  read_json(config['map_details']), config['input_location'], config['mapping'],
					  read_json(config['test_details']) if config['test_details'] else None, config['augmentation'])
	return mapfast, mapfast.get_train_valid_test_list()

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Current device:', device)

	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='datasets/mapfast_paper/four-solvers', help='Give the dataset folder, or the location of its config.json file')
	parser.add_argument('-T', '--type', default=1, help='Training => 1, Testing => 0')
	args = parser.parse_args()

	config = read_config(args.config)
	units = config['output_units']
	units = (units['cl'], units['fin'], units['pair'])

	# the same seed for every dataset: the split, the weight initialisation and the batch order are reproducible
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	mapfast, (train_list, test_list, valid_list) = load(device, config)
	if int(args.type):
		print('\n\n---------------- Training started ----------------\n\n')
		mapfast.train_model(train_list, valid_list, config['model_loc'], config['model_name'], config['batch_size'],
							config['epochs'], *units, config.get('loss', 'ce'))
		print('\n\n---------------- Training completed ----------------\n\n')
	else:
		start = time.perf_counter()
		predictions = mapfast.test_model(test_list, config['model_loc'], 'model_' + config['model_name'] + '_best.pth',
										 config['batch_size'], *units)
		seconds = time.perf_counter() - start
		# loading the images and the model included, in batches of batch_size
		print('Model inference: {:.1f} ms per instance; {:.2f} min over the test set'.format(1000 * seconds / len(test_list), seconds / 60))
		with open(config['prediction_output'], 'w') as f:
			json.dump(predictions, f)
