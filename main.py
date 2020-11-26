from utils import *
from MAPFAST import *

def train(device, config):
	yaml_details = read_json(config['yaml_details'])
	agent_details = read_json(config['agent_details'])
	map_details = read_json(config['map_details'])

	if config['test_details']:
		test_details = read_json(config['test_details'])
	else:
		test_details = None

	augmentation = config['augmentation']
	batch_size = config['batch_size']
	epochs = config['epochs']

	out_units = config['output_units']
	model_name = config['model_name']
	log_interval = config['log_interval']

	cl_units = out_units['cl']
	fin_pred_units = out_units['fin']
	pair_units = out_units['pair']
	model_loc = config['model_loc']
	mapping = config['mapping']
	is_image = config['is_image']
	input_location = config['input_location']

	mapfast = MAPFAST(device, yaml_details, agent_details, map_details, input_location, mapping, test_details, augmentation, is_image)
	train_list, test_list, valid_list = mapfast.get_train_valid_test_list()
	print('\n\n---------------- Training started ----------------\n\n')
	net = mapfast.train_model(train_list, valid_list, model_loc, model_name, batch_size, epochs, log_interval, cl_units, fin_pred_units, pair_units)
	print('\n\n---------------- Training completed ----------------\n\n')

def test(device, config):
	yaml_details = read_json(config['yaml_details'])
	agent_details = read_json(config['agent_details'])
	map_details = read_json(config['map_details'])

	if config['test_details']:
		test_details = read_json(config['test_details'])
	else:
		test_details = None

	augmentation = config['augmentation']
	batch_size = config['batch_size']
	is_image = config['is_image']
	input_location = config['input_location']
	out_units = config['output_units']
	model_name = config['model_name']
	cl_units = out_units['cl']
	fin_pred_units = out_units['fin']
	pair_units = out_units['pair']
	model_loc = config['model_loc']
	mapping = config['mapping']

	mapfast = MAPFAST(device, yaml_details, agent_details, map_details, input_location, mapping, test_details, augmentation, is_image)

	train_list, test_list, valid_list = mapfast.get_train_valid_test_list()

	prediction_data = mapfast.test_model(test_list, model_loc, model_name, batch_size, cl_units, fin_pred_units, pair_units)

	with open(config['prediction_output'], 'w') as f:
		json.dump(prediction_data, f)

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Current device:', device)

	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='json_files/config.json', help='Give the location of config.json file')
	parser.add_argument('-T', '--type', default=1, help='Training => 1, Testing => 0')
	args = parser.parse_args()

	config = read_json(args.config)

	if int(args.type):
		if 'Training' not in config:
			print('Add Training parameters to config.json')
			sys.exit(0)

		config = config['Training']
		
		train(device, config)
	else:
		if 'Testing' not in config:
			print('Add Testing parameters to config.json')
			sys.exit(0)

		config = config['Testing']
		test(device, config)
