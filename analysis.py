import numpy as np
import json
import sys

import argparse

from utils import *

def find_custom_score(yaml_details, mapping, problem, predicted):
	solvers = []
	our = [0 for i in range(len(predicted))]

	for i in mapping:
		t = 0
		if yaml_details[problem][i] != -1:
			t = yaml_details[problem][i]
			t =300 / (1 + t)
		solvers.append(t)

	for i in range(len(predicted)):
		if yaml_details[problem][predicted[i]] != -1:
			our[i] = yaml_details[problem][predicted[i]]
			our[i] = 300 / (1 + our[i])

	denominator = sum(solvers) + sum(our)
	scores_solvers = [solvers[i] / denominator for i in range(len(solvers))]

	score_our = [o / denominator for o in our]

	return scores_solvers, score_our


def find_count(yaml_details, files, solver_types):
	jj = {'fast': [set()], 'solved': [set()]}
	for i in range(len(files)):
		x = yaml_details[files[i][:-2]]
		fast_flag = False
		solved_flag = False
		for _ in solver_types[i]:
			sol = inv_mapping[_]
			if x['SOLVER'] == sol:
				fast_flag = True
			if x[sol] != -1:
				solved_flag = True

		if fast_flag:
			jj['fast'][0].add(files[i])
		if solved_flag:
			jj['solved'][0].add(files[i])

	jj['fast'].append(len(jj['fast'][0]) / len(files))
	jj['solved'].append(len(jj['solved'][0]) / len(files))
	return jj


def calculate_time(yaml_details, inv_mapping, files, solver_type=None):
	ans = 0
	for i in range(len(files)):
		x = yaml_details[files[i][:-2]]
		if solver_type:
			sol = inv_mapping[solver_type[i]]
		else:
			sol = x['SOLVER']
		if x[sol] != -1:
			ans += x[sol]
		else:
			ans += 300
	return ans

def print_util(yaml_details, Y_prediction_data, solver_types=None):
	if solver_types:
		temp = []
		st = ""
		for i in range(len(solver_types)):
			temp.append(mapping[solver_types[i]])
			if i == 0:
				st += solver_types[i]
			else:
				st += ' + ' + solver_types[i]

		solvers = [temp] * len(Y_prediction_data)
	else:
		try:
			solvers = [[Y_prediction_data[i]['best'][0]] for i in Y_prediction_data]
			st = "Our Model"
		except:
			solvers = []
			for i in Y_prediction_data:
				
				x = [Y_prediction_data[i][str(_)] for _ in range(6)]
				if x[0] == 1 and x[1] == 1 and x[2] == 1:
					solver = 0
				elif x[3] == 1 and x[4] == 1:
					solver = 1
				elif x[5] == 1:
					solver = 2
				else:
					solver = 3
				solvers.append([solver])
			st = "Our Model"
	st += ':'

	jj = find_count(yaml_details, list(Y_prediction_data.keys()), solvers)

	print(st)
	print('\tfast:', '\n\t\tcount:', len(jj['fast'][0]), '\n\t\tpercentage:', jj['fast'][1])
	print('\tsolved:', '\n\t\tcount:', len(jj['solved'][0]), '\n\t\tpercentage:', jj['solved'][1])
	return solvers

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-C', '--config', default='json_files/config.json',
						help='Give the location of config.json file')
	args = parser.parse_args()

	config = read_json(args.config)
	if 'Analysis' not in config:
		print('Add Analysis parameters to config.json')
		sys.exit(0)

	config = config['Analysis']

	yaml_details = read_json(config['yaml_details'])
	agent_details = read_json(config['agent_details'])
	map_details = read_json(config['map_details'])

	mapping = config['mapping']
	inv_mapping = get_inv_mapping(mapping)

	with open(config['prediction_output']) as f:
		Y_prediction_data = json.load(f)

	print_util(yaml_details, Y_prediction_data, ['BCP'])
	print_util(yaml_details, Y_prediction_data, ['CBS'])
	print_util(yaml_details, Y_prediction_data, ['CBSH'])
	print_util(yaml_details, Y_prediction_data, ['SAT'])
	s = print_util(yaml_details, Y_prediction_data)
	print('\nTotal Runtime:')
	print('BCP:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys()), [mapping['BCP']]*len(Y_prediction_data)))
	print('CBS:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys()), [mapping['CBS']]*len(Y_prediction_data)))
	print('CBSH:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys()), [mapping['CBSH']]*len(Y_prediction_data)))
	print('SAT:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys()), [mapping['SAT']]*len(Y_prediction_data)))
	print('Our Model:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys()), [i[0] for i in s]))
	print('Optimal:', calculate_time(yaml_details, inv_mapping, list(Y_prediction_data.keys())))


	#print(len(keys[0].intersection(keys[1].intersection(keys[2].intersection(keys[3])))))
	data = [Y_prediction_data]
	model_names = ['Our Model']
	score_solver, score_our = [0 for i in range(len(mapping))], [0 for i in range(len(data))]
	for i in Y_prediction_data:
		predicted = [inv_mapping[data[_][i]['best'][0]] for _ in range(len(data))]
		solver_score, our_score = find_custom_score(yaml_details, mapping, i[:-2], predicted)
		score_solver = [score_solver[j] + solver_score[j] for j in range(len(mapping))]
		score_our = [score_our[j] + our_score[j] for j in range(len(score_our))]

	print('\nCustom Score:')
	t = 0
	for i in mapping:
		print(i+':', score_solver[t])
		t += 1
	t = 0
	for i in model_names:
		print(i+':', score_our[t])
		t += 1