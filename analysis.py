import time
import numpy as np
from numpy import asarray
import os
from glob import glob
import json
import sys
from PIL import Image
import pickle
import random

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse

random.seed(42)

from utils import *

def show_confusion(Y_prediction_data):
	y_true = []
	y_pred = []
	for i in Y_prediction_data:
		y_pred.append(Y_prediction_data[i][0])
		y_true.append(Y_prediction_data[i][1])
	plt.imshow(tf.math.confusion_matrix(y_true, y_pred, 4))
	plt.show()


def find_custom_score(problem, predicted):
	bcp = 0
	cbsh = 0
	sat = 0
	cbs = 0
	our = [0 for i in range(len(predicted))]

	if yaml_details[problem[:-2]]['BCP'] != -1:
		bcp = yaml_details[problem[:-2]]['BCP']
		bcp = 300 / (1 + bcp)

	if yaml_details[problem[:-2]]['CBS'] != -1:
		cbs = yaml_details[problem[:-2]]['CBS']
		cbs = 300 / (1 + cbs)

	if yaml_details[problem[:-2]]['CBSH'] != -1:
		cbsh = yaml_details[problem[:-2]]['CBSH']
		cbsh = 300 / (1 + cbsh)

	if yaml_details[problem[:-2]]['SAT'] != -1:
		sat = yaml_details[problem[:-2]]['SAT']
		sat = 300 / (1 + sat)

	for i in range(len(predicted)):
		if yaml_details[problem[:-2]][predicted[i]] != -1:
			our[i] = yaml_details[problem[:-2]][predicted[i]]
			our[i] = 300 / (1 + our[i])

	score_bcp = (1 * bcp) / (bcp + cbs + cbsh + sat + sum(our))
	score_cbs = (1 * cbs) / (bcp + cbs + cbsh + sat + sum(our))
	score_cbsh = (1 * cbsh) / (bcp + cbs + cbsh + sat + sum(our))
	score_sat = (1 * sat) / (bcp + cbs + cbsh + sat + sum(our))
	score_our = [(1 * o) / (bcp + cbs + cbsh + sat + sum(our)) for o in our]

	return score_bcp, score_cbs, score_cbsh, score_sat, score_our


def find_count(files, solver_types):
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


def calculate_time(files, solver_type=None):
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

def analyze_classification_units(Y_prediction_data):
	vals = {'BCP':0, 'CBSH':0, 'CBS':0, 'SAT':0}
	jj = {'BCP':0, 'CBSH':0, 'CBS':0, 'SAT':0}
	cov = {'BCP':0, 'CBSH':0, 'CBS':0, 'SAT':0}
	ove = {'BCP':0, 'CBSH':0, 'CBS':0, 'SAT':0}
	
	for i in Y_prediction_data:
		for m in range(4):
			if inv_mapping[m] not in Y_prediction_data[i]:
				print('The json doesnt have the necessary keys for this analysis')
				return
			if Y_prediction_data[i][inv_mapping[m]][0] == Y_prediction_data[i][inv_mapping[m]][1]:
				vals[inv_mapping[m]] += 1
			if Y_prediction_data[i][inv_mapping[m]][0] == 1:
				jj[inv_mapping[m]] += 1
			if Y_prediction_data[i][inv_mapping[m]][1] == 1:
				cov[inv_mapping[m]] += 1
			if Y_prediction_data[i][inv_mapping[m]][0] == Y_prediction_data[i][inv_mapping[m]][1] == 1:
				ove[inv_mapping[m]] += 1

	print('\n\nPredicted Coverage:')
	for i in jj:
		print(i + ':', jj[i] / len(Y_prediction_data))	

	print('\nCorrectness:')
	for i in vals:
		print(i + ':', vals[i] / len(Y_prediction_data))

	print('\nOverlap:')
	for i in ove:
		print(i + ':', ove[i] / cov[i], ove[i], cov[i])

def print_util(Y_prediction_data, solver_types=None):
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

	jj = find_count(list(Y_prediction_data.keys()), solvers)

	print(st)
	print('\tfast:', '\n\t\tcount:', len(jj['fast'][0]), '\n\t\tpercentage:', jj['fast'][1])
	print('\tsolved:', '\n\t\tcount:', len(jj['solved'][0]), '\n\t\tpercentage:', jj['solved'][1])
	return solvers


def ven_diagram(Y_prediction_data, solver_types1=None, solver_types2=None):
	if solver_types1:
		temp = []
		st1 = ""
		for i in range(len(solver_types1)):
			temp.append(mapping[solver_types1[i]])
			if i == 0:
				st1 += solver_types1[i]
			else:
				st1 += ' + ' + solver_types1[i]

		solvers = [temp] * len(Y_prediction_data)
	else:
		solvers = [[Y_prediction_data[i][0]] for i in Y_prediction_data]
		st1 = "Our Model"
	st1 += ':'

	jj1 = find_count(list(Y_prediction_data.keys()), solvers)

	if solver_types2:
		temp = []
		st2 = ""
		for i in range(len(solver_types2)):
			temp.append(mapping[solver_types2[i]])
			if i == 0:
				st2 += solver_types2[i]
			else:
				st2 += ' + ' + solver_types2[i]

		solvers = [temp] * len(Y_prediction_data)
	else:
		solvers = [[Y_prediction_data[i][0]] for i in Y_prediction_data]
		st2 = "Our Model"
	st2 += ':'

	jj2 = find_count(list(Y_prediction_data.keys()), solvers)

	a = len(Y_prediction_data)
	b = len(jj1['solved'][0])
	c = len(set(list(Y_prediction_data.keys())).intersection(jj1['solved'][0]))
	d = len(jj2['solved'][0])
	e = len(set(list(Y_prediction_data.keys())).intersection(jj2['solved'][0]))
	f = len(jj1['solved'][0].intersection(jj2['solved'][0]))
	g = len(jj1['solved'][0].intersection(jj2['solved'][0]).intersection(set(Y_prediction_data.keys())))
	venn3(subsets=(a, b, c, d, e, f, g), set_labels=('Total', st1, st2), alpha=0.5);
	plt.show()


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

	with open(config['prediction_output']) as f:
		Y_prediction_data = json.load(f)

	print_util(Y_prediction_data, ['BCP'])
	print_util(Y_prediction_data, ['CBS'])
	print_util(Y_prediction_data, ['CBSH'])
	print_util(Y_prediction_data, ['SAT'])
	s = print_util(Y_prediction_data)
	print('\nTotal Runtime:')
	print('BCP:', calculate_time(list(Y_prediction_data.keys()), [mapping['BCP']]*len(Y_prediction_data)))
	print('CBS:', calculate_time(list(Y_prediction_data.keys()), [mapping['CBS']]*len(Y_prediction_data)))
	print('CBSH:', calculate_time(list(Y_prediction_data.keys()), [mapping['CBSH']]*len(Y_prediction_data)))
	print('SAT:', calculate_time(list(Y_prediction_data.keys()), [mapping['SAT']]*len(Y_prediction_data)))
	print('Our Model:', calculate_time(list(Y_prediction_data.keys()), [i[0] for i in s]))
	print('Optimal:', calculate_time(list(Y_prediction_data.keys())))


	#print(len(keys[0].intersection(keys[1].intersection(keys[2].intersection(keys[3])))))
	data = [Y_prediction_data]
	score_bcp, score_cbs, score_cbsh, score_sat, score_our = 0, 0, 0, 0, [0 for i in range(len(data))]
	for i in Y_prediction_data:
		predicted = [inv_mapping[data[_][i]['best'][0]] for _ in range(len(data))]
		bcp, cbs, cbsh, sat, our = find_custom_score(i, predicted)
		score_bcp += bcp
		score_cbs += cbs
		score_cbsh += cbsh
		score_sat += sat
		score_our = [score_our[j] + our[j] for j in range(len(score_our))]

	print('\nCustom Score:')
	print('BCP:', score_bcp)
	print('CBS:', score_cbs)
	print('CBSH:', score_cbsh)
	print('SAT:', score_sat)
	print('Our Model:', score_our)

	#analyze_classification_units(Y_prediction_data)


	'''
	print_util(Y_prediction_data, ['BCP', 'CBS'])
	print_util(Y_prediction_data, ['BCP', 'CBSH'])
	print_util(Y_prediction_data, ['CBSH', 'CBS'])
	print_util(Y_prediction_data, ['BCP', 'CBSH'])
	
	ven_diagram(Y_prediction_data, ['BCP'], ['CBS'])
	ven_diagram(Y_prediction_data, ['BCP'])
	
	show_confusion(Y_prediction_data)
	'''