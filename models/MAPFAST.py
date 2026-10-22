import time
import random

import numpy as np
from numpy import asarray
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch.optim as optim

from torchvision.io import read_image
from torchvision import transforms

from utils import *

class MAPFAST:
	'''
	MAPFAST class - The official implementation for our paper "MAPFAST: A Deep Algorithm Selector for Multi Agent Path Finding using Shortest Path Embeddings"

	The arguments are:
		Required Arguments:
			1. device -> Device to train(cpu/gpu)
			2. yaml_details -> Json object which contains file name as key and another json object(which has 'SOLVER' storing the name of the fastest solver and solving time(in seconds) for all the solvers) as value.
			3. agent_details -> Json object which contains file name as key and its value is a json object containing the start and goal locations of the agents.
			4. map_details -> Json object which contains file name as key and another json object(which has number of agents, dimensions of the input map) as value.
			5. input_location -> String denoting the location of all the image files
			6. mapping -> Json object for mapping solver names with a number
		Optional Arguments:
			7. test_details -> Default value of None. It is a json object which contains names of files as keys. This is provided when different models have to be trained and tested with same data.
			8. augmentation -> Default value of 1. Refer the get_transition function in utils.py for more details

	Returns: None
	'''
	def __init__(self, device, yaml_details, agent_details, map_details, input_location, mapping, test_details=None, augmentation=1):
		self.device = device
		self.yaml_details = yaml_details
		self.agent_details = agent_details
		self.map_details = map_details
		self.input_location = input_location
		self.test_details = test_details
		self.augmentation = augmentation
		self.mapping = mapping
		self.inv_mapping = get_inv_mapping(mapping)
		self.files = {i + '_' + str(k): i for i in yaml_details for k in range(augmentation)}
		self.resize = transforms.Resize([320, 320])

	def get_train_valid_test_list(self):
		'''
		Function to get the split for training, testing and validation data. When test_details is provided, the function performs 90:10 split on the remaining data for train and valid list. Else, it performs 80:10:10 for train, valid and test data.

		The split is made over the instances, not their augmented copies, so that no copy of a test or validation
		instance is trained on: the training list holds every augmented copy of its instances, and the validation
		and test lists only the original (transition 0) of theirs. With augmentation 1 the split is the same as
		splitting the copies.

		Returns: Tuple of three lists
				1. File names for training data
				2. File names for test data
				3. File names for validation data
		'''
		file_list = set(self.yaml_details.keys())

		def copies(instances):
			return [i + '_' + str(_) for i in instances for _ in range(self.augmentation)]

		def originals(instances):
			return [i + '_0' for i in instances]

		if self.test_details:
			test_list = list(self.test_details.keys())
			m = list(file_list - set(self.files[f] for f in test_list))
			random.shuffle(m)
			train_size = (1 + (len(m) // 100)) * 90
			train_list = copies(m[:train_size])
			valid_list = originals(m[train_size:])
		else:
			file_list = list(file_list)
			file_list.sort()
			random.shuffle(file_list)
			train_size = (1 + (len(file_list) // 100)) * 80
			valid_size = (len(file_list) - train_size) // 2

			train_list = copies(file_list[:train_size])
			valid_list = originals(file_list[train_size:train_size+valid_size])
			test_list = originals(file_list[train_size+valid_size:])

		if self.augmentation > 1:
			# the copies of an instance would otherwise follow each other in every batch
			random.shuffle(train_list)
		return train_list, test_list, valid_list

	def times(self, file_name):
		'''
		Returns: List of every solver's time on a file in mapping order, -1 for those that did not solve it
		'''
		return [self.yaml_details[file_name][self.inv_mapping[i]] for i in range(len(self.mapping))]

	def compute_y2(self, file_name):
		'''
		Ground truth of the finish prediction neurons: 1 for each solver that solved the input, else 0.
		'''
		return asarray([float(t != -1) for t in self.times(file_name)])

	def compute_y3(self, file_name):
		'''
		Ground truth of the pairwise comparison neurons: for every pair i < j of solvers in mapping order, 1 if solver i
		is at least as fast as solver j (an unsolved input counting 400 s), else 0.
		'''
		t = [400 if x == -1 else x for x in self.times(file_name)]
		return asarray([float(t[i] <= t[j]) for i in range(len(t)) for j in range(i + 1, len(t))])

	def compute_score(self, file_name, relative=True):
		'''
		The score of every solver on a file, in mapping order, 0 for those that did not solve it: the custom score of
		analysis.py, 1 / (1 + time) (up to its constant factor), or with `relative` that divided by the fastest
		solver's, (1 + fastest time) / (1 + time).
		'''
		score = asarray([0. if t == -1 else 1 / (1 + t) for t in self.times(file_name)])
		return score / score.max() if relative and score.max() > 0 else score

	def evaluate(self, net, names, batch_size, relative):
		'''
		Picks a solver with the classification neurons of net for every file in names.

		Returns: Tuple of the mean score (see compute_score) of the picked solvers and the fraction of files where the
			pick is the fastest solver
		'''
		score = correct = 0
		datagen = self.data_generator(names, batch_size)
		with torch.no_grad():
			for _ in range((len(names) + batch_size - 1) // batch_size):
				batch, X, Y1, _, _ = next(datagen)
				picks = net(torch.from_numpy(X).float().to(self.device))['cl'].argmax(dim=1).cpu().numpy()
				for f, p, y in zip(batch, picks, Y1):
					score += self.compute_score(self.files[f], relative)[p]
					correct += int(p == y)
		return score / len(names), correct / len(names)

	def data_generator(self, files_names_list, batch_size):
		'''
		Yields, batch by batch and starting over at the end of files_names_list, a tuple of
			1. List containing the names of files in the current batch
			2. Image data after augmentation and reshaping
			3. Ground truth for best solver classification neurons
			4. Ground truth for finish prediction neurons
			5. Ground truth for pairwise comparison neurons
		'''
		i = 0
		while True:
			batch = files_names_list[i:i + batch_size]
			X, Y1, Y2, Y3 = [], [], [], []
			for f in batch:
				kk = self.files[f]
				img = self.resize(read_image(self.input_location + kk[:-4] + 'png'))
				new_image, _, _ = get_transition(img, self.agent_details[kk]['starts'], self.agent_details[kk]['goals'],
												 self.map_details[kk], int(f.split('_')[-1]))
				X.append(np.asarray(new_image))
				Y1.append(self.mapping[self.yaml_details[kk]['SOLVER']])
				Y2.append(self.compute_y2(kk))
				Y3.append(self.compute_y3(kk))
			yield batch, asarray(X), asarray(Y1), asarray(Y2), asarray(Y3)
			i += batch_size
			if i >= len(files_names_list):
				i = 0

	def train_model(self, train_list, valid_list, model_loc=None, model_name=None, batch_size=16, epochs=10, cl_units=1, fin_pred_units=1, pair_units=1, loss='ce'):
		'''
		Function to train a model of class InceptionClassificationNet in utils.py

		The arguments are:
			Required Arguments:
				1. train_list -> List with file names that belong to the training data
				2. valid_list -> List with file names that belong to the validation data
			Optional Arguments:
				3. model_loc -> Default value of None. Location to store the model
				4. model_name -> Default value of None, Name of the model. When given a model is stored for each epoch at model_loc with name model_name + "_epoch_" + epoch_number
				5. batch_size -> Default value of 16. The size per batch for training the neural network.
				6. epochs -> Default value of 10. The number of epochs to train the network.
				7. cl_units -> Default value of 1. 0/1 for indicating if best solver classification neurons should be present.
				8. fin_pred_units -> Default value of 1. 0/1 for indicating if finish prediction neurons should be present.
				9. pair_units -> Default value of 1. 0/1 for indicating if pairwise comparison neurons should be present.
				10. loss -> Default value of 'ce'. Loss of the best solver classification neurons: 'ce' for cross entropy on
					the fastest solver; 'score' or 'custom' for 1 - the expected score (see compute_score) of the softmax over
					the solvers, relative to the fastest solver ('score') or absolute, analysis.py's custom score ('custom').

		With valid_list, model_loc and model_name, the epoch that does best on the validation data by what the loss trains
		for (the accuracy of its picks for 'ce', their mean score otherwise) is also stored at model_loc with name
		model_name + "_best".

		Returns: The trained model
		'''
		steps = (len(train_list) + batch_size - 1) // batch_size
		datagen = self.data_generator(train_list, batch_size)
		relative = loss != 'custom'
		if model_loc:
			create_model_weights(model_loc)

		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)
		bce = nn.BCEWithLogitsLoss()

		def classification_loss(out, Y, names):
			if loss == 'ce':
				return F.cross_entropy(out, torch.tensor(Y).to(self.device))
			R = torch.tensor(asarray([self.compute_score(self.files[f], relative) for f in names])).float().to(self.device)
			return 1 - (F.softmax(out, dim=1) * R).sum(dim=1).mean()

		optimizer = optim.Adam(net.parameters())
		best_score = None
		for i in range(epochs):
			run_loss = 0.0
			for j in range(1, steps + 1):
				names, X, Y1, Y2, Y3 = next(datagen)
				outs = net(torch.from_numpy(X).float().to(self.device))
				losses = []
				if cl_units:
					losses.append(classification_loss(outs['cl'], Y1, names))
				if fin_pred_units:
					losses.append(bce(outs['fin'], torch.tensor(Y2).to(self.device)))
				if pair_units:
					losses.append(bce(outs['pair'], torch.tensor(Y3).to(self.device)))
				batch_loss = sum(losses)
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
				run_loss += batch_loss.item()
				if j % 50 == 0:
					print('Epoch {} | Step {}/{} | avg loss {:.4f} | batch loss {:.4f}'.format(
						i, j, steps, run_loss / j, batch_loss.item()), flush=True)

			print('Epoch {} complete | mean loss {:.4f}'.format(i, run_loss / steps), flush=True)
			if model_name and model_loc:
				torch.save(net.state_dict(), model_loc + 'model_' + model_name + '_epoch_' + str(i) + '.pth')

			if cl_units and valid_list:
				net.eval()
				score, accuracy = self.evaluate(net, valid_list, batch_size, relative)
				net.train()
				print('Epoch {} validation | mean score {:.4f} | accuracy {:.4f}'.format(i, score, accuracy), flush=True)
				metric = accuracy if loss == 'ce' else score
				if best_score is None or metric > best_score:
					best_score, best_epoch = metric, i
					if model_name and model_loc:
						torch.save(net.state_dict(), model_loc + 'model_' + model_name + '_best.pth')

		if best_score is not None:
			print('Best epoch {} | validation {} {:.4f}'.format(best_epoch, 'accuracy' if loss == 'ce' else 'mean score', best_score), flush=True)
		return net

	def test_model(self, test_list, model_loc, model_name, batch_size=16, cl_units=1, fin_pred_units=1, pair_units=1):
		'''
		Function to test a model of class InceptionClassificationNet in utils.py

		The arguments are:
			Required Arguments:
				1. test_list -> List with file names that belong to the testing data
				2. model_loc -> Location of the model to be retrieved
				3. model_name -> Name of the model to be retrieved
			Optional Arguments:
				4. batch_size -> Default value of 16. The size per batch for testing the neural network.
				5. cl_units -> Default value of 1. 0/1 for indicating if best solver classification neurons should be present.
				6. fin_pred_units -> Default value of 1. 0/1 for indicating if finish prediction neurons should be present.
				7. pair_units -> Default value of 1. 0/1 for indicating if pairwise comparison neurons should be present.

		Returns: Json object containing the file name as key. The value is another json object whose content depends on the model.
			If the model has classification units, the json will have a key 'best' and a value [predicted value, ground truth](a list)
			If the model has finish prediction neurons, the json will have key with name of solver and value [predicted value, ground truth](a list)
			If the model has pairwise classification neurons, the json will have a key as number(corresponding to the index values of compute_y3) and the value is predicted answer(an integer)
		'''
		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)
		net.load_state_dict(torch.load(model_loc + model_name, map_location=torch.device(self.device)))
		net.eval()

		predictions = {}
		datagen = self.data_generator(test_list, batch_size)
		with torch.no_grad():
			for _ in range((len(test_list) + batch_size - 1) // batch_size):
				names, X, Y1, Y2, Y3 = next(datagen)
				outs = {k: v.cpu() for k, v in net(torch.from_numpy(X).float().to(self.device)).items()}
				for i, name in enumerate(names):
					p = predictions[name] = {}
					if cl_units:
						p['best'] = [int(outs['cl'][i].argmax()), int(Y1[i])]
					if fin_pred_units:
						for s, out in enumerate(outs['fin'][i]):
							p[self.inv_mapping[s]] = [int(out >= 0), int(Y2[i][s])]
					if pair_units:
						for s, out in enumerate(outs['pair'][i]):
							p[s] = int(out >= 0)
		return predictions
