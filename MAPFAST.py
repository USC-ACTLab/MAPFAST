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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

		self._model_cache = {}

		file_list = list(yaml_details.keys())
		self.files = {}
		for i in file_list:
			for _ in range(augmentation):
				f = i + '_' + str(_)
				self.files[f] = i

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

	def compute_y2(self, file_name):
		'''
		Computes the ground truth for the finish prediction neurons of a given filename.

		Returns: A numpy array of 0 or 1 denoting if each of the solver successfully solved the input.
		'''
		di = self.yaml_details[file_name]
		y2 = []
		for i in self.mapping:
			y2.append(float(di[i] != -1))
		return asarray(y2)

	def compute_y3(self, file_name):
		'''
		Computes the ground truth for the pairwise comparison neurons of a given filename.

		Returns: A numpy array of 0 or 1 denoting [bcp < cbs, bcp < cbsh, bcp < sat, cbs < cbsh, cbs < sat, cbsh < sat] where < implies faster.
		Note: The length may vary depending on the mapping Json given as input.
		'''
		di = self.yaml_details[file_name]

		solvers_time = []
		for i in self.mapping:
			t = 400
			if di[i] != -1:
				t = di[i]
			solvers_time.append(t)

		y3 = []
		for i in range(len(solvers_time)-1):
			for j in range(i+1, len(solvers_time)):
				y3.append(float(solvers_time[i] <= solvers_time[j]))

		y3 = asarray(y3)

		return y3

	def data_generator(self, files_names_list, batch_size):
		'''
		A data generator function used when training the neural network.

		The arguments are:
			Required Arguments:
				1. files_names_list -> List containing the file names
				2. batch_size -> Size for each batch when yielding the necessary data

		Yields: a tuple of five items
				1. List containing the names of files in the current batch
				2. Image data after augmentation and reshaping
				3. Ground truth for best solver classification neurons
				4. Ground truth for finish prediction neurons
				5. Ground truth for pairwise comparison neurons
		'''
		files_names = files_names_list
		i = 0
		while 1:
			next_batch = files_names[i:i + batch_size]

			X = []
			Y1 = []
			Y2 = []
			Y3 = []

			for _ in next_batch:
				kk = self.files[_]
				# img = img_to_array(load_img(self.input_location + kk[:-4] + 'png', target_size=(320, 320)))
				#img = img_to_array(load_img(self.input_location + kk[:-4] + 'png').resize((320, 320)))
				img = read_image(self.input_location + kk[:-4] + 'png')
				img_transforms = torch.nn.Sequential(
					transforms.Resize([320, 320])
				)
				img = img_transforms(img)

				start = self.agent_details[kk]['starts']
				goal = self.agent_details[kk]['goals']
				new_image, new_start, new_goal = get_transition(img, start, goal, self.map_details[kk], int(_.split('_')[-1]))
				#new_image = np.transpose(new_image, (2, 0, 1))
				new_image = np.asarray(new_image)

				X.append(new_image)

				Y1.append(self.mapping[self.yaml_details[kk]['SOLVER']])
				Y2.append(self.compute_y2(kk))
				Y3.append(self.compute_y3(kk))

			X = asarray(X)
			Y1 = asarray(Y1)
			Y2 = asarray(Y2)
			Y3 = asarray(Y3)
			
			yield next_batch, X, Y1, Y2, Y3
			i += batch_size
			if i >= len(files_names_list):
				i = 0

	def train_model(self, train_list, valid_list, model_loc=None, model_name=None, batch_size=16, epochs=10, log_interval=1000, cl_units=1, fin_pred_units=1, pair_units=1):
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
				7. log_interval -> Default value of 1000. The steps after which the performance should be evaluated with the validation data.
				8. cl_units -> Default value of 1. 0/1 for indicating if best solver classification neurons should be present.
				9. fin_pred_units -> Default value of 1. 0/1 for indicating if finish prediction neurons should be present.
				10. pair_units -> Default value of 1. 0/1 for indicating if pairwise comparison neurons should be present.

		Returns: The trained model
		'''
		train_steps = (len(train_list) + batch_size + 1) // batch_size
		valid_steps = (len(valid_list) + batch_size + 1) // batch_size

		train_datagen = self.data_generator(train_list, batch_size)
		valid_datagen = self.data_generator(valid_list, batch_size)

		if model_loc:
			create_model_weights(model_loc)
		
		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)
		
		if cl_units:
			criterion1 = nn.CrossEntropyLoss()
		if fin_pred_units:
			criterion2 = nn.BCEWithLogitsLoss()
		if pair_units:
			criterion3 = nn.BCEWithLogitsLoss()
		
		optimizer = optim.Adam(net.parameters())

		for i in range(epochs):
			j = 0
			run_loss = 0.0
			for next_batch, X, Y1, Y2, Y3 in train_datagen:
				j += 1
				if j == train_steps:
					break
				X_to = torch.from_numpy(X).float().to(self.device)
				optimizer.zero_grad()
				outs = net(X_to)

				del X_to
				
				losses = []
				if cl_units:
					out1 = outs['cl']
					Y1_to = torch.tensor(Y1).to(self.device)
					loss1 = criterion1(out1, Y1_to)
					losses.append(loss1)
					del Y1_to
					del outs['cl']

				if fin_pred_units:
					out2 = outs['fin']
					Y2_to = torch.tensor(Y2).to(self.device)
					loss2 = criterion2(out2, Y2_to)
					losses.append(loss2)
					del Y2_to
					del outs['fin']

				if pair_units:
					out3 = outs['pair']
					Y3_to = torch.tensor(Y3).to(self.device)
					loss3 = criterion3(out3, Y3_to)
					losses.append(loss3)
					del Y3_to
					del outs['pair']

				loss = sum(losses)
				loss.backward()
				optimizer.step()
				run_loss += loss.item()

				if j % 50 == 0:
					print('Epoch {} | Step {}/{} | avg loss {:.4f} | batch loss {:.4f}'.format(
						i, j, train_steps, run_loss / j, loss.item()), flush=True)

				del outs
				del X
				del Y1
				del Y2
				del Y3
				

				if j % log_interval == 0 and valid_list:
					net.eval()
					valid_loss = 0
					valid_j = 0
					for next_batch, V, L1, L2, L3 in valid_datagen:
						valid_j += 1
						if valid_j == valid_steps:
							break
						#print('valid:', valid_j)
						V_to = torch.from_numpy(V).float().to(self.device)
						Louts = net(V_to)

						del V_to

						valid_losses = []

						if cl_units:
							Lout1 = Louts['cl']
							L1_to = torch.tensor(L1).to(self.device)
							valid_losses.append(criterion1(Lout1, L1_to).item())
							del Louts['cl']
							del Lout1
							del L1_to

						if fin_pred_units:
							Lout2 = Louts['fin']
							L2_to = torch.tensor(L2).to(self.device)
							valid_losses.append(criterion2(Lout2, L2_to).item())
							del Louts['fin']
							del Lout2
							del L2_to

						if pair_units:
							Lout3 = Louts['pair']
							L3_to = torch.tensor(L3).to(self.device)
							valid_losses.append(criterion3(Lout3, L3_to).item())
							del Louts['pair']
							del Lout3
							del L3_to

						del Louts
						del V
						del L1
						del L2
						del L3
						
					print("Num Batches {} / {} | Batch_Loss {} | Valid_Loss {} | Valid_Losses {}".format(j, train_steps, loss / batch_size, sum(valid_losses) / len(valid_list), valid_losses))
					net.train()

			print('Epoch {} complete | mean loss {:.4f}'.format(i, run_loss / max(j, 1)), flush=True)
	
			if model_name and model_loc:
				torch.save(net.state_dict(), model_loc + 'model_' + model_name + '_epoch_' + str(i) + '.pth')
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
			If the model has pairwise classification neurons, the json will have a key as number(corresponsing to the index values of computer_y3) and the value is predicted answer(an integer)
		'''
		
		test_steps = (len(test_list) + batch_size + 1) // batch_size

		test_datagen = self.data_generator(test_list, batch_size)

		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)

		net.load_state_dict(torch.load(model_loc + model_name, map_location=torch.device(self.device)))
		net.eval()

		Y_prediction_data = {}
		j = 0
		sig = nn.Sigmoid()

		for n_b, X, Y1, Y2, Y3 in test_datagen:
			if j == test_steps:
				break
			j += 1
			X_to = torch.from_numpy(X).float().to(self.device)
			outs = net(X_to)
			
			del X_to

			if cl_units:
				out1 = outs['cl']
				k = np.argmax(torch.Tensor.cpu(out1.detach()).numpy(), axis=1)
				del outs['cl']
				del out1
			
			if fin_pred_units:
				out2 = outs['fin']
				temp_out2 = torch.Tensor.cpu(out2.detach())
				del outs['fin']
				del out2

			if pair_units:
				out3 = outs['pair']
				temp_out3 = torch.Tensor.cpu(out3.detach())
				del outs['pair']
				del out3
			
			for i in range(len(n_b)):
				Y_prediction_data[n_b[i]] = {}
		
				if cl_units:
					Y_prediction_data[n_b[i]]['best'] = [k[i].item(), Y1[i].item()]
		
				if fin_pred_units:
					temp_sig = sig(temp_out2[i]).numpy()
					temp_sig_1 = 1 - temp_sig
					for _ in range(len(self.mapping)):
						val = 0
						if temp_sig[_] >= temp_sig_1[_]:
							val = 1
						Y_prediction_data[n_b[i]][self.inv_mapping[_]] = [val, int(Y2[i][_].item())]

				if pair_units:
					temp_sig_2 = sig(temp_out3[i]).numpy()
					temp_sig_2_1 = 1 - temp_sig_2
					for _ in range(len(self.mapping) * (len(self.mapping) - 1) // 2):
						val = 0
						if temp_sig_2[_] >= temp_sig_2_1[_]:
							val = 1
						Y_prediction_data[n_b[i]][_] = val
			
			del outs
			if cl_units:
				del k
			del X
			del Y1
			del Y2
			del Y3
			torch.cuda.empty_cache()

		return Y_prediction_data

	def get_model(self, model_loc, model_name, cl_units=1, fin_pred_units=1, pair_units=1):
		'''
		Builds (or reuses a cached) InceptionClassificationNet with weights loaded from disk.

		Loading the checkpoint from disk and constructing the network is the slow part of
		running inference, so the result is cached on the instance and reused for any later
		call with the same (model_loc, model_name, cl_units, fin_pred_units, pair_units).

		Returns: The loaded net in eval mode, on self.device
		'''
		key = (model_loc, model_name, cl_units, fin_pred_units, pair_units)
		if key not in self._model_cache:
			net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
			net.to(self.device)
			net.load_state_dict(torch.load(model_loc + model_name, map_location=torch.device(self.device)))
			net.eval()
			self._model_cache[key] = net
		return self._model_cache[key]

	def predict(self, file_key, model_loc, model_name, cl_units=1, fin_pred_units=1, pair_units=1):
		'''
		Function to run inference on a single input

		The arguments are:
			Required Arguments:
				1. file_key -> Name of the file (a key in self.files) to run inference on
				2. model_loc -> Location of the model to be retrieved
				3. model_name -> Name of the model to be retrieved
			Optional Arguments:
				4. cl_units -> Default value of 1. 0/1 for indicating if best solver classification neurons should be present.
				5. fin_pred_units -> Default value of 1. 0/1 for indicating if finish prediction neurons should be present.
				6. pair_units -> Default value of 1. 0/1 for indicating if pairwise comparison neurons should be present.

		Returns: Json object with the same keys as a single entry of test_model's output, but without ground truth values
			(since a single-shot input has none).
		'''
		kk = self.files[file_key]
		img = read_image(self.input_location + kk[:-4] + 'png')
		img_transforms = torch.nn.Sequential(
			transforms.Resize([320, 320])
		)
		img = img_transforms(img)

		start = self.agent_details[kk]['starts']
		goal = self.agent_details[kk]['goals']
		new_image, new_start, new_goal = get_transition(img, start, goal, self.map_details[kk], int(file_key.split('_')[-1]))
		new_image = np.asarray(new_image)

		X = asarray([new_image])

		net = self.get_model(model_loc, model_name, cl_units, fin_pred_units, pair_units)

		sig = nn.Sigmoid()

		X_to = torch.from_numpy(X).float().to(self.device)
		with torch.no_grad():
			outs = net(X_to)

		prediction = {}

		if cl_units:
			k = np.argmax(torch.Tensor.cpu(outs['cl']).numpy(), axis=1)
			prediction['best'] = k[0].item()

		if fin_pred_units:
			temp_sig = sig(torch.Tensor.cpu(outs['fin'][0])).numpy()
			temp_sig_1 = 1 - temp_sig
			for _ in range(len(self.mapping)):
				prediction[self.inv_mapping[_]] = int(temp_sig[_] >= temp_sig_1[_])

		if pair_units:
			temp_sig_2 = sig(torch.Tensor.cpu(outs['pair'][0])).numpy()
			temp_sig_2_1 = 1 - temp_sig_2
			for _ in range(len(self.mapping) * (len(self.mapping) - 1) // 2):
				prediction[_] = int(temp_sig_2[_] >= temp_sig_2_1[_])

		return prediction