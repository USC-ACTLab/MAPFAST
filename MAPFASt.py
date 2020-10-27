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

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.optim as optim

random.seed(42)

from utils import *

class MAPFASt_Classification:
	def __init__(self, device, yaml_details, agent_details, map_details, input_location, test_details=None, augmentation=1, is_image=1):
		self.device = device
		self.yaml_details = yaml_details
		self.agent_details = agent_details
		self.map_details = map_details
		self.input_location = input_location
		self.test_details = test_details
		self.augmentation = augmentation
		self.is_image = is_image
		
		file_list = list(yaml_details.keys())
		self.files = {}
		for i in file_list:
			for _ in range(augmentation):
				f = i + '_' + str(_)
				self.files[f] = i

	def get_train_valid_test_list(self):

		file_list = set(self.files.keys())

		if self.test_details:
			m = list(file_list - set(test_details.keys()))
			train_size = (1 + (len(m) // 100)) * 90
			train_list = m[:train_size]
			valid_list = m[train_size:]
			test_list = list(test_details.keys())
		else:
			file_list = list(file_list)
			random.shuffle(file_list)
			train_size = (1 + (len(file_list) // 100)) * 80
			valid_size = (len(file_list) - train_size) // 2
			
			train_list = file_list[:train_size]
			valid_list = file_list[train_size:train_size+valid_size]
			test_list = file_list[train_size+valid_size:]

		return train_list, test_list, valid_list

	def compute_y2(self, file_name):
		di = self.yaml_details[file_name]
		y2 = []
		y2.append(float(di['BCP'] != -1))
		y2.append(float(di['CBSH'] != -1))
		y2.append(float(di['CBS'] != -1))
		y2.append(float(di['SAT'] != -1))
		return asarray(y2)

	def compute_y3(self, file_name):
		di = self.yaml_details[file_name]
		bcp = 400
		cbs = 400
		cbsh = 400
		sat = 400

		if di['BCP'] != -1:
			bcp = di['BCP']
		if di['CBS'] != -1:
			cbs = di['CBS']
		if di['CBSH'] != -1:
			cbsh = di['CBSH']
		if di['SAT'] != -1:
			sat = di['SAT']

		'''
		< implies faster
		bcp < cbs, bcp < cbsh, bcp < sat, cbs < cbsh, cbs < sat, cbsh < sat
		'''
		return asarray([float(bcp <= cbs), float(bcp <= cbsh), float(bcp <= sat), float(cbs <= cbsh), float(cbs <= sat), float(cbsh <= sat)])


	def data_generator(self, files_names_list, batch_size):
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
				if self.is_image:
					img = img_to_array(load_img(self.input_location + kk[:-4] + 'png').resize((320, 320)))
					start = self.agent_details[kk]['starts']
					goal = self.agent_details[kk]['goals']
					new_image, new_start, new_goal = get_transition(img, start, goal, self.map_details[kk], int(_.split('_')[-1]))
					new_image = np.transpose(new_image, (2, 0, 1))
				else:
					with np.load(inp_loc + kk[:-4] + 'npz') as fi:
						img = fi.f.arr_0
					img.resize((3, 320, 320))
					start = self.agent_details[kk]['starts']
					goal = self.agent_details[kk]['goals']
					new_image, new_start, new_goal = get_transition(img, start, goal, self.map_details[kk], int(_.split('_')[-1]))

				X.append(new_image)

				Y1.append(mapping[self.yaml_details[kk]['SOLVER']])
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
		train_steps = (len(train_list) + batch_size + 1) // batch_size
		valid_steps = (len(valid_list) + batch_size + 1) // batch_size

		train_datagen = self.data_generator(train_list, batch_size)
		valid_datagen = self.data_generator(valid_list, batch_size)

		if model_loc:
			create_model_weights(model_loc)
		
		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units)
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

				del outs
				del X
				del Y1
				del Y2
				del Y3
				

				if j % log_interval == 0 and valid_list:
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

			print('Iteration', i, ': Loss =', run_loss)
	
			if model_name and model_loc:
				torch.save(net.state_dict(), model_loc + 'model_' + model_name + '_epoch_' + str(i) + '.pth')
		return net

	def test_model(self, test_list, model_loc, model_name, batch_size=16, cl_units=1, fin_pred_units=1, pair_units=1):
		
		test_steps = (len(test_list) + batch_size + 1) // batch_size
		test_steps = 10

		test_datagen = self.data_generator(test_list, batch_size)

		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units)
		net.to(self.device)

		net.load_state_dict(torch.load(model_loc + model_name, map_location=torch.device(self.device)))

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
					for _ in range(4):
						val = 0
						if temp_sig[_] >= temp_sig_1[_]:
							val = 1
						Y_prediction_data[n_b[i]][inv_mapping[_]] = [val, int(Y2[i][_].item())]

				if pair_units:
					temp_sig_2 = sig(temp_out3[i]).numpy()
					temp_sig_2_1 = 1 - temp_sig_2
					for _ in range(6):
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