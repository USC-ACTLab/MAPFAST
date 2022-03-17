from numpy import asarray
import numpy as np
import os
import json
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def create_model_weights(loc='model_weights'):
	if not os.path.isdir(loc):
		os.makedirs(loc)

def read_json(file):
	if os.path.exists(file):
		json_obj = json.load(open(file))
		return json_obj
	else:
		print(file, 'was not found!')
		sys.exit(0)

def get_inv_mapping(mapping):
	'''
	Given a Json object which stores name of solvers as key and a number a value, find the inverse mapping(number-solver)

	Returns: Json object corresponding to inverse mapping
	'''
	j = {}
	for i in mapping:
		if mapping[i] in j:
			print('Ensure that values in mapping json is unique')
		j[mapping[i]] = i
	return j

class Conv2dSame(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
		super().__init__()
		self.k = kernel_size // 2
		self.net = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

	def forward(self, x):
		x = F.pad(input=x, pad=(self.k, self.k, self.k, self.k), mode='constant', value=0)
		return self.net(x)

class MaxPool2dSame(nn.Module):
	def __init__(self, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
		super().__init__()
		self.k = kernel_size // 2
		self.net = nn.MaxPool2d(kernel_size, stride=1)

	def forward(self, x):
		x = F.pad(input=x, pad=(self.k, self.k, self.k, self.k), mode='constant', value=0)
		return self.net(x)

class InceptionClassificationNet(nn.Module):
	'''
	Inception model class as described in MAPFAST paper.

	The arguments are:
		Optional Arguments:
			1. cl_units -> Default value of 1. 0/1 for indicating if best solver classification neurons should be present.
			2. fin_pred_units -> Default value of 1. 0/1 for indicating if finish prediction neurons should be present.
			3. pair_units -> Default value of 1. 0/1 for indicating if pairwise comparison neurons should be present.
			4. input_d -> Default value of 3. Number of channels in the input
			5. solvers -> Default value of 4. Number of solvers in our portfolio. This is used to change the number of neurons in the output layer.

	Returns: None
	'''
	def __init__(self, cl_units=True, fin_pred_units=True, pair_units=True, input_d=3, solvers=4):
		super(InceptionClassificationNet, self).__init__()

		self.cl_units = cl_units
		self.fin_pred_units = fin_pred_units
		self.pair_units = pair_units
		self.solvers_count = solvers

		self.conv1 = Conv2dSame(input_d, 32, 1)
		self.conv_mid_1 = Conv2dSame(input_d, 96, 1)
		self.conv3 = Conv2dSame(96, 32, 3)
		self.conv_mid_2 = Conv2dSame(input_d, 16, 1)
		self.conv5 = Conv2dSame(16, 32, 5)
		self.pool1 = MaxPool2dSame(3)
		self.conv_after = Conv2dSame(input_d, 32, 1)

		self.conv1_ = Conv2dSame(128, 32, 1)
		self.conv_mid_1_ = Conv2dSame(128, 96, 1)
		self.conv3_ = Conv2dSame(96, 32, 3)
		self.conv_mid_2_ = Conv2dSame(128, 16, 1)
		self.conv5_ = Conv2dSame(16, 32, 5)
		self.pool1_ = MaxPool2dSame(3)
		self.conv_after_ = Conv2dSame(128, 32, 1)

		self.pool2 = nn.MaxPool2d(3, stride=3)

		self.batch1 = nn.BatchNorm2d(128)
		self.batch2 = nn.BatchNorm2d(128)
		self.batch3 = nn.BatchNorm2d(128)

		val = 15488
		self.linear1 = nn.Linear(val, 200)

		if cl_units:
			self.linear2 = nn.Linear(200, self.solvers_count)

		if fin_pred_units:
			self.linear3 = nn.Linear(200, self.solvers_count)

		if pair_units:
			m = (self.solvers_count * (self.solvers_count - 1))//2
			self.linear4 = nn.Linear(200, m)

	def forward(self, x1):
		'''
		Forward function for our Inception module.

		Returns: Json object whose content depends on the model.
			If the model has classification units, the json will have a key 'cl' and a value corresponding to the output of forward propagation step.
			If the model has finish prediction neurons, the json will have a key 'fin' and a value corresponding to the output of forward propagation step.
			If the model has pairwise classification neurons, the json will have a key 'pair' and a value corresponding to the output of forward propagation step.
		'''
		cov1 = self.conv1(x1)
		cov2 = self.conv3(self.conv_mid_1(x1))
		cov3 = self.conv5(self.conv_mid_2(x1))
		cov4 = self.conv_after(self.pool1(x1))

		del x1

		conv = torch.cat((cov1, cov2, cov3, cov4), dim=1)

		del cov1
		del cov2
		del cov3
		del cov4

		conv = self.pool2(conv)
		conv = self.batch1(conv)
		conv = F.relu(conv)

		cov1 = self.conv1_(conv)
		cov2 = self.conv3_(self.conv_mid_1_(conv))
		cov3 = self.conv5_(self.conv_mid_2_(conv))
		cov4 = self.conv_after_(self.pool1_(conv))

		del conv

		conv = torch.cat((cov1, cov2, cov3, cov4), dim=1)

		del cov1
		del cov2
		del cov3
		del cov4

		conv = self.pool2(conv)
		conv = self.batch2(conv)
		conv = F.relu(conv)

		cov1 = self.conv1_(conv)
		cov2 = self.conv3_(self.conv_mid_1_(conv))
		cov3 = self.conv5_(self.conv_mid_2_(conv))
		cov4 = self.conv_after_(self.pool1_(conv))

		del conv

		conv = torch.cat((cov1, cov2, cov3, cov4), dim=1)

		del cov1
		del cov2
		del cov3
		del cov4

		conv = self.pool2(conv)
		conv = self.batch3(conv)
		conv = F.relu(conv)
		conv = conv.view(-1, self.num_features(conv))

		both = self.linear1(conv)

		del conv

		outs = {}
	
		if self.cl_units:
			out1 = self.linear2(both)
			outs['cl'] = out1

		if self.fin_pred_units:
			out2 = self.linear3(both)
			outs['fin'] = out2

		if self.pair_units:
			out3 = self.linear4(both)
			outs['pair'] = out3

		del both

		return outs


	def num_features(self, x):
		s = x.size()[1:]
		n = 1
		for i in s:
			n *= i
		return n

def horizontal_flip(li, map_details):
	'''
	Returns the (x, y) coordinates after horizantal flip

	Columns -> Same
	Rows -> Map Height - 1 - Current Row Val #0 indexed
	'''
	row = map_details['mp_dim'][0] - 1

	ans = []
	for i in li:
		ans.append([row - i[0], i[1]])
	return ans

def vertical_flip(li, map_details):
	'''
	Returns the (x, y) coordinates after vertical flip

	Columns -> Map Width - 1 - Current Column Val #0 indexed
	Rows -> Same
	'''
	column = map_details['mp_dim'][1] - 1

	ans = []
	for i in li:
		ans.append([i[0], column - i[1]])
	return ans

def ninety_degree_rotation(li, map_details):
	'''
	Returns the (x, y) coordinates after 90 degree rotation

	Columns -> Map Height - 1 - Current Row Val #0 indexed
	Rows -> Current Column Val
	'''
	row = map_details['mp_dim'][0] - 1

	ans = []
	for i in li:
		ans.append([i[1], row - i[0]])
	return ans

def one_eighty_degree_rotation(li, map_details):
	'''
	Returns the (x, y) coordinates after 180 degree rotation
	'''
	return vertical_flip(horizontal_flip(li, map_details), map_details)

def two_seventy_degree_rotation(li, map_details):
	'''
	Returns the (x, y) coordinates after 270 degree rotation
	'''
	return ninety_degree_rotation(one_eighty_degree_rotation(li, map_details), map_details)

def get_transition(image_data, start, goal, map_details, transition):
	'''
	Get a transition for the image, start and goal location

	The arguments are:
		Required Arguments:
			1. image_data -> Numpy array of the image
			2. start -> List containing the (x, y) positions of the start locations of agents in the input map
			3. goal -> List containing the (x, y) positions of the goal locations of agents in the input map
			4. transition -> Number of the transition
								0 -> No transition
								1 -> Horizantal flip
								2 -> Vertical flip
								3 -> 90 degree rotation
								4 -> 180 degree rotation
								5 -> 270 degree rotation

	Returns: Tuple of three items
			1. Numpy array of image data after transition
			2. List of start locations after transition
			3. List of goal locations after transition
	'''
	if transition == 0:
		#No transition
		return image_data, start, goal
	if transition == 1:
		#Horizantal flip
		new_start = horizontal_flip(start, map_details)
		new_goal = horizontal_flip(goal, map_details)
		new_image_data = np.flipud(image_data)
		return new_image_data, new_start, new_goal
	if transition == 2:
		#Vertical flip
		new_start = vertical_flip(start, map_details)
		new_goal = vertical_flip(goal, map_details)
		new_image_data = np.fliplr(image_data)
		return new_image_data, new_start, new_goal
	if transition == 3:
		#90 degree rotation
		new_start = ninety_degree_rotation(start, map_details)
		new_goal = ninety_degree_rotation(goal, map_details)
		new_image_data = np.rot90(image_data, axes=(1,0))
		return new_image_data, new_start, new_goal
	if transition == 4:
		#180 degree rotation
		new_start = one_eighty_degree_rotation(start, map_details)
		new_goal = one_eighty_degree_rotation(goal, map_details)
		image_data = np.rot90(image_data)
		new_image_data = np.rot90(image_data)
		return new_image_data, new_start, new_goal
	if transition == 5:
		#270 degree rotation
		new_start = two_seventy_degree_rotation(start, map_details)
		new_goal = two_seventy_degree_rotation(goal, map_details)
		new_image_data = np.rot90(image_data)
		return new_image_data, new_start, new_goal
	return image_data, start, goal