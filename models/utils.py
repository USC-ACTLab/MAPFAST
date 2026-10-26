import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_json(file):
	if not os.path.exists(file):
		raise SystemExit(file + ' was not found')
	with open(file) as f:
		return json.load(f)

CONFIG_PATHS = ('yaml_details', 'agent_details', 'map_details', 'input_location', 'model_loc', 'prediction_output')

def read_config(path):
	'''Config of a dataset folder (or its config.json), with paths joined to that folder; flattens the old layout.'''
	if os.path.isdir(path):
		path = os.path.join(path, 'config.json')
	config = read_json(path)
	if 'Training' in config:
		config = {**config['Analysis'], **config['Testing'], **config['Training']}
	root = os.path.dirname(path)
	for key in CONFIG_PATHS:
		if config.get(key):
			config[key] = os.path.join(root, config[key])
	return config

class Conv2dSame(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super().__init__()
		self.k = kernel_size // 2
		self.net = nn.Conv2d(in_channels, out_channels, kernel_size)

	def forward(self, x):
		return self.net(F.pad(x, (self.k,) * 4))

class MaxPool2dSame(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.k = kernel_size // 2
		self.net = nn.MaxPool2d(kernel_size, stride=1)

	def forward(self, x):
		return self.net(F.pad(x, (self.k,) * 4))

class InceptionClassificationNet(nn.Module):
	'''The paper's Inception network; cl/fin/pair_units toggle the best-solver, finish and pairwise heads.'''
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

	def inception(self, x, first):
		'''1x1, 3x3, 5x5 convolutions and a max pool side by side; modules 2 and 3 share weights.'''
		if first:
			c1, m1, c3, m2, c5, pool, after = self.conv1, self.conv_mid_1, self.conv3, self.conv_mid_2, self.conv5, self.pool1, self.conv_after
		else:
			c1, m1, c3, m2, c5, pool, after = self.conv1_, self.conv_mid_1_, self.conv3_, self.conv_mid_2_, self.conv5_, self.pool1_, self.conv_after_
		return torch.cat((c1(x), c3(m1(x)), c5(m2(x)), after(pool(x))), dim=1)

	def forward(self, x):
		x = F.relu(self.batch1(self.pool2(self.inception(x, True))))
		x = F.relu(self.batch2(self.pool2(self.inception(x, False))))
		x = F.relu(self.batch3(self.pool2(self.inception(x, False))))
		both = self.linear1(x.flatten(1))
		outs = {}
		if self.cl_units:
			outs['cl'] = self.linear2(both)
		if self.fin_pred_units:
			outs['fin'] = self.linear3(both)
		if self.pair_units:
			outs['pair'] = self.linear4(both)
		return outs

# none, flip rows, flip columns, rotate 90, 180, 270 degrees, of a (3, height, width) image
TRANSITIONS = (
	lambda image: image,
	lambda image: np.flip(image, axis=1),
	lambda image: np.flip(image, axis=2),
	lambda image: np.rot90(image, axes=(2, 1)),
	lambda image: np.rot90(image, 2, axes=(1, 2)),
	lambda image: np.rot90(image, axes=(1, 2)),
)

def get_transition(image, transition):
	return TRANSITIONS[transition](np.asarray(image))
