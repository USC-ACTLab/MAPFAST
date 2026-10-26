import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_image

from utils import InceptionClassificationNet, get_transition

class MAPFAST:
	'''
	MAPFAST (Ren et al., "MAPFAST: A Deep Algorithm Selector for Multi Agent Path Finding using Shortest Path Embeddings").
	yaml_details: instance -> {'SOLVER': fastest, solver: seconds or -1}.
	'''
	def __init__(self, device, yaml_details, input_location, mapping, augmentation=1):
		self.device = device
		self.yaml_details = yaml_details
		self.input_location = input_location
		self.augmentation = augmentation
		self.mapping = mapping
		self.inv_mapping = {v: k for k, v in mapping.items()}
		self.files = {i + '_' + str(k): i for i in yaml_details for k in range(augmentation)}
		self.resize = transforms.Resize([320, 320])

	def get_train_valid_test_list(self):
		'''80:10:10 split over instances, so no augmented copy of a held-out one is trained on. Returns (train, test, valid).'''
		file_list = sorted(self.yaml_details)
		random.shuffle(file_list)
		train_size = (1 + (len(file_list) // 100)) * 80
		valid_size = (len(file_list) - train_size) // 2
		train_list = [i + '_' + str(k) for i in file_list[:train_size] for k in range(self.augmentation)]
		valid_list = [i + '_0' for i in file_list[train_size:train_size + valid_size]]
		test_list = [i + '_0' for i in file_list[train_size + valid_size:]]
		if self.augmentation > 1:
			# otherwise an instance's copies share batches
			random.shuffle(train_list)
		return train_list, test_list, valid_list

	def times(self, file_name):
		return [self.yaml_details[file_name][self.inv_mapping[i]] for i in range(len(self.mapping))]

	def compute_y2(self, file_name):
		'''Finish-prediction targets: 1 per solver that solved it.'''
		return np.asarray([float(t != -1) for t in self.times(file_name)])

	def compute_y3(self, file_name):
		'''Pairwise targets: for i < j, 1 if solver i is at least as fast as j (unsolved = 400 s).'''
		t = [400 if x == -1 else x for x in self.times(file_name)]
		return np.asarray([float(t[i] <= t[j]) for i in range(len(t)) for j in range(i + 1, len(t))])

	def batches(self, names, batch_size):
		'''Yields (names, images, best solver, finish, pairwise) batches.'''
		for i in range(0, len(names), batch_size):
			batch = names[i:i + batch_size]
			X, Y1, Y2, Y3 = [], [], [], []
			for f in batch:
				kk = self.files[f]
				X.append(get_transition(self.resize(read_image(self.input_location + kk[:-4] + 'png')), int(f.split('_')[-1])))
				Y1.append(self.mapping[self.yaml_details[kk]['SOLVER']])
				Y2.append(self.compute_y2(kk))
				Y3.append(self.compute_y3(kk))
			yield batch, np.asarray(X), np.asarray(Y1), np.asarray(Y2), np.asarray(Y3)

	def accuracy(self, net, names, batch_size):
		correct = 0
		with torch.no_grad():
			for _, X, Y1, _, _ in self.batches(names, batch_size):
				correct += int((net(torch.from_numpy(X).float().to(self.device))['cl'].argmax(dim=1).cpu().numpy() == Y1).sum())
		return correct / len(names)

	def train_model(self, train_list, valid_list, model_loc=None, model_name=None, batch_size=16, epochs=10, cl_units=1, fin_pred_units=1, pair_units=1):
		'''Saves every epoch, and the most accurate on validation as model_<name>_best.pth.'''
		steps = (len(train_list) + batch_size - 1) // batch_size
		if model_loc:
			os.makedirs(model_loc, exist_ok=True)

		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)
		bce = nn.BCEWithLogitsLoss()
		optimizer = optim.Adam(net.parameters())
		best_accuracy = None
		for i in range(epochs):
			run_loss = 0.0
			for j, (names, X, Y1, Y2, Y3) in enumerate(self.batches(train_list, batch_size), 1):
				outs = net(torch.from_numpy(X).float().to(self.device))
				losses = []
				if cl_units:
					losses.append(F.cross_entropy(outs['cl'], torch.tensor(Y1).to(self.device)))
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
				accuracy = self.accuracy(net, valid_list, batch_size)
				net.train()
				print('Epoch {} validation | accuracy {:.4f}'.format(i, accuracy), flush=True)
				if best_accuracy is None or accuracy > best_accuracy:
					best_accuracy, best_epoch = accuracy, i
					if model_name and model_loc:
						torch.save(net.state_dict(), model_loc + 'model_' + model_name + '_best.pth')

		if best_accuracy is not None:
			print('Best epoch {} | validation accuracy {:.4f}'.format(best_epoch, best_accuracy), flush=True)
		return net

	def test_model(self, test_list, model_loc, model_name, batch_size=16, cl_units=1, fin_pred_units=1, pair_units=1):
		'''Returns {name: {'best': [pred, true], solver: [pred, true], pair index: pred}} for the heads it has.'''
		net = InceptionClassificationNet(cl_units, fin_pred_units, pair_units, solvers=len(self.mapping))
		net.to(self.device)
		net.load_state_dict(torch.load(model_loc + model_name, map_location=torch.device(self.device)))
		net.eval()

		predictions = {}
		with torch.no_grad():
			for names, X, Y1, Y2, Y3 in self.batches(test_list, batch_size):
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
