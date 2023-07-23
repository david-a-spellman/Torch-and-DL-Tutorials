# How to use torch to transform datasets when they are loaded

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

Class WineDataset (Dataset):

	def __init__ (self, transform = None):
		xy = np.loadtxt (".\wine\wine.csv", delimiter = ",", dtype = np.float32, skiprows = 1)
		self.n_samples = xy.shape [0]
		self.transform = transform

	def __getitem__ (self, index):
		sample = self.x [index], self.y [index]
		if self.transform:
			return self.transform (sample)
		else:
			return sample
	def __len__ (self):
		return self.n_samples

# ToTensor class for going from numpy array to tensor
class ToTensor:
	def __call__ (self, sample):
		inputs, targets = sample
		return torch.from_numpy (inputs), torch.from_numpy (targets)

class MulTransform:
	def __init__ (self, factor):
		self.factor = factor

	def __call__ (self, sample):
		inputs, target = sample
		inputs *= self.factor
		return inputs, target



dataset = WineDataset (transform = ToTransform ())
print (dataset)

first_sample = dataset [0]
first_feature = dataset [:, 0]
print (first_sample)
print (first_feature)

# Making a composed transformation
composed = torchvision.transforms.Compose ([ToTensor (), MulTransform (2)])
dataset = WineDataset (transform = composed)

# Using DataLoader class
# Will implement using batches for you
num_epochs = 20
n_samples = len (dataset)
n_iters = math.ceil (n_samples / 4)
print (n_samples)
print (n_iters)
dl = DataLoader (dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2)

# Iterate over dataset
for epoch in range (num_epochs):
	for i, (inputs, labels) in enumerate (dl):
		if (i + 1) % 10 == 0:
			print (epoch)
			print (i + 1)
di = iter (dl)
data = dataiter.next ()
features, labels = data
print (features)
print (labels)

