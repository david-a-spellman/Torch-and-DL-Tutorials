# Training in batches
# 1 epoch means a full forward and backward pass on all training data
# The batch size is the number of training examples in a single forward and backward pass
# The number of itterations are the number of forward and backward passes where the batch size is used
# If there are 100 samples in the training data with a batch size of 20, that means each epoch will complete 5 itterations to cover all training data
# 50 epochs would then mean 250 itterations in total for training

# Constructing a training loop with batches
for epoch in range (1000):
	# loop over batches
	for i in range (total_batches):
		x_batch, y_batch = ...
# Use the datasets and dataloader to get csv file
# This is where making a class for your dataset is done
# Instead of manually doing all dataset operations
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

Class WineDataset (Dataset):

	def __init__ (self):
		xy = np.loadtxt (".\wine\wine.csv", delimiter = ",", dtype = np.float32, skiprows = 1)
		self.x = torch.from_numpy (xy [:, 1:])
		self.y = torch.from_numpy (xy [:, [0]])
		self.n_samples = xy.shape [0]

	def __getitem__ (self, index):
		return self.x [index], self.y [index]
	def __len__ (self):
		return self.n_samples

# Finished implementing Wine Dataset
dataset = WineDataset ()
print (dataset)

first_sample = dataset [0]
first_feature = dataset [:, 0]
print (first_sample)
print (first_feature)

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

