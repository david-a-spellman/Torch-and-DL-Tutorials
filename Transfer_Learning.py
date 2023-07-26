# Example of using transfer learning with a CNN to get better results with less data and less training time/resources
# Going to simply fine tune the last few layers of an already trained CNN
# Example using the resnet 18 CNN
# Can classify objects into 1000 categories, and has 18 layers
# Going to just train the final layer
# This will fine-tune the network for a specific task
# Using the CIFAR10 dataset for those 10 classes

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
#device = "cpu"

input_size = (32 * 32)
filter = 5
pool = 2
number_of_classes = 10
epochs = 60
batch_size = 4
lr = 0.0001

# output image size formula for first convolution
# (input_dim - filter_size + padding) / stride_length + 1
# formula for affect on image shape for first pooling operation
# first_conv_output_dim / pool_dim

transform = transforms.Compose ([transforms.ToTensor (), transforms.Normalize ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10 (root = "./data", train = True,
	transform = transform, download = True)

test_dataset = datasets.CIFAR10 (root = "./data", train = False,
	transform = transform, download = True)

train_loader = torch.utils.data.DataLoader (dataset = train_dataset, batch_size = batch_size,
	shuffle = True)

test_loader = torch.utils.data.DataLoader (dataset = test_dataset, batch_size = batch_size,
	shuffle = False)

examples = iter (train_loader)
samples, labels = next (examples)

for i in range (4):
	plt.subplot (2, 2, (i + 1))
	plt.imshow (samples [i] [0])
#plt.show ()

class ConvNet (nn.Module):

	def __init__ (self, conv, pool, classes, batch_size):
		super (ConvNet, self).__init__ ()
		self.c1 = nn.Conv2d (3, 6, conv)
		self.pool = nn.MaxPool2d (pool, pool)
		self.c2 = nn.Conv2d (6, 16, conv)
		self.l1 = nn.Linear ((16 * 5 * 5), 180)
		self.l2 = nn.Linear (180, 60)
		self.l3 = nn.Linear (60, classes)
		self.relu = nn.ReLU ()

	def forward (self, x):
		out = self.pool (self.relu (self.c1 (x)))
		out = self.pool (self.relu (self.c2 (out)))
		out = out.view (-1, 16 * 5 * 5)
		out = self.relu (self.l1 (out))
		out = self.relu (self.l2 (out))
		out = self.l3 (out)
		return out

#model = ConvNet (conv = filter, pool = pool, classes = number_of_classes, batch_size = batch_size)
# Using transfer learning instead
model = models.resnet18 (pretrained = True)
"""
# There is also a second option for transfer learning that involves freezing all model weights save the final or some of the final FC layers
for par in model.parameters ():
	# Set the requires_grad flag to False for all carried over weights to prevent these from updating further
	par.requires_grad = False
"""
# Getting number of input features for the final layer that will be the fine-tuning layer
nf = model.fc.in_features
# Re-initialize to get new fine-tuned FC layer
model.fc = nn.Linear (nf, number_of_classes)
criterion = None
if number_of_classes > 2:
	criterion = nn.CrossEntropyLoss ()
else:
	criterion = nn.BCELoss ()
opt = torch.optim.Adam (model.parameters (), lr = lr)
s_lr_s = lr_scheduler.StepLR (opt, step_size = 10, gamma = 0.25)

# loop
steps = len (train_loader)
model = model.to (device)
for epoch in range (epochs):
	print (str ("Starting epoch " + str (epoch)))
	for i, (images, labels) in enumerate (train_loader):
		images = images.to (device)
		labels = labels.to (device)
		outputs = model (images)
		loss = criterion (outputs, labels)
		opt.zero_grad ()
		loss.backward ()
		opt.step ()
		s_lr_s.step ()
		# Will print out a single training loss for batch per epoch, since CIFAR10 has 50k training images, the batch size is 4, and this prints out one loss per 10k steps
		# 10k training steps in this instance covers 40k images
		if ((i + 1) % 10000) == 0:
			print (loss.item ())

# Testing
with torch.no_grad ():
	tp = 0
	samples = 0
	# Strange pythonic syntax for more efficiently initializing a list of all 0s to a length of 10 list
	n_tc = [0 for i in range (10)]
	n_cs = [0 for i in range (10)]
	for images, labels in test_loader:
		images = images.to (device)
		labels = labels.to (device)
		outputs = model (images)
		_, predictions = torch.max (outputs, 1)
		samples += labels.shape [0]
		tp += (predictions == labels).sum ().item ()

		for i in range (batch_size):
			label = labels [i]
			pred = predictions [i]
			if (label == pred):
				n_tc [label] += 1
			n_cs [label] += 1
	acc = (100.0 * (tp / samples))
	c_acc = [0 for i in range (10)]
	for i in range (len (n_cs)):
		c_acc [i] = (100.0 * (n_tc [i] / n_cs [i]))
	print (str ("Final testing accuracy = " + str (acc)))
	i = 0
	for c in c_acc:
		i += 1
		print (str ("Final class " + str (i) + " testing accuracy = " + str (c)))

# save model
path = "C:\\Projects\\Torch-and-DL-Tutorials\\pretrained\\"
if not os.path.isdir (path):
	os.mkdir (path)
torch.save(model.state_dict(), path)
print ("DONE !!!")