# Example using classic number classification of hand written digits dataset
# Using torch

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys

# Writing data to tensorboard
writer = SummaryWriter ("E:\\runs\\MNIST_FFNN")

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
# device = "cpu"

input_size = (28 * 28)
hidden_size = 5
number_of_classes = 10
epochs = 10
batch_size = 100
lr = 0.0001

train_dataset = torchvision.datasets.MNIST (root = "./data", train = True,
	transform = transforms.ToTensor (), download = False)

test_dataset = torchvision.datasets.MNIST (root = "./data", train = False,
	transform = transforms.ToTensor (), download = False)

train_loader = torch.utils.data.DataLoader (dataset = train_dataset, batch_size = batch_size,
	shuffle = True)

test_loader = torch.utils.data.DataLoader (dataset = test_dataset, batch_size = batch_size,
	shuffle = False)

examples = iter (train_loader)
samples, labels = next (examples)

for i in range (6):
	plt.subplot (2, 3, (i + 1))
	plt.imshow (samples [i] [0], cmap = "gray")
#plt.show ()

# Write images to tensorboard
im_grid = torchvision.utils.make_grid (samples)
writer.add_image ("MNIST_data", im_grid)
#writer.close ()
#sys.exit ()

class NeuralNet (nn.Module):

	def __init__ (self, input, hidden, classes):
		super (NeuralNet, self).__init__ ()
		self.l1 = nn.Linear (input, hidden)
		self.relu = nn.ReLU ()
		if classes > 2:
			self.l2 = nn.Linear (hidden, classes)
		else:
			self.l2 = nn.Linear (hidden, 1)

	def forward (self, x):
		out = self.l1 (x)
		out = self.relu (out)
		out = self.l2 (out)
		return out

model = NeuralNet (input = input_size, hidden = hidden_size, classes = number_of_classes)
criterion = None
if number_of_classes > 2:
	criterion = nn.CrossEntropyLoss ()
else:
	criterion = nn.BCELoss ()
opt = torch.optim.Adam (model.parameters (), lr = lr)

# Graph model performance with tensorboard
writer.add_graph (model, samples.reshape (-1, (28 * 28)))
#writer.close ()
#sys.exit ()

# loop
steps = len (train_loader)
model = model.to (device)
rl = 0.0
rc = 0
every_n_steps = 50
for epoch in range (epochs):
	for i, (images, labels) in enumerate (train_loader):
		images = images.reshape (-1, (28 * 28)).to (device)
		labels = labels.to (device)
		outputs = model (images)
		loss = criterion (outputs, labels)
		opt.zero_grad ()
		loss.backward ()
		opt.step ()
		rl += loss.item ()
		_, predicted = torch.max (outputs.data, 1)
		rc += (labels == predicted).sum ().item ()
		if ((i + 1) % every_n_steps) == 0:
			print (str ("LOSS \t" + str (rl / every_n_steps)))
			print (str ("ACCURACY \t" + str ((rc / (every_n_steps * batch_size)) * 100)))
			writer.add_scalar ("training loss for every_n_steps steps", (rl / every_n_steps), ((epoch * steps) + i))
			writer.add_scalar ("training accuracy for every_n_steps steps", (rc / every_n_steps), ((epoch * steps) + i))
			rl = 0
			rc = 0
writer.close ()
#sys.exit ()
# Testing
# Adding PR curve
labs = []
preds = []
with torch.no_grad ():
	tp = 0
	samples = 0
	for images, labels in test_loader:
		images = images.reshape (-1, (28 * 28)).to (device)
		labels = labels.to (device)
		outputs = model (images)
		_, predictions = torch.max (outputs, 1)
		samples += labels.shape [0]
		tp = (predictions == labels).sum ().item ()
		cp = [F.softmax (output, dim = 0) for output in outputs]
		preds.append (cp)
		labs.append (predictions)
	preds = torch.cat ([torch.stack (batch) for batch in preds])
	labs = torch.cat (labs)
	acc = (100.0 * (tp / samples))
	print (str ("Final testing accuracy = " + str (acc)))
	classes = range (10)
	for i in classes:
		labs_i = labs == i
		preds_i = preds [:, i]
		writer.add_pr_curve (str (i), labs_i, preds_i, global_step = 0)
	writer.close ()