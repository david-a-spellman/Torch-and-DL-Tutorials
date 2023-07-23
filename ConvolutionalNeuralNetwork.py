# Example using classification of 10 types of objects in images
# With convolutional neural network
# Using torch

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
#device = "cpu"

input_size = (28 * 28)
hidden_size = 5
number_of_classes = 10
epochs = 10
batch_size = 100
lr = 0.0001

transform = transforms.Compose ([transforms.ToTensor (), transforms.Normalize ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10 (root = "./data", train = True,
	transform = transform, download = True)

test_dataset = torchvision.datasets.CIFAR10 (root = "./data", train = False,
	transform = transform, download = True)

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

class ConvNet (nn.Module):

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

# loop
steps = len (train_loader)
for epoch in range (epochs):
	for i, (images, labels) in enumerate (train_loader):
		images = images.reshape (-1, (28 * 28)).to (device)
		labels = labels.to (device)
		outputs = model (images)
		loss = criterion (outputs, labels)
		opt.zero_grad ()
		loss.backward ()
		opt.step ()
		if ((i + 1) % 50) == 0:
			print (loss.item ())

# Testing
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
	acc = (100.0 * (tp / samples))
	print (str ("Final testing accuracy = " + str (acc)))