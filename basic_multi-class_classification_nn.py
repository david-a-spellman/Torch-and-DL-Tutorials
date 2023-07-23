# Implementing a simple classification nn with torch using softmax and CEL

import torch
import torch.nn as nn

input_size = (28 * 28)
hidden_size = 5
number_of_classes = 3

class NeuralNet (nn.Module):

	def __init__ (self, input, hidden, classes):
		super (NeuralNet, self).__init__ ()
		self.l1 = nn.linear (input, hidden)
		self.relu = nn.ReLU ()
		if classes > 2:
			self.l2 = nn.linear (hidden, classes)
		else:
			self.l2 = nn.linear (hidden, 1)

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

