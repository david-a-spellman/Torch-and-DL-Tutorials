# Activation functions are important for deep learning in order to get the linear regression layers to actually function like neurons
# The activation functions add in a non-linear component that allows the network to have a more complex behavior
# The binary activation function
# the sigmoid activation function 1 / (1 - e**(-x))
# The hyperbolic tangent function, a scaled version of the sigmoid function
# A good choice in hidden layers
# ReLU function
# Basically just turns all negative values into 0s
# Most popular activation function
# Leeky ReLU function improves upon ReLU by giving a slight gradient to the negative values
# In a way simulates the neuron better because of this
# Takes negative values and makes them a lot less negative, but does not pull them all straight to 0
# When weights will not update during training it is best to use Leeky ReLU instead of normal ReLU
# Softmax will squash all values down to a value between 0 and 1
# Almost allways a good choice in the last layer of a multi-class classification problem

# Using torch
# All of these activation functions are available in the torch.nn module
# Some others are available in the torch.nn.functional module

import torch
import torch.nn as nn
import torch.nn.functional as f

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

