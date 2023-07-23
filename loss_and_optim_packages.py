#Using pytorch to automate loss and optimization of gradient descent optimization
#Using linear regression
# Moving towards automation of full pipeline
# 3 main steps to designing pytorch models
# 1 determine size and type of input along with the size and type of the output, determine all of the forward operations of model
# 2 construct the loss and the optimizer
# 3 construct the training loop, compute the gradient and perform the backward pass, update parameters and complete some number of itterations
# Replace the loss and optimization with the torch.nn and torch.optim packages now 

import torch
import torch.nn as nn

# Changing the shapes of the input and output tensors to work with the model object
# Making both input and output 2 dimensional
x = torch.tensor ([[1], [2], [3], [4]], dtype = torch.float32)
y = torch.tensor ([[2], [4], [6], [8]], dtype = torch.float32)
x_test = torch.tensor ([5], dtype = torch.float32)

# Getting the number of samples and number of features
n_samples, n_features = x.shape

# Now use torch to calculate the prediction, loss, and gradient
# Deleting the manual forward and loss functions
# replace the forward function with a torch model object

input_size = n_features
output_size = n_features

# Can also design and implement your own custom model
# More model classes such as this one can be written in order to try different models with very different algorithms and architectures
class LinearRegression (nn.Module):

	def __init__ (self, input_dim, output_dim):
		# Run the initialization method of the parent class
		super (LinearRegression, self).__init__ ()
		# definition of architected layers
		self.lin = nn.linear (input_dim, output_dim)
		
	# Implement forward method
	def forward (self, x):
		return self.lin (x)

model = LinearRegression (input_size, output_size)

# Getting the loss function from nn module for MSE
loss = nn.MSELoss ()

# Now to do an update step for optimization we need to define some other parameters
# learning rate
lr = 0.001
# learning itterations
num_i = 100

# Getting the SGD optimizer from optim module
opt = torch.optim.SGD (model.parameters (), lr = lr)

# Now for training loop
for epoch in range (num_i):
	# pred forward pass
	y_pred = model (x)
	# get loss
	l = loss (y, y_pred)
	# get gradient
	l.backward ()
	# Update the weights using the gradient descent formula
	# Deleting manual update of weights, using the .step function instead
	# This automatically performs the optimization step for SGD
	opt.step ()
	# The weight gradients still need to be zeroed out after each optimization step
	w.grad.zero_ ()
	if (epoch % 10) == 0:
		# Unpacking parameters
		# The weights along with an extra bias tensor, both will be 2d tensors
		# Can be used in prints
		[w, b] = model.parameters ()
		print (str ("Epoch " + str (epoch) + " loss: " + str (l) + " prediction: " + str (y_pred)))
print (str ("Final prediction: " + str (model (x_test).item)))

# Every step increases the weights and decreases the loss as GD runs

# This code implements a full training pipeline
