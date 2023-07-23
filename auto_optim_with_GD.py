#Using pytorch to automate gradient descent optimization
#Using linear regression

import torch

x = torch.tensor ([1, 2, 3, 4], dtype = torch.float32)
y = torch.tensor ([2, 4, 6, 8], dtype = torch.float32)
# initialize w
w = torch.tensor (0.0, dtype = torch.float32, requires_grad = True)
# Now use torch to calculate the prediction, loss, and gradient
# Forward pass for linear regression
def forward (x):
	return x * w

# loss function for linear regression, mse
def loss (y, y_pred):
	return ((y_pred - y)**2).mean ()

# Now to do an update step for optimization we need to define some other parameters
# learning rate
lr = 0.001
# learning itterations
num_i = 100
# Now for training loop
for epoch in range (num_i):
	# pred forward pass
	y_pred = forward (x)
	# get loss
	l = loss (y, y_pred)
	# get gradient
	l.backward ()
	# Update the weights using the gradient descent formula
	# Move in the opposite direction of the gradient using the learning rate to determine how far to move
	# Use the .grad attribute to get the calculated gradient from one of the variables at the start of the computation chain, in this case w.grad
	# Remember that with the autograd package gradients need to be turned off so that the update calculation is not included in the computation graph for gradients
	with torch.no_grad ():
		w -= (lr * w.grad)
	# Accumulated gradient for weights needs to be zeroed out after each epoch of training
	w.grad.zero_ ()
	print (str ("Epoch " + str (epoch) + " loss: " + str (l) + " prediction: " + str (y_pred)))
print (str ("Final prediction: " + str (forward (x))))

# Every step increases the weights and decreases the loss as GD runs

