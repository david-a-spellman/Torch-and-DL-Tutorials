# How to do a fully automated learning pipeline in torch
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# step 0, data preprocessing
# step 1 implement the model and the forward pass
# step 2 implement the loss and optimization
# step 3 implement the gradient calculation and the backward step

xn, yn = datasets.make_regression (n_samples = 100, n_features = 1, noise = 20, random_state = 1)

x = torch.from_numpy (xn.astype (np.float32))
y = torch.from_numpy (yn.astype (np.float32))
y = y.view (y.shape [0], 1)

n_samples, n_features = x.shape

# For this simple example that deals with all of the data
# Now for designing the model

model = nn.Linear (n_features, n_features)

# Loss and optimizer

criterion = nn.MSELoss ()
opt = torch.optim.SGD (model.parameters (), lr = 0.0005)

# Now for producing training loop
itterations = 50000

for epoch in range (itterations):
	# Forward pass
	y_pred = model (x)
	#print (str ("EPOCH " + str (epoch) + " preds: " + str (y_pred)))
	l = criterion (y_pred, y)
	if (epoch % 1000) == 0:
		print (str ("EPOCH " + str (epoch) + " LOSS: " + str (l.item ())))
	# Optimization step
	l.backward ()
	opt.step ()
	opt.zero_grad ()

#end of training loop
print (str ("FINAL PREDICTION: " + str (model (x).detach ())))
predicted = model (x).detach ().numpy ()

# plot predictions
plt.plot (xn, yn, 'ro')
plt.plot (xn, predicted, 'b')
plt.show ()