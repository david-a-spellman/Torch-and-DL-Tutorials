# How to do LR in torch
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# step 0, data preprocessing
# step 1 implement the model and the forward pass
# step 2 implement the loss and optimization
# step 3 implement the gradient calculation and the backward step

# Using breast cancer dataset
bc = datasets.load_breast_cancer ()
x, y = bc.data, bc.target
n_samples, n_features = x.shape
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 1234)

# Normalizing data for use of logistic regression
sc = StandardScaler ()
# After creating the standard scalar object for scaling the dataset, you have to first call the .fit_transform method to both fit the object to the datset and then transform the dataset
x_train = sc.fit_transform (x_train)
# Second time around only .transform method needs to be called since the object is already fitted to the dataset
x_test = sc.transform (x_test)

x_train = torch.from_numpy (x_train.astype (np.float32))
x_test = torch.from_numpy (x_test.astype (np.float32))
y_train = torch.from_numpy (y_train.astype (np.float32))
y_test = torch.from_numpy (y_test.astype (np.float32))
y_train = y_train.view (y_train.shape [0], 1)
y_test = y_test.view (y_test.shape [0], 1)

# Model class code for LR
class LogisticRegression (nn.Module):

	def __init__ (self, n_features):
		super (LogisticRegression, self).__init__ ()
		self.linear = nn.Linear (n_features, 1)

	def forward (self, x):
		y_pred = torch.sigmoid (self.linear (x))
		return y_pred

n_samples, n_features = x_train.shape
model = LogisticRegression (n_features)



# Loss and optimizer
# Using BCE loss for logistic regression

criterion = nn.BCELoss ()
opt = torch.optim.SGD (model.parameters (), lr = 0.0005)

# Now for producing training loop
itterations = 50000

for epoch in range (itterations):
	# Forward pass
	y_pred = model (x_train)
	#print (str ("EPOCH " + str (epoch) + " preds: " + str (y_pred)))
	l = criterion (y_pred, y_train)
	if (epoch % 1000) == 0:
		print (str ("EPOCH " + str (epoch) + " BCE LOSS: " + str (l.item ())))
	# Optimization step
	l.backward ()
	opt.step ()
	opt.zero_grad ()

#end of training loop
with torch.no_grad ():
	#print (str ("FINAL PREDICTION: " + str (model (x_train).detach ())))
	predicted = model (x_test)
	predicted_class = predicted.round ()
	acc = predicted_class.eq (y_test).sum ()
	n_samples = y_test.shape [0]
	print (acc)
	print (n_samples)
	acc = (acc / n_samples)
	print (acc)

#print (type (acc))
#acc = acc [0].item ()
print (f'FINAL ACCURACY = {acc:.4f}')

# plot predictions
