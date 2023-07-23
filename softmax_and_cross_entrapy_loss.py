# How to use softmax and CEL function in pytorch and numpy
# Cross entrapy loss in torch, the usual learning criterion used in NN
# Softmax is the exponential for the current class over the sum of the exponentials for all of the classes
# Will squash each probability to be between 0 and 1
# Summing up the probabilities obtained by applying the softmax to each class probability will allways be 1
# Helps make the decision process for classification easier than usual thresholding 

import torch
import torch.nn as nn
import numpy as np

def softmax (x):
	return (np.exp (x) / np.sum (np.exp (x), axis = 0))

x = np.array ([2.0, 1.0, 0.1])
outputs = softmax (x)
print (x)
print (outputs)

# Softmax with torch
x = torch.tensor ([2.0, 1.0, 0.1])
outputs = torch.softmax (x, dim = 0)
print (x)
print (outputs)

# In deep learning often the softmax is used with cross entrapy loss
# The larger the CEL the more the predicted probability diverges from the ground truth
# Need to decrease the CEL during optimization
# The higher the probability of the prediction for the ground truth the lower the CEL will be
# example with CEL used

# CEL function
def cross_entropy (truth, pred):
	l = np.sum (truth * np.log (pred))
	return (l / y.shape [0])

# Cross entropy loss with numpy
y = np.array ([1, 0, 0])
y_pred_good = np.array ([0.7, 0.2, 0.1])
y_pred_bad = np.array ([0.1, 0.3, 0.6])
l1 = cross_entropy (y, y_pred_good)
l2 = cross_entropy (y, y_pred_bad)
print (l1)
print (l2)

# CEL with torch
loss = nn.CrossEntropyLoss ()
# This CrossEntropyLoss class already implements the softmax, so the softmax layer should not be implemented when using this class.
# GT is one hot encoded
# Have 3 classes but only one sample
y = torch.tensor ([0])
y_pred_good = torch.tensor ([[4.0, 1.4, 0.3]])
y_pred_bad = torch.tensor ([[0.9, 1.8, 5.8]])
l1 = loss (y_pred_good, y)
l2 = loss (y_pred_bad, y)
print (l1.item ())
print (l2.item ())

# How to get predictions with softmax
_, preds1 = torch.max (y_pred_good, 1)
_, preds2 = torch.max (y_pred_bad, 1)
print (preds1)
print (preds2)

# With 3 samples

y = torch.tensor ([0, 2, 1])
y_pred_good = torch.tensor ([[4.0, 1.4, 0.3], [0.2, 0.9, 3.2], [0.6, 2.7, 0.3]])
y_pred_bad = torch.tensor ([[0.9, 1.8, 5.8], [4.0, 1.1, 0.7], [3.3, 0.4, 3.9]])
l1 = loss (y_pred_good, y)
l2 = loss (y_pred_bad, y)
print (l1)
print (l2)

# How to get predictions with softmax
_, preds1 = torch.max (y_pred_good, 1)
_, preds2 = torch.max (y_pred_bad, 1)
print (preds1)
print (preds2)