import torch
import torch.nn as nn

"""
Most complicated part of the deep transformer
The self attention mechanism that allows the model to optimize what information to weight when examining context
This way a large transformer can have affectively limitless access to cross textual context by making larger and larger transformer models
"""

class SelfAttention (nn.module):

	def __init__ (self, emb_size, heads):
		super (SelfAttention, self).__init__ ()
		self.emb_size = emb_size
		self.heads = heads
		self.head_dim = emb_size // heads
		assert (self.head_dim * heads == emb_size, "ISSUE !!! The embedding space needs to be divided across the heads!")
		# Now for defining the linear fully-connected layers that the queries, keys, and values will get sent through
		self.values = nn.Linear (self.head_dim, self.head_dim, bias = False)
		self.keys = nn.Linear (self.head_dim, self.head_dim, bias = False)
		self.queries = nn.Linear (self.head_dim, self.head_dim, bias = False)
		# Fully connected output layer, where the input size is all heads and their dimensions, and the output is simply the size of an embedding
		self.fco = nn.Linear (self.head_dim * heads, emb_size)

	def forward (self, values, keys, query, mask):
		n = query.shape [0]
		vl, kl, ql = values.shape [1], keys.shape [1], query.shape [1]
		# Now the important part of splitting the embedding space accross the different attention heads
		values = values.reshape (n, vl, self.heads, self.head_dim)
		keys = keys.reshape (n, kl, self.heads, self.head_dim)
		queries = query.reshape (n, ql, self.heads, self.head_dim)
		# Performing the matrix multiplication with the torch einsum () function, can deal with extra dimentions
		# The resulting energy shape will be (n, heads, ql, kl)
		# This call simplies the tensor operations so you do not need to perform tensor flattening and other operations to get the dimentions right
		nrg = torch.einsum ("nqhd,nkhd->nhqk", [queries, keys])
		# Now logic for if there is a mask depending on whether this is the attention mechanism for an encoder or a decoder
		if mask != None:
			# The mask will be a triangular matrix and will set all values to a very negative number
			nrg = nrg.masked_fill (mask = 0, float ("-1e20"))
		# Now the attention will be calculated by running the energy through a softmax
		attention = torch.softmax (nrg / (self.emb_size ** (1 / 2)), dim = 3)
		# Shape of the attention is n, heads, query length, key length
		# Shape of the values is n, value length, heads, head dimensions
		# shape of the queries n, query length, heads, head dimensions
		# Using the torch einsum function again to make the notation and code easier
		out = torch.einsum ("nhql,nlhd->nqhd", [attention, values]).reshape ()
			n, ql, self.heads * self.head_dim
		)
		# Now the output is n, query length, heads, head dimensions, and the last 2 dimensions are flattened
		# Last operation is to now send the output through the fully connected output layer
		# simply maps from embedding size input to embedding size output
		out = self.fco (out)
		return out

class TransformerBlock (nn.Module):

	def __init__ (self, emb_size, heads, drop_out, f_expansion):
		super (TransformerBlock, self).__init__ ()
		self.attention = SelfAttention (emb_size, heads)
		"""
		Layer norm is similar to batch norm but instead of normalizing across all examples in a batch, it normalizes for each sample
		Batch norm is usually used for mini-batch stoichastic gradient descent in order to significantly reduce exspensive updates to the mini-batch mean and std
		Batch normalization achieves this through subtracting the batch mean and dividing by the batch std in order to obtain a unit gaussian for each batch
		This batch normalization is applied before being passed through a hidden layer activation
		Each time the mean and std is calculated for the particular batch before normalization
		However, for some features the fluctuation in the distributions across the batches may be important for learning those particular features, so batch norm also 
		adds 2 additional learnable parameters gamma and beta, gamma being the scaling value for a features distribution accross batches, and beta being an offset for that distribution
		This way the learned gamma and beta parameters allows the back-propigation to derive the actual distributions that the features follow from the pre-activation 
		normalized pre-activation unit gaussians
		In order to get the original distribution you take the values from the unit gaussian and multiply them by gamma, and then add beta as the offset
		WARNING !!! serious limitation of batch norm is using a small batch size!
		will cause mean and std of the mini-batches to ill-represent the data, and cause the learning efficiency to deteriorate
		More importantly for this example, batch norm does not work well for sequence models, though it can work for CV
		This is because with sequence models inputs can be of variable length
		This is why layer norm is far preferred here
		Layer normalization differs in that all neurons and input features in a given layer are normalized to have the same distribution, instead of normalizing to 
		have a different distribution for each input feature
		Layer normalization calculates the mean and varriance for the output of the layer before, instead of calculating it for each individual feature
		Instead of calculating the mean and std of the batches, the mean and std of the tensor dimensions is used, or the mean and std accross the features being input
		into the layer
		layer norm still uses the gamma and beta parameters that are optimized, but these parameters are for all input features, to form a single distribution for all inputs
		A simpler explaination is that layer norm normalizes over the current input example, while batch norm normalizes over the batch of examples, producing a seperate
		normalized distribution for each feature
		So since the transformer is a sequence to sequence focused architecture, this code will use 2 layer normalizations
		All layer norm takes as parameter is the size of the input feature space, or in this case the embedding space size
		"""
		self.n1 = nn.LayerNormalization (emb_size)
		self.n2 = nn.LayerNorm (emb_size)
		# Now for the feed forward layers
		# The expansion relu and then reduction back to the embedding space size performs some extra latent computations
		self.ff = nn.Sequential (
			nn.Linear (emb_size, (f_expansion * emb_size)),
			nn.ReLU (),
			nn.Linear ((f_expansion * emb_size), emb_size)
		)
		# Drop out
		self.drop_out = nn.Dropout (drop_out)

	# Forward method
	def forward (value, key, query, mask):
		