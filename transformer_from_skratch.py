import torch
import torch.nn as nn

"""
Fully implements the transformer with both encoder and decoder
Has detailed comments in order to explain the architecture and what the Pytorch code is doing
This file can be imported in to serve as the model implementation of a transformer for running experiments or training deployable models with Pytorch
Most complicated part of the deep transformer is the self attention mechanism
The self attention mechanism that allows the model to optimize what information to weight when examining context
This way a large transformer can have affectively limitless access to cross textual context by making larger and larger transformer models
Is key for solving sequential deep learning problems such as NLP
Will be adding examples on how to train and deploy this model type
Will need some text examples and a vocabulary
The encoder and decoder classes can also be used separately
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
		attention = self.attention (value, key, query, mask)
		# Perform dropout on the attention output concatinated with the query
		# The concatination with the query is a skip connection
		x = self.drop_out (self.n1 (attention + query))
		forward = self.ff (x)
		# Second drop-out and skip connection with the output from the first layer normalization
		out = self.drop_out (self.n2 (forward + x))
		# This completes the attention block
		return out

# Now take the attention block to create encoder and decoder classes

class TransformerEncoder (nn.Module):

	def __init__ (self,
		vocab_size,
		emb_size,
		layers,
		heads,
		device,
		f_expansion,
		drop_out,
		max_length):
		super (TransformerEncoder, self).__init__ ()
		self.emb_size = emb_size
		# The vocabulary is mapped onto the allowable embedding space size
		# Pytorch has the nn.Embedding module that can handle mapping sequential text data of a certain vocab size onto an embedding space of certain size
		self.word_emb = nn.Embedding (vocab_size, emb_size)
		self.device = device
		# Now for setting up the module to map position embeddings for tokens
		self.position_emb = nn.Embedding (max_length, emb_size)
		self.layers = nn.ModuleList ([
			TransformerBlock (emb_size, heads, drop_out = drop_out,
				f_expansion = f_expansion)])
		self.drop_out = nn.Dropout (drop_out)

	def forward (self, x, mask):
		n, seq_len = x.shape
		# Get the tensor of the token positions
		positions = torch.arange (0, seq_len).expand (n, seq_len).to (self.device)
		# Now do concatination of the word embedding with the positions embedding for that word
		# Then perform dropout on this concatinated embedding to get a result
		# This allows the model to learn the patterens in which individual words appear in the type of text the model is trained on
		out = self.drop_out (self.word_emb (x) + self.position_emb (positions))
		# Now loop through and run the attention layers
		for layer in self.layers:
			# Since this is an encoder the value, key, and query are all the same embedding output with drop_out applied to it
			# One layer passes its output to the next layer as that attention layers value, key, and query
			out = layer (out, out, out, mask)
		return out

# Now for the transformer decoder network
# Must have the decoder block module first

class DecoderTransformerBlock (nn.Module):

	def __init__ (self,
		emb_size,
		heads,
		f_expansion,
		drop_out,
		device):
		super (DecoderTransformerBlock, self).__init__ ()
		self.attention = SelfAttention (emb_size, heads)
		# Decoder block adds one extra layer normalization layer
		self.n = nn.LayerNorm (emb_size)
		# Also contains a regular transformer block
		self.tb = TransformerBlock (emb_size, heads, drop_out, f_expansion)
		# Decoder also has an added drop-out layer
		self.drop_out = nn.Dropout (drop_out)

	# Difference here is that there is both a source and a target mask, and no query used
	def forward (self, x, value, key, s_mask, t_mask):
		# The target mask is mandatory for padding variable length inputs
		# The source mask is optional for preventing unnecessary computations for values that are padded
		# Model will not work without the target mask, but without a source mask the model will just run a lot less efficiently
		# Just like with regular transformer block you get the attention first
		# The mask used here is the target mask for ensuring the correct padding is used for calculating the self-attention
		attention = self.attention (x, x, x, t_mask)
		# The query is here derived using the drop-out of the layer normalization of the attention with the skip-connection of the input
		query = self.drop_out (self.n (attention + x))
		# Now the regular transformer block is run last
		# This is where the optional source mask is used
		out = self.tb (value, key, query, s_mask)
		return out

class TransformerDecoder (nn.Module):

	def __init__ (self,
		vocab_size,
		emb_size,
		layers,
		heads,
		device,
		f_expansion,
		drop_out,
		max_length):
		super (TransformerDecoder, self).__init__ ()
		self.emb_size = emb_size
		self.word_emb = nn.Embedding (vocab_size, emb_size)
		self.device = device
		# Now for setting up the module to map position embeddings for tokens
		self.position_emb = nn.Embedding (max_length, emb_size)
		self.layers = nn.ModuleList ([
			DecoderTransformerBlock (emb_size, heads, drop_out = drop_out,
				f_expansion = f_expansion, device = device)
			for _ in range (layers)])
		# The decoder has an added fully-connected output layer
		# Will take an embedding tensor as input and produce a vocab token as output
		self.fc = nn.Linear (emb_size, vocab_size)
		self.drop_out = nn.Dropout (drop_out)

	# Takes an input from the encoder
	def forward (self, x, encoder_out, s_mask, t_mask):
		n, seq_len = x.shape
		positions = torch.arange (0, seq_len).expand (n, seq_len).to (self.device)
		x = self.drop_out (self.word_emb (x) + self.position_emb (positions))
		for layer in self.layers:
			# Since this is a decoder the key and query are the output from the encoder
			# It also takes as input the source and target masks
			# x becomes the output of a layer, and becomes the value input to the next layer
			# Maybe the most mistifying detail of cutting edge deep learning
			x = layer (x, encoder_out, encoder_out, s_mask, t_mask)
		# The actual final output is the fully-connected output result of the x output of the final decoder transformer layer
		# This embedding that is output is the decoder's prediction of the next token that should come next based on the text input to the decoder
		# And based on what type of text and vocabulary the decoder has been trained on
		out = self.fc (x)
		return out

# Now putting all these classes together into a full transformer

class Transformer (nn.Module):

	def __init__ (self,
		source_vocab_s,
		target_vocab_s,
		source_pad_idx,
		target_pad_idx,
		embed_s = 256,
		layers = 3,
		f_expansion = 4,
		heads = 6,
		drop_out = 0,
		device = "cuda",
		max_length = 128):
		super (Transformer, self).__init__ ()
		self.encoder = TransformerEncoder (
			source_vocab_s,
			embed_s,
			layers,
			heads,
			device,
			f_expansion,
			drop_out,
			max_length)
		self.decoder = TransformerDecoder (
			target_vocab_s,
			embed_s,
			layers,
			heads,
			device,
			f_expansion,
			drop_out,
			max_length)
		# Padding indicies for calculating the masks
		self.spi = source_pad_idx
		self.tpi = target_pad_idx
		self.device = device

	# Method for calculating the source mask
	def source_mask (self, source):
		# The two unsqueeze calls here are to reshape the tensor to dimensions n, 1, 1, source length
		mask = (source != self.spi).unsqueeze (1).unsqueeze (2)
		return mask.to (self.device)

	# Method for calculating the source mask
	def target_mask (self, target):
		n, target_len = target.shape
		# Produces a lower triangular matrix for the target mask
		# Is expanded in order to produce a triangular matrix mask for each example
		mask = torch.tril (torch.ones (target_len, target_len)).expand (n, 1, target_len, target_len)
		return mask.to (self.device)

	def forward (self, source, target):
		source_mask = self.source_mask (source)
		target_mask = self.target_mask (target)
		# The encoder part simply encodes the source tokens using attention mechanism
		encoded_source = self.encoder (source, source_mask)
		# The decoder part takes the target and the encoded source in order to predict the next target
		out = self.decoder (target, encoded_source, source_mask, target_mask)
		return out

# EOF