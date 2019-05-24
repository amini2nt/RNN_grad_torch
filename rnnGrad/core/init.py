import torch
from torch import nn


def normal(size, mean=0, std=0.01):
	"""Returns a torch matrix with entries sampled from a Gaussian.

	Args:
		size: tuple, size of the torch matrix.
		mean: float, mean of the Gaussian.
		std: float, standard deviation of the Gaussian.
	"""
	mat = torch.zeros(size)
	nn.init.normal_(mat, mean=mean, std=std)
	return mat

def xavier_uniform(size):
	"""Returns a torch matrix with entries sampled from Xavier uniform initialization.

	Args:
		size: tuple, size of the torch matrix.
	"""
	mat = torch.zeros(size)
	nn.init.xavier_uniform_(mat)
	return mat
	
def xavier_normal(size):
	"""Returns a torch matrix with entries sampled from Xavier normal initialization.

	Args:
		size: tuple, size of the torch matrix.
	"""
	mat = torch.zeros(size)
	nn.init.xavier_normal_(mat)
	return mat


def orthogonal(size):
	"""Returns a torch matrix with entries sampled from orthogonal initialization.

	Args:
		size: tuple, size of the torch matrix.
	"""
	mat = torch.zeros(size)
	nn.init.orthogonal_(mat)
	return mat


def uniform(size, a=0.0, b=1.0):
	"""Returns a torch matrix with entries sampled from uniform distribution.

	Args:
		size: tuple, size of the torch matrix.
		a: float, low value of the uniform distribution.
		b: float, high value of the uniform distribution.
	"""
	mat = torch.zeros(size)
	nn.init.uniform_(mat, a=a, b=b)
	return mat

def zeros(size):
	"""Returns a torch matrix filled with zeros.

	Args:
		size: tuple, size of the torch matrix.
	"""
	return torch.zeros(size)

def constant(size, value=1.0):
	"""Returns a torch matrix filled with a constant value.

	Args:
		size: tuple, size of the torch matrix.
		value: float, constant value.
	"""
	return torch.ones(size)*value