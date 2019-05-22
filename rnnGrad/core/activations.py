import numpy as np
from scipy.special import expit
from rnnGrad.core.model import Layer
import torch

class torch_sigmoid(Layer):
	"""Implementation of sigmoid activation function."""

	def __init__(self):
		"""Initializes a sigmoid layer."""
		super().__init__()
		self._output = {}

	def forward(self, input, time=1):
		"""Forward prop through the sigmoid layer.
		
		Args:
			input: numpy array, input to the sigmoid layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = torch.sigmoid(input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the sigmoid layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		self._grads[time] = self._output[time] * (1 - self._output[time])
		self._grads[time] = torch.t(self._grads[time]) * in_grad
		return self._grads[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._output = {}
		self._grads = {}
		self._updates = {}	


class sigmoid(Layer):
	"""Implementation of sigmoid activation function."""

	def __init__(self):
		"""Initializes a sigmoid layer."""
		super().__init__()
		self._output = {}

	def forward(self, input, time=1):
		"""Forward prop through the sigmoid layer.
		
		Args:
			input: numpy array, input to the sigmoid layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = expit(input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the sigmoid layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		self._grads[time] = self._output[time] * (1 - self._output[time])
		self._grads[time] = self._grads[time].T * in_grad
		return self._grads[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._output = {}
		self._grads = {}
		self._updates = {}	


class torch_tanh(Layer):
	"""Implementation of tanh activation function."""

	def __init__(self):
		"""Initializes a tanh layer."""
		super().__init__()
		self._output = {}
		
	def forward(self, input, time=1):
		"""Forward prop through the tanh layer.
		
		Args:
			input: numpy array, input to the tanh layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = torch.tanh(input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the tanh layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		self._grads[time] = 1 - (self._output[time]**2)
		self._grads[time] = torch.t(self._grads[time]) * in_grad
		return self._grads[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._output = {}
		self._grads = {}
		self._updates = {}

class tanh(Layer):
	"""Implementation of tanh activation function."""

	def __init__(self):
		"""Initializes a tanh layer."""
		super().__init__()
		self._output = {}
		
	def forward(self, input, time=1):
		"""Forward prop through the tanh layer.
		
		Args:
			input: numpy array, input to the tanh layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = np.tanh(input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the tanh layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		self._grads[time] = 1 - (self._output[time]**2)
		self._grads[time] = self._grads[time].T * in_grad
		return self._grads[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._output = {}
		self._grads = {}
		self._updates = {}

class torch_relu(Layer):
	"""Implementation of relu activation function."""

	def __init__(self):
		"""Initializes a relu layer."""
		super().__init__()
		self._output = {}
		
	def forward(self, input, time=1):
		"""Forward prop through the relu layer.
		
		Args:
			input: numpy array, input to the relu layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = torch.max(torch.Tensor([0]), input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the relu layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""

		self._grads[time] = torch.gt(self._output[time],0).float() ## not sure why the max above shouldnt be used or x[x > 0]
		self._grads[time] = torch.t(self._grads[time]) * in_grad
		return self._grads[time]



class relu(Layer):
	"""Implementation of relu activation function."""

	def __init__(self):
		"""Initializes a relu layer."""
		super().__init__()
		self._output = {}
		
	def forward(self, input, time=1):
		"""Forward prop through the relu layer.
		
		Args:
			input: numpy array, input to the relu layer
			time: int, time step

		returns:
			output, a numpy array of same size as input.
		"""
		self._output[time] = np.maximum(0, input)
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backward prop through the relu layer.

		Args:
			in_grad: numpy array, incoming gradient.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		self._grads[time] = np.greater(self._output[time], 0).astype(float)
		self._grads[time] = self._grads[time].T * in_grad
		return self._grads[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._output = {}
		self._grads = {}
		self._updates = {}		


def get_activation(activation):
	"""Returns the activation object.

	Args:
		activation: str, name of the activation function.

	return:
		object of the corresponding activation function.
	"""
	if activation=="sigmoid":
		return sigmoid()
	elif activation=="torch_sigmoid":
		return torch_sigmoid()
	elif activation=="tanh":
		return tanh()
	elif activation=="torch_tanh":
		return torch_tanh()
	elif activation=="relu":
		return relu()
	elif activation=="torch_relu":
		return torch_relu()