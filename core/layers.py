import numpy as np
from core import init
from core.activations import *
from core.model import Layer
import torch



class torch_linear(Layer):
	def __init__(self, input_dim, output_dim, bias=True):	
		super().__init__()
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._bias = bias
		self._params['W'] = init.xavier_uniform(size=(input_dim, output_dim))
		if self._bias: 
			self._params['b'] = init.normal(size=output_dim)
		self._input = {}
		self._output = {}

	def forward(self, input, time=1):
		"""forward prop through the linear layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step

		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		self._input[time] = input
		self._output[time] = torch.matmul(input, self._params['W'])
		if self._bias:
			self._output[time] += self._params['b']
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer

		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		delta = in_grad

		# This line is to try normalizing the in_grad
		#delta = delta / (np.linalg.norm(delta, axis=0)+1e-8)

		self._grads[time] = {}

		self._grads[time]['W'] = torch.t(torch.matmul(delta, self._input[time]))
		if self._bias:
			self._grads[time]['b'] = torch.sum(torch.t(delta), dim=0)
		delta = torch.matmul(self._params['W'], delta)	
		
		# This line normalizes the out_grad
		#ns = max(np.linalg.norm(delta, axis=0)+1e-8)
		#delta = delta / ns

		return delta

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._output = {}
		self._grads = {}
		self._updates = {}


class linear(Layer):
	"""Implementation of a linear layer."""

	def __init__(self, input_dim, output_dim, bias=True):
		"""Initializes a linear layer.

		Args:
			input_dim: int, input dimension.
			output_dim: int, output dimension.
			bias: bool, True to add bias, False otherwise
		"""
		super().__init__()
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._bias = bias
		self._params['W'] = init.xavier_uniform(size=(input_dim, output_dim)).numpy().astype("float32")
		if self._bias: 
			self._params['b'] = init.normal(size=output_dim).numpy().astype("float32")
		self._input = {}
		self._output = {}

	def forward(self, input, time=1):
		"""forward prop through the linear layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step

		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		self._input[time] = input
		self._output[time] = np.matmul(input, self._params['W'])
		if self._bias:
			self._output[time] += self._params['b']
		return self._output[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer

		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		delta = in_grad

		# This line is to try normalizing the in_grad
		#delta = delta / (np.linalg.norm(delta, axis=0)+1e-8)

		self._grads[time] = {}
		self._grads[time]['W'] = np.matmul(delta, self._input[time]).T
		if self._bias:
			self._grads[time]['b'] = np.sum(delta.T, axis=0)
		delta = np.matmul(self._params['W'], delta)	
		
		# This line normalizes the out_grad
		#ns = max(np.linalg.norm(delta, axis=0)+1e-8)
		#delta = delta / ns

		return delta

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._output = {}
		self._grads = {}
		self._updates = {}


class torch_RNNCell(Layer):
	"""Implementation of a Vanilla RNN Cell."""

	def __init__(self, input_dim, output_dim, activation="torch_tanh", device=None):
		"""Initializes the RNN Cell

		Args:
			input_dim: int, input dimension.
			output_dim: int, output dimension.
			activation, str, valid activation function name.
		"""
		super().__init__()
		self._is_recurrent = True
		self._right_grads = {}
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._activation = get_activation(activation)
		self._device = device

		self._params['W'] = init.xavier_uniform(size=(input_dim, output_dim))
		self._params['U'] = init.xavier_uniform(size=(output_dim, output_dim))
		self._params['b'] = init.normal(size=output_dim)

		self._input = {}
		self._h = {}

	def reset_hidden(self, batch_size=1):
		"""Resets the hidden state of the RNN.

		Args:
			batch_size: int, batch size.
		"""

		self._h = {}
		self._h[0] = torch.zeros(size=(batch_size, self._output_dim))
		if self._device is not None:
			self._h[0] = self._h[0].to(self._device)

	def forward(self, input, time=1):
		"""forward prop through the RNN layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step

		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		if time-1 not in self._h:
			raise Exception("Trying to compute h{} when h{} is missing!".format(time, time-1))
		self._input[time] = input
		self._h[time] = torch.matmul(input, self._params['W']) + torch.matmul(self._h[time-1], self._params['U'])
		self._h[time] += self._params['b']

		self._h[time] = self._activation.forward(self._h[time], time=time)
		return self._h[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer

		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		if time+1 not in self._h:
			right_grad = None
		else:
			right_grad = self._right_grads[time+1]

		grad = None
		if in_grad is not None and right_grad is not None:
			grad = in_grad + right_grad
		elif in_grad is None:
			grad = right_grad
		elif right_grad is None:
			grad = in_grad
		else:
			raise Exception("There is no incoming gradient.")


		delta = self._activation.backward(grad, time=time)
		self._grads[time] = {}
		self._grads[time]['W'] = torch.t(torch.matmul(delta, self._input[time]))
		self._grads[time]['U'] = torch.t(torch.matmul(delta, self._h[time-1]))
		self._grads[time]['b'] = torch.sum(torch.t(delta), dim=0)


		self._right_grads[time] = torch.matmul(self._params['U'], delta)
		
		# This line normalizes the gradient going to past time step.
		#ns = max(np.linalg.norm(self._right_grads[time], axis=0))+1e-8
		#self._right_grads[time] = self._right_grads[time] / ns

		out_grad = torch.matmul(self._params['W'], delta)
		return out_grad

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._grads = {}
		self._updates = {}
		self._right_grads = {}
		self._activation.reset_computation()





class RNNCell(Layer):
	"""Implementation of a Vanilla RNN Cell."""

	def __init__(self, input_dim, output_dim, activation="tanh"):
		"""Initializes the RNN Cell

		Args:
			input_dim: int, input dimension.
			output_dim: int, output dimension.
			activation, str, valid activation function name.
		"""
		super().__init__()
		self._is_recurrent = True
		self._right_grads = {}
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._activation = get_activation(activation)

		self._params['W'] = init.xavier_uniform(size=(input_dim, output_dim)).numpy().astype("float32")
		self._params['U'] = init.xavier_uniform(size=(output_dim, output_dim)).numpy().astype("float32")
		self._params['b'] = init.normal(size=output_dim).numpy().astype("float32")

		self._input = {}
		self._h = {}

	def reset_hidden(self, batch_size=1):
		"""Resets the hidden state of the RNN.

		Args:
			batch_size: int, batch size.
		"""

		self._h = {}
		self._h[0] = np.zeros(shape=(batch_size, self._output_dim)).astype("float32")

	def forward(self, input, time=1):
		"""forward prop through the RNN layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step

		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		if time-1 not in self._h:
			raise Exception("Trying to compute h{} when h{} is missing!".format(time, time-1))
		self._input[time] = input
		self._h[time] = np.matmul(input, self._params['W']) + np.matmul(self._h[time-1], self._params['U'])
		self._h[time] += self._params['b']

		self._h[time] = self._activation.forward(self._h[time], time=time)
		return self._h[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer

		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step

		returns:
			numpy array, outgoing gradient.
		"""
		if time+1 not in self._h:
			right_grad = None
		else:
			right_grad = self._right_grads[time+1]

		grad = None
		if in_grad is not None and right_grad is not None:
			grad = in_grad + right_grad
		elif in_grad is None:
			grad = right_grad
		elif right_grad is None:
			grad = in_grad
		else:
			raise Exception("There is no incoming gradient.")


		delta = self._activation.backward(grad, time=time)
		self._grads[time] = {}
		self._grads[time]['W'] = np.matmul(delta, self._input[time]).T
		self._grads[time]['U'] = np.matmul(delta, self._h[time-1]).T
		self._grads[time]['b'] = np.sum(delta.T, axis=0)


		self._right_grads[time] = np.matmul(self._params['U'], delta)
		
		# This line normalizes the gradient going to past time step.
		#ns = max(np.linalg.norm(self._right_grads[time], axis=0))+1e-8
		#self._right_grads[time] = self._right_grads[time] / ns

		out_grad = np.matmul(self._params['W'], delta)
		return out_grad

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._grads = {}
		self._updates = {}
		self._right_grads = {}
		self._activation.reset_computation()



class torch_JANETCELL(Layer):
	"""Implementation of a JANET Cell."""

	def __init__(self, input_dim, output_dim, chrono_init=False, t_max=10, device=None):
		"""Initializes the JANET Cell
		Args:
			input_dim: int, input dimension.
			output_dim: int, output dimension.
		"""

		super().__init__()
		
		self._is_recurrent = True
		self._right_grads = {}
		self._right_grads['h'] = {}
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._chrono_init = chrono_init
		self._t_max = t_max
		self._device = device

		self._params['W_x2f'] = init.xavier_normal(size=(input_dim, output_dim))
		self._params['W_h2f'] = init.orthogonal(size=(output_dim, output_dim))
		self._params['b_f'] = init.constant(size=output_dim, value=1.0)

		self._params['W_x2c'] = init.xavier_normal(size=(input_dim, output_dim))
		self._params['W_h2c'] = init.orthogonal(size=(output_dim, output_dim))
		self._params['b_c'] = init.constant(size=output_dim, value=0.0)

		self._activation = {}
		self._activation['f'] = get_activation('torch_sigmoid')
		self._activation['c'] = get_activation('torch_tanh')

		self._input = {}
		self._h = {}

	def reset_hidden(self, batch_size=1):
		"""Resets the hidden state of the RNN.
		Args:
			batch_size: int, batch size.
		"""

		self._h = {}
		self._h[0] = torch.zeros(size=(batch_size, self._output_dim))
		if self._device is not None:
			self._h[0] = self._h[0].to(self._device)

	def forward(self, input, time=1):
		"""forward prop through the RNN layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step
		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		if time-1 not in self._h:
			raise Exception("Trying to compute h{} when h{} is missing!".format(time, time-1))

		self._input[time] = input

	
		pre_f = torch.matmul(input, self._params['W_x2f']) + torch.matmul(self._h[time-1], self._params['W_h2f'])
		pre_f += self._params['b_f']
		f = self._activation['f'].forward(pre_f, time=time)

		pre_c = torch.matmul(input, self._params['W_x2c']) + torch.matmul(self._h[time-1], self._params['W_h2c'])
		pre_c += self._params['b_c']
		h = f * self._h[time-1] + (1-f) * self._activation['c'].forward(pre_c, time=time)
		self._h[time] = h

		return self._h[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer
		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step
		returns:
			numpy array, outgoing gradient.
		"""
		if time+1 not in self._h:
			right_grad_h = None
		else:
			right_grad_h = self._right_grads['h'][time+1]

		grad_h = None
		if in_grad is not None and right_grad_h is not None:
			grad_h = in_grad + right_grad_h
		elif in_grad is None:
			grad_h = right_grad_h
		elif right_grad_h is None:
			grad_h = in_grad
		else:
			raise Exception("There is no incoming gradient.")


		self._grads[time] = {}

		del_pre_h = grad_h * torch.t((1 - self._activation['f']._output[time]))
		del_pre_h = self._activation['c'].backward(del_pre_h, time)


		self._grads[time]['W_x2c'] = torch.t(torch.matmul(del_pre_h, self._input[time]))
		self._grads[time]['W_h2c'] = torch.t(torch.matmul(del_pre_h, self._h[time-1]))
		self._grads[time]['b_c'] = torch.sum(torch.t(del_pre_h), dim=0)

		del_f = grad_h * torch.t(self._h[time-1])
		del_f -= grad_h * torch.t(self._activation['c']._output[time])

		del_f = self._activation['f'].backward(del_f, time)

		self._grads[time]['W_x2f'] = torch.t(torch.matmul(del_f, self._input[time]))
		self._grads[time]['W_h2f'] = torch.t(torch.matmul(del_f, self._h[time-1]))
		self._grads[time]['b_f'] = torch.sum(torch.t(del_f), dim=0)


		out_grad = torch.matmul(self._params['W_x2c'], del_pre_h) + torch.matmul(self._params['W_x2f'], del_f)
		
		self._right_grads['h'][time] = torch.matmul(self._params['W_h2c'], del_pre_h) + torch.matmul(self._params['W_h2f'], del_f)
		self._right_grads['h'][time] += grad_h * torch.t(self._activation['f']._output[time])
		
		
		return out_grad

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._grads = {}
		self._updates = {}
		self._right_grads = {}
		self._right_grads['h'] = {}
		for activation in self._activation:
			self._activation[activation].reset_computation()



class JANETCELL(Layer):
	"""Implementation of a JANET Cell."""

	def __init__(self, input_dim, output_dim):
		"""Initializes the JANET Cell
		Args:
			input_dim: int, input dimension.
			output_dim: int, output dimension.
		"""
		super().__init__()
		self._is_recurrent = True
		self._right_grads = {}
		self._right_grads['h'] = {}
		self._input_dim = input_dim
		self._output_dim = output_dim

		self._params['W_x2f'] = init.xavier_uniform(size=(input_dim, output_dim)).numpy().astype("float32")
		self._params['W_h2f'] = init.xavier_uniform(size=(output_dim, output_dim)).numpy().astype("float32")
		self._params['b_f'] = init.constant(size=output_dim, value=1.0).numpy().astype("float32")

		self._params['W_x2c'] = init.xavier_uniform(size=(input_dim, output_dim)).numpy().astype("float32")
		self._params['W_h2c'] = init.xavier_uniform(size=(output_dim, output_dim)).numpy().astype("float32")
		self._params['b_c'] = init.constant(size=output_dim, value=0.0).numpy().astype("float32")

		self._activation = {}
		self._activation['f'] = get_activation('sigmoid')
		self._activation['c'] = get_activation('tanh')

		self._input = {}
		self._h = {}

	def reset_hidden(self, batch_size=1):
		"""Resets the hidden state of the RNN.
		Args:
			batch_size: int, batch size.
		"""

		self._h = {}
		self._h[0] = np.zeros(shape=(batch_size, self._output_dim)).astype("float32")

	def forward(self, input, time=1):
		"""forward prop through the RNN layer.
		
		Args:
			input: numpy array of size (batch_size, input_dim)
			time: int, time step
		returns:
			output: numpy array of size (batch_size, output_dim)
		"""
		if time-1 not in self._h:
			raise Exception("Trying to compute h{} when h{} is missing!".format(time, time-1))

		self._input[time] = input

	
		pre_f = np.matmul(input, self._params['W_x2f']) + np.matmul(self._h[time-1], self._params['W_h2f'])
		pre_f += self._params['b_f']
		f = self._activation['f'].forward(pre_f, time=time)

		pre_c = np.matmul(input, self._params['W_x2c']) + np.matmul(self._h[time-1], self._params['W_h2c'])
		pre_c += self._params['b_c']
		h = f * self._h[time-1] + (1-f) *self._activation['c'].forward(pre_c, time=time)
		self._h[time] = h

		return self._h[time]

	def backward(self, in_grad, time=1):
		"""backprop through the linear layer
		Args:
			in_grad: numpy array, gradient from the upper layer.
			time: int, time step
		returns:
			numpy array, outgoing gradient.
		"""
		if time+1 not in self._h:
			right_grad_h = None
		else:
			right_grad_h = self._right_grads['h'][time+1]

		grad_h = None
		if in_grad is not None and right_grad_h is not None:
			grad_h = in_grad + right_grad_h
		elif in_grad is None:
			grad_h = right_grad_h
		elif right_grad_h is None:
			grad_h = in_grad
		else:
			raise Exception("There is no incoming gradient.")


		self._grads[time] = {}

		del_pre_h = grad_h * (1 - self._activation['f']._output[time]).T
		del_pre_h = self._activation['c'].backward(del_pre_h, time)


		self._grads[time]['W_x2c'] = np.matmul(del_pre_h, self._input[time]).T
		self._grads[time]['W_h2c'] = np.matmul(del_pre_h, self._h[time-1]).T
		self._grads[time]['b_c'] = np.sum(del_pre_h.T, axis=0)

		del_f = grad_h * self._h[time-1].T
		del_f -= grad_h * self._activation['c']._output[time].T

		del_f = self._activation['f'].backward(del_f, time)

		self._grads[time]['W_x2f'] = np.matmul(del_f, self._input[time]).T
		self._grads[time]['W_h2f'] = np.matmul(del_f, self._h[time-1]).T
		self._grads[time]['b_f'] = np.sum(del_f.T, axis=0)


		out_grad = np.matmul(self._params['W_x2c'], del_pre_h) + np.matmul(self._params['W_x2f'], del_f)
		
		self._right_grads['h'][time] = np.matmul(self._params['W_h2c'], del_pre_h) + np.matmul(self._params['W_h2f'], del_f)
		self._right_grads['h'][time] += grad_h * self._activation['f']._output[time].T
		
		
		return out_grad

	def reset_computation(self):
		"""resets the layer computations."""
		self._input = {}
		self._grads = {}
		self._updates = {}
		self._right_grads = {}
		self._right_grads['h'] = {}
		for activation in self._activation:
			self._activation[activation].reset_computation()



