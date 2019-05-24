import pickle
import os
import torch


class Layer(object):
	"""Base class for all layers."""

	def __init__(self):
		"""Basic initialization."""
		self._params = {}
		self._grads = {}
		self._updates = {}
		self._is_recurrent = False
		self._time_shared = False

	def forward(self, input, time=1):
		"""forward prop through the layer.
		
		Args:
			input: torch array of size (batch_size, input_dim)
			time: int, time step
		"""
		raise Exception("Not Implemented!")

	def backward(self, in_grad, time=1):
		"""backward prop through the layer.

		Args:
			in_grad: torch array, incoming gradient.
			time: int, time step
		"""
		raise Exception("Not Implemented!")

	def reset_computation(self):
		"""resets the layer computations."""
		raise Exception("Not Implemented!")

	def is_recurrent(self):
		"""Returns true if the layer is recurrent."""
		return self._is_recurrent

	def is_time_shared(self):
		"""Returns true is the layer is shared across time steps."""
		return self._time_shared


class Model(object):
	"""Implementation of a model class with basic functionalities."""

	def __init__(self):
		"""Initializes a model object."""
		self._layer_list = []

	def append(self, layer):
		"""Appends a layer on the top of the network.

		Args:
			layer: a layer object.
		"""
		if layer.is_time_shared():
			for layer in self._layer_list:
				layer._time_shared = True
		self._layer_list.append(layer)

	def add_loss(self, loss):
		"""Adds the loss which will be used to train the network.
		Args:
			loss: a loss object.
		"""
		self._loss = loss

	def reset(self):
		"""Resets the model computations."""
		self._loss.reset_computation()
		for layer in self._layer_list:
			layer.reset_computation()

	def reset_hidden(self, batch_size):
		"""Resets the hidden state of the recurrent layers.

		Args:
			batch_size: int, batch size.
		"""
		for layer in self._layer_list:
			if layer.is_recurrent():
				layer.reset_hidden(batch_size)

	def forward(self, input, time=1):
		"""Forward prop through the network.
		
		Args:
			input: torch array, input to the network
			time: int, time step

		returns:
			torch array, output of the network
		"""
		x = input
		for i in range(len(self._layer_list)):
			x = self._layer_list[i].forward(x, time)
		return x

	def compute_loss(self, pred, target, time=1):
		"""Computes the loss.
		Args:
			pred: torch array, model's prediction.
			target:	torch array, target value.
			time: int, time step

		returns:
			loss, scalar or array based on the loss type.
		"""
		return self._loss.forward(pred, target, time)

	def backward(self, time=1):
		"""Backprop through the network.

		Args:
			time: int, time step
		"""

		grad = self._loss.backward(time)
		for i in range(len(self._layer_list)-1, -1, -1):
			grad = self._layer_list[i].backward(grad, time)

		#backprop through time.
		if time>1:
			for r in range(time-1, 0, -1):
				grad = None
				beg_rec = False
				for i in range(len(self._layer_list)-1, -1, -1):
					if self._layer_list[i].is_recurrent():
						beg_rec = True
					if beg_rec:
						grad = self._layer_list[i].backward(grad, r)

	def register_optimizer(self, optimizer):
		"""Registers an optimizer for the model.

        Args:
            optimizer: optimizer object.
        """

		self.optimizer = optimizer

	def save(self, dir_name):
		"""Saves the model parameters.

		Args:
			file_name: str, name of the backup file.
		"""
		params_dict = {}
		for i in range(0, len(self._layer_list)):
			params_dict[i] = self._layer_list[i]._params

		file_name = os.path.join(dir_name, "model.p")
		pickle.dump(params_dict, open(file_name, "wb"))

		file_name = os.path.join(dir_name, "optim.p")
		pickle.dump(self.optimizer._params, open(file_name, "wb"))

	def load(self, dir_name):
		"""Loads the model parameters from the backup file.

		Args:
			file_name: str, name of the backup file.
		"""
		file_name = os.path.join(dir_name, "model.p")
		params_dict = pickle.load(open(file_name, "rb"))
		for i in range(0, len(self._layer_list)):
			self._layer_list[i]._params = params_dict[i]

		file_name = os.path.join(dir_name, "optim.p")
		self.optimizer._params = pickle.load(open(file_name, "rb"))

	def num_parameters(self):
		"""Returns the number of parameters in the model."""
		total = 0
		for layer in self._layer_list:
			if len(layer._params) > 0:
				for p in layer._params:
					total += layer._params[p].numel()
		return total

	def cuda(self):

		for layer in self._layer_list:
			if len(layer._params) > 0:
				for p in layer._params:
					layer._params[p] = layer._params[p].cuda()