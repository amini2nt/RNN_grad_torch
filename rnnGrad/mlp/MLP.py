from rnnGrad.core.model import Model
from rnnGrad.core.activations import get_activation
from rnnGrad.core.layers import *


class MLP(Model):
	"""A simple MLP architecture."""

	def __init__(self, num_hidden_layers=2, hidden_layer_size=[10,10], activation="relu",
		input_dim=100, output_dim=10):
		"""Initializes the MLP architecture.

		Args:
			num_hidden_layers: int, number of hidden layers.
			hidden_layer_size: list, list of hidden layer sizes.
			activation: str, valid activation function.
			input_dim: int, number of input features.
			output_dim: int, number of output units.
		"""
		super(MLP, self).__init__()
		self._num_hidden_layers = num_hidden_layers
		self._hidden_layer_size = hidden_layer_size
		self._activation = activation
		self._input_dim = input_dim
		self._output_dim = output_dim

		self.create_network()

	def create_network(self):
		"""Creates the network."""

		self.append(linear(self._input_dim, self._hidden_layer_size[0]))
		self.append(get_activation(self._activation))
		for i in range(1, self._num_hidden_layers):
			self.append(linear(self._hidden_layer_size[i-1], self._hidden_layer_size[i]))
			self.append(get_activation(self._activation))
		self.append(linear(self._hidden_layer_size[-1], self._output_dim))