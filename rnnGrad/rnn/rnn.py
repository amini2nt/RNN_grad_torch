from rnnGrad.core.activations import *
from rnnGrad.core.layers import *
from rnnGrad.core.model import *

class Recurrent(Model):
	"""A simple recurrent architecture."""

	def __init__(self, input_size, output_size, hidden_size, cell='rnn', activation='tanh',
		chrono_init=False, t_max=10):
		"""Initializes the Recurrent architecture.

		Args:
			input_size: int, number of input features.
			output_size: int, number of output units.
			hidden_size: int, number of hidden units.
			cell: str, valid RNN cell name.
			activation: str, valid activation function. Used only for RNNCell.
			chrono_init: bool, True for chrono initialization of JANET.
			t_max: int, maximum dependency length for chrono initialization.
		"""
		super(Recurrent, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size
		self._cell = cell
		self._activation = activation
		self._chrono_init = chrono_init
		self._t_max = t_max

		self._create_network()

	def _create_network(self):
		"""Creates the network."""
		self._add_cell()
		self.append(linear(self._hidden_size, self._output_size))

	def _add_cell(self):
		"""Adds a recurrent cell to the model."""
		if self._cell == 'rnn':
			self.append(RNNCell(self._input_size, self._hidden_size, self._activation))
		elif self._cell == 'janet':
			self.append(JANETCell(self._input_size, self._hidden_size, self._chrono_init, self._t_max))