import numpy as np

from rnnGrad.core.activations import *
from rnnGrad.core.losses import *
from rnnGrad.core.layers import *
from rnnGrad.core.model import *
from rnnGrad.core.optimizers import *

class Recurrent(Model):

	def __init__(self, input_size, output_size, hidden_size, chrono_init=False,
		t_max=10, device=None):

		super(Recurrent, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size
		self._chrono_init = chrono_init
		self._t_max = t_max
		self._device = device

		self._create_network()

	def _create_network(self):

		#self.append(torch_JANETCELL(self._input_size, self._hidden_size,
		#	self._chrono_init, self._t_max, self._device))
		self.append(torch_RNNCell(self._input_size, self._hidden_size, "torch_tanh", self._device))
		self.append(torch_linear(self._hidden_size, self._output_size))

		self.add_loss(torch_ce_with_logits_loss(average='mean'))
