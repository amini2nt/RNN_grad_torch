import numpy as np

from core.activations import *
from core.losses import *
from core.layers import *
from core.model import *
from core.optimizers import *

class Recurrent(Model):

	def __init__(self, input_size, output_size, hidden_size):

		super(Recurrent, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size

		self._create_network()

	def _create_network(self):

		self.append(torch_JANETCELL(self._input_size, self._hidden_size))
		self.append(torch_linear(self._hidden_size, self._output_size))

		self.add_loss(torch_ce_with_logits_loss(average='mean'))
