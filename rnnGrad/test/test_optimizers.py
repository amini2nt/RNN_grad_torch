import unittest

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rnnGrad.core.model import Model
from rnnGrad.core.activations import get_activation
from rnnGrad.core.layers import *
from rnnGrad.core.losses import *
from rnnGrad.core.optimizers import *


class MLP(Model):

	def __init__(self):

		super(MLP, self).__init__()

		self.append(linear(5, 10))
		self.append(get_activation("sigmoid"))
		self.append(linear(10, 4))

		self.add_loss(bce_with_logits_loss(average='mean'))


class torch_MLP(nn.Module):

	def __init__(self):

		super(torch_MLP, self).__init__()

		self.l1 = nn.Linear(5, 10)
		self.l2 = nn.Linear(10, 4)

	def forward(self, x):

		h1 = torch.sigmoid(self.l1(x))
		out = self.l2(h1)
		return out

	def compute_loss(self, pred, target):

		loss = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
		return loss


class TestOptimizers(unittest.TestCase):

	def _test_optimizer(self, my_model, torch_model, my_optimizer, torch_optimizer):

		l1_loss = torch.nn.L1Loss()

		torch_model.l1.weight.data.copy_(torch.t(my_model._layer_list[0]._params["W"]))
		torch_model.l1.bias.data.copy_(my_model._layer_list[0]._params["b"])
		torch_model.l2.weight.data.copy_(torch.t(my_model._layer_list[2]._params["W"]))
		torch_model.l2.bias.data.copy_(my_model._layer_list[2]._params["b"])

		x  = torch.randn(100,5)
		y = torch.randint(0,2,(100,4)).float()
		
		for i in range(0, 10):
			my_model.reset()
			my_model.optimizer.zero_grad()
			my_output = my_model.forward(x)
			my_loss = my_model.compute_loss(my_output, y)
			my_model.backward()
			my_model.optimizer.update()
			
			torch_optimizer.zero_grad()
			output = torch_model.forward(x)
			loss = torch_model.compute_loss(output, y)
			loss.backward()
			torch_optimizer.step()

			self.assertTrue(l1_loss(loss, my_loss) < 1e-6)
			self.assertTrue(l1_loss(output, my_output) < 1e-6)

			
	def test_SGD(self):

		my_model = MLP()
		my_optimizer = SGD(learning_rate=0.1, momentum=0)
		my_optimizer.register_model(my_model)
		my_model.register_optimizer(my_optimizer)

		torch_model = torch_MLP()
		torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.1, momentum=0)

		self._test_optimizer(my_model, torch_model, my_optimizer, torch_optimizer)

	def test_SGD_with_Momentum(self):

		my_model = MLP()
		my_optimizer = SGD(learning_rate=0.1, momentum=0.9)
		my_optimizer.register_model(my_model)
		my_model.register_optimizer(my_optimizer)

		torch_model = torch_MLP()
		torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.9)

		self._test_optimizer(my_model, torch_model, my_optimizer, torch_optimizer)

	def test_ADAGRAD(self):

		my_model = MLP()
		my_optimizer = ADAGRAD(learning_rate=0.01, eps=1.0e-10)
		my_optimizer.register_model(my_model)
		my_model.register_optimizer(my_optimizer)

		torch_model = torch_MLP()
		torch_optimizer = optim.Adagrad(torch_model.parameters(), lr=0.01)

		self._test_optimizer(my_model, torch_model, my_optimizer, torch_optimizer)

	def test_RMSprop(self):

		my_model = MLP()
		my_optimizer = RMSprop(learning_rate=0.01, alpha=0.99, eps=1.0e-8)
		my_optimizer.register_model(my_model)
		my_model.register_optimizer(my_optimizer)

		torch_model = torch_MLP()
		torch_optimizer = optim.RMSprop(torch_model.parameters(), lr=0.01, alpha=0.99, eps=1.0e-8)

		self._test_optimizer(my_model, torch_model, my_optimizer, torch_optimizer)






