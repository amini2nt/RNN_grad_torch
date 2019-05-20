import unittest
import torch

from rnnGrad.core.activations import *

class TestActivations(unittest.TestCase):


	def _test_activation(self, my_act, torch_act):

		x = torch.rand(5,4)
		x.requires_grad = True
		my_result = my_act.forward(x)
		torch_result = torch_act(x)
		self.assertTrue((my_result-torch_result).sum() < 1e-16)

		in_grad = torch.rand(4,5)
		my_result = my_act.backward(in_grad)
		res = torch_result.backward(torch.t(in_grad))
		torch_result = torch.t(x.grad)
		self.assertTrue((my_result-torch_result).sum() < 1e-7)


	def test_sigmoid(self):

		my_act = sigmoid()
		torch_act = torch.nn.Sigmoid() 
		self._test_activation(my_act, torch_act)

	def test_tanh(self):

		my_act = tanh()
		torch_act = torch.nn.Tanh() 
		self._test_activation(my_act, torch_act)

	def test_relu(self):

		my_act = relu()
		torch_act = torch.nn.ReLU() 
		self._test_activation(my_act, torch_act)
