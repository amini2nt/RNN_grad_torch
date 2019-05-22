import unittest
import torch

from rnnGrad.core.activations import *

class TestActivations(unittest.TestCase):

	def test_sigmoid(self):

		x = torch.rand(5,4)
		my_act = torch_sigmoid()
		torch_act = torch.nn.Sigmoid() 
		my_result = my_act.forward(x)
		torch_result = torch_act(x)
		self.assertEqual((my_result-torch_result).sum(), 0)