import unittest
import torch

from rnnGrad.core.losses import *

class TestLosses(unittest.TestCase):


	def _test_loss(self, my_loss, torch_loss, pred, target):

		my_result = my_loss.forward(pred, target)
		torch_result = torch_loss(pred, target)
		loss = torch.nn.L1Loss()
		self.assertTrue(loss(my_result,torch_result) < 1e-5)

		my_result = my_loss.backward()
		res = torch_result.sum().backward()
		torch_result = torch.t(pred.grad)

		self.assertTrue(loss(my_result,torch_result) < 1e-7)


	def test_l2_loss(self):

		my_loss = l2_loss()
		torch_loss = torch.nn.MSELoss()
		pred = torch.randn(5,4)
		target = torch.randn(5,4)
		pred.requires_grad = True 
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = l2_loss(average='sum')
		torch_loss = torch.nn.MSELoss(reduction='sum')
		pred = torch.randn(5,4)
		target = torch.randn(5,4)
		pred.requires_grad = True 
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = l2_loss(average='none')
		torch_loss = torch.nn.MSELoss(reduction='none')
		pred = torch.randn(5,4)
		target = torch.randn(5,4)
		pred.requires_grad = True 
		self._test_loss(my_loss, torch_loss, pred, target)