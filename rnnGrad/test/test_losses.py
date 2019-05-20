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

	def test_bce_loss(self):

		my_loss = bce_loss()
		torch_loss = torch.nn.BCELoss()
		pred = torch.sigmoid(torch.randn(5,4))
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = bce_loss(average='sum')
		torch_loss = torch.nn.BCELoss(reduction='sum')
		pred = torch.sigmoid(torch.randn(5,4))
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = bce_loss(average='none')
		torch_loss = torch.nn.BCELoss(reduction='none')
		pred = torch.sigmoid(torch.randn(5,4))
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

	def test_bce_with_logits_loss(self):

		my_loss = bce_with_logits_loss()
		torch_loss = torch.nn.BCEWithLogitsLoss()
		pred = torch.randn(5,4)
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = bce_with_logits_loss(average='sum')
		torch_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
		pred = torch.randn(5,4)
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = bce_with_logits_loss(average='none')
		torch_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
		pred = torch.randn(5,4)
		target = torch.randint(0,2,(5,4)).float()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

	def test_ce_with_logits_loss(self):

		my_loss = ce_with_logits_loss()
		torch_loss = torch.nn.CrossEntropyLoss()
		pred = torch.randn(5,10)
		target = torch.randint(0,10,(1,5)).flatten()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = ce_with_logits_loss(average='sum')
		torch_loss = torch.nn.CrossEntropyLoss(reduction='sum')
		pred = torch.randn(5,10)
		target = torch.randint(0,10,(1,5)).flatten()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)

		my_loss = ce_with_logits_loss(average='none')
		torch_loss = torch.nn.CrossEntropyLoss(reduction='none')
		pred = torch.randn(5,10)
		target = torch.randint(0,10,(1,5)).flatten()
		pred.requires_grad = True
		self._test_loss(my_loss, torch_loss, pred, target)