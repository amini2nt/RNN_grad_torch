import unittest
import torch

from rnnGrad.core.layers import *
from rnnGrad.myTorch.memory import RNNCell as TorchRNN
from rnnGrad.myTorch.memory import JANETCell as TorchJANET

class TestLayers(unittest.TestCase):


	def test_linear(self):

		my_layer = linear(5, 10)
		torch_layer = torch.nn.Linear(5, 10)
		loss = torch.nn.L1Loss()
		torch_layer.weight.data.copy_(torch.t(my_layer._params['W']))
		torch_layer.bias.data.copy_(my_layer._params['b'])

		input = torch.randn(2,5)
		input.requires_grad = True

		my_out = my_layer.forward(input)
		torch_out = torch_layer.forward(input)

		self.assertTrue(loss(my_out, torch_out) < 1e-5)

		in_grad = torch.rand(10,2)
		my_result = my_layer.backward(in_grad)
		res = torch_out.backward(torch.t(in_grad))
		torch_result = torch.t(input.grad)
		self.assertTrue(loss(my_result, torch_result) < 1e-6)

		my_grad = my_layer._grads[1]["W"]
		torch_grad = torch.t(torch_layer.weight.grad)
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b"]
		torch_grad = torch_layer.bias.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

	def test_RNNCell(self):

		my_layer = RNNCell(5, 10)
		torch_layer =  TorchRNN(torch.device('cpu'), 5, 10)
		loss = torch.nn.L1Loss()
		torch_layer._W_i2h.data.copy_(my_layer._params['W'])
		torch_layer._W_h2h.data.copy_(my_layer._params['U'])
		torch_layer._b_h.data.copy_(my_layer._params['b'])

		input = torch.randn(2,5)
		input.requires_grad = True

		my_layer.reset_hidden(2)
		h = torch_layer.reset_hidden(2)
		my_out = my_layer.forward(input)
		torch_out = torch_layer.forward(input, h)["h"]

		self.assertTrue(loss(my_out, torch_out) < 1e-5)

		in_grad = torch.rand(10,2)
		my_result = my_layer.backward(in_grad)
		res = torch_out.backward(torch.t(in_grad))
		torch_result = torch.t(input.grad)
		self.assertTrue(loss(my_result, torch_result) < 1e-6)

		my_grad = my_layer._grads[1]["W"]
		torch_grad = torch_layer._W_i2h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["U"]
		torch_grad = torch_layer._W_h2h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b"]
		torch_grad = torch_layer._b_h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

	def test_RNNCell_multi_step(self):

		my_layer = RNNCell(5, 10)
		torch_layer =  TorchRNN(torch.device('cpu'), 5, 10)
		loss = torch.nn.L1Loss()
		torch_layer._W_i2h.data.copy_(my_layer._params['W'])
		torch_layer._W_h2h.data.copy_(my_layer._params['U'])
		torch_layer._b_h.data.copy_(my_layer._params['b'])

		input = torch.randn(2,5)

		my_layer.reset_hidden(2)
		h = torch_layer.reset_hidden(2)
		my_out_1 = my_layer.forward(input, time=1)
		torch_out_1 = torch_layer.forward(input, h)
		my_out_2 = my_layer.forward(input, time=2)
		torch_out_2 = torch_layer.forward(input, torch_out_1)

		in_grad = torch.rand(10,2)
		my_layer.backward(in_grad, time=2)
		my_layer.backward(None, time=1)
		res = torch_out_2["h"].backward(torch.t(in_grad))

		my_grad = my_layer._grads[1]["W"] + my_layer._grads[2]["W"]
		torch_grad = torch_layer._W_i2h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["U"] + my_layer._grads[2]["U"]
		torch_grad = torch_layer._W_h2h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b"] + my_layer._grads[2]["b"]
		torch_grad = torch_layer._b_h.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

	def test_JANETCell(self):

		my_layer = JANETCell(5, 10)
		torch_layer =  TorchJANET(torch.device('cpu'), 5, 10)
		loss = torch.nn.L1Loss()
		torch_layer._W_x2f.data.copy_(my_layer._params['W_x2f'])
		torch_layer._W_h2f.data.copy_(my_layer._params['W_h2f'])
		torch_layer._b_f.data.copy_(my_layer._params['b_f'])
		torch_layer._W_x2c.data.copy_(my_layer._params['W_x2c'])
		torch_layer._W_h2c.data.copy_(my_layer._params['W_h2c'])
		torch_layer._b_c.data.copy_(my_layer._params['b_c'])

		input = torch.randn(2,5)
		input.requires_grad = True

		my_layer.reset_hidden(2)
		h = torch_layer.reset_hidden(2)
		my_out = my_layer.forward(input)
		torch_out = torch_layer.forward(input, h)["h"]

		self.assertTrue(loss(my_out, torch_out) < 1e-5)

		in_grad = torch.rand(10,2)
		my_result = my_layer.backward(in_grad)
		res = torch_out.backward(torch.t(in_grad))
		torch_result = torch.t(input.grad)
		self.assertTrue(loss(my_result, torch_result) < 1e-6)

		my_grad = my_layer._grads[1]["W_x2f"]
		torch_grad = torch_layer._W_x2f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_h2f"] 
		torch_grad = torch_layer._W_h2f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b_f"] 
		torch_grad = torch_layer._b_f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_x2c"]
		torch_grad = torch_layer._W_x2c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_h2c"] 
		torch_grad = torch_layer._W_h2c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b_c"] 
		torch_grad = torch_layer._b_c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

	def test_JANETCell_multi_step(self):

		my_layer = JANETCell(5, 10)
		torch_layer =  TorchJANET(torch.device('cpu'), 5, 10)
		loss = torch.nn.L1Loss()
		torch_layer._W_x2f.data.copy_(my_layer._params['W_x2f'])
		torch_layer._W_h2f.data.copy_(my_layer._params['W_h2f'])
		torch_layer._b_f.data.copy_(my_layer._params['b_f'])
		torch_layer._W_x2c.data.copy_(my_layer._params['W_x2c'])
		torch_layer._W_h2c.data.copy_(my_layer._params['W_h2c'])
		torch_layer._b_c.data.copy_(my_layer._params['b_c'])

		input = torch.randn(2,5)

		my_layer.reset_hidden(2)
		h = torch_layer.reset_hidden(2)
		my_out_1 = my_layer.forward(input, time=1)
		torch_out_1 = torch_layer.forward(input, h)
		my_out_2 = my_layer.forward(input, time=2)
		torch_out_2 = torch_layer.forward(input, torch_out_1)

		in_grad = torch.rand(10,2)
		my_layer.backward(in_grad, time=2)
		my_layer.backward(None, time=1)
		res = torch_out_2["h"].backward(torch.t(in_grad))

		my_grad = my_layer._grads[1]["W_x2f"] + my_layer._grads[2]["W_x2f"]
		torch_grad = torch_layer._W_x2f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_h2f"] + my_layer._grads[2]["W_h2f"]
		torch_grad = torch_layer._W_h2f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b_f"] + my_layer._grads[2]["b_f"]
		torch_grad = torch_layer._b_f.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_x2c"] + my_layer._grads[2]["W_x2c"]
		torch_grad = torch_layer._W_x2c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["W_h2c"] + my_layer._grads[2]["W_h2c"]
		torch_grad = torch_layer._W_h2c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)

		my_grad = my_layer._grads[1]["b_c"] + my_layer._grads[2]["b_c"]
		torch_grad = torch_layer._b_c.grad
		self.assertTrue(loss(my_grad, torch_grad) < 1e-6)