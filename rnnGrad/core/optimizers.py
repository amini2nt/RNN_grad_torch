import numpy as np
import math
import torch


class Optimizer(object):
	"""Base class for all the optimizers."""

	def __init__(self, learning_rate):
		"""Initializes the optimizer.

		Args:
			learning_rate: float, learning rate.
		"""
		self._lr = learning_rate

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""
		raise Exception("Not Implemented!")

	def zero_grad(self):
		"""Zeros out the previous gradients."""

		for layer in self._model._layer_list:
			layer._grads = {}
			layer._updates = {}


class SGD(Optimizer):
	"""Implementation of SGD update rule."""

	def __init__(self, learning_rate, momentum=0):
		"""Initialize an SGD optimizer.

		Args:
			learning_rate: float, learning rate.
			momentum: float, momentum coefficient.
		"""
		super().__init__(learning_rate)
		self._momentum = momentum

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		super().register_model(model)
		if self._momentum > 0 :
			self._params = {}
			self._params["v"] = {}
			for i in range(0, len(self._model._layer_list)):
				layer = self._model._layer_list[i]
				self._params["v"][i] = {}
				if len(layer._params) > 0:
					for p in layer._params.keys():
						self._params["v"][i][p] = torch.zeros_like(layer._params[p])

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""

		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_time_shared():
					for param in layer._params.keys():
						grad = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param] 
						if self._momentum > 0 :
							self._params["v"][i][param] = self._momentum * self._params["v"][i][param] + grad
							layer._updates[param] = - (self._lr * self._params["v"][i][param])
						else:
							layer._updates[param] = - self._lr * grad
						
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]					
						if self._momentum > 0 :
							self._params["v"][i][param] = self._momentum * self._params["v"][i][param] + grad
							layer._updates[param] = - (self._lr * self._params["v"][i][param])
						else:
							layer._updates[param] = - self._lr * grad
						layer._params[param] = layer._params[param] + layer._updates[param]


class ADAGRAD(Optimizer):
	"""Implementation of ADAGRAD update rule."""

	def __init__(self, learning_rate=0.01, eps=1.0e-8):
		"""Initialize an ADAGRAD optimizer.

		Args:
			learning_rate: float, learning rate.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)		
		self._eps = eps

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""	
		self._model = model
		self._params = {}
		self._params["sqrs"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["sqrs"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["sqrs"][i][p] = torch.zeros_like(layer._params[p])

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_time_shared():
					for param in layer._params.keys():
						grad = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param] 
						self._params["sqrs"][i][param] += ( grad ** 2) 
						layer._updates[param] = -(self._lr * grad / (torch.sqrt(self._params["sqrs"][i][param] ) + self._eps ))
						layer._params[param] = layer._params[param] + layer._updates[param]
					
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["sqrs"][i][param] += ( grad ** 2) 
						layer._updates[param] = -(self._lr * grad / (torch.sqrt(self._params["sqrs"][i][param]) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]		


class RMSprop(Optimizer):
	"""Implementation of RMSprop update rule."""

	def __init__(self, learning_rate=0.01, alpha=0.99, eps=1.0e-8):
		"""Initialize an RMSprop optimizer.

		Args:
			learning_rate: float, learning rate.
			alpha: float, smoothing constant.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		self._alpha = alpha
		self._eps = eps
		
	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["g2"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["g2"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["g2"][i][p] = torch.zeros_like(layer._params[p])

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_time_shared():
					for param in layer._params.keys():
						grad = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param] 
						self._params["g2"][i][param] = self._alpha * self._params["g2"][i][param] + (1 - self._alpha) * (grad**2)
						layer._updates[param] = - (self._lr/(torch.sqrt(self._params["g2"][i][param])+self._eps)) * grad
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]					
						self._params["g2"][i][param] = self._alpha * self._params["g2"][i][param] + (1 - self._alpha) * (grad**2)
						layer._updates[param] = - (self._lr/(torch.sqrt(self._params["g2"][i][param])+self._eps)) * grad
						layer._params[param] = layer._params[param] + layer._updates[param]


class ADADELTA(Optimizer):
	"""Implementation of ADADELTA update rule."""

	def __init__(self, learning_rate=1.0 , gamma=0.9, eps=1.0e-8):
		"""Initialize an ADADELTA optimizer.

		Args:
			learning_rate: float, learning rate.
			gamma: float, smoothing constant.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)		
		self._eps = eps
		self._gamma = gamma

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["sqr"] = {}
		self._params["delta"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["sqr"][i] = {}
			self._params["delta"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["sqr"][i][p] = torch.zeros_like(layer._params[p])
					self._params["delta"][i][p] = torch.zeros_like(layer._params[p])

				
	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_time_shared():
					for param in layer._params.keys():
						grad = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param]
						self._params["sqr"][i][param] = self._gamma * self._params["sqr"][i][param] + (1 - self._gamma) * (grad**2)
						layer._updates[param] = - (torch.sqrt(self._params["delta"][i][param] + self._eps) / torch.sqrt(self._params["sqr"][i][param] + self._eps) ) * grad
						self._params["delta"][i][param] = self._gamma * self._params["delta"][i][param] + (1 - self._gamma) * layer._updates[param] * layer._updates[param]
						layer._updates[param] = self._lr * layer._updates[param]
						layer._params[param] = layer._params[param] + layer._updates[param]

				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["sqr"][i][param] = self._gamma * self._params["sqr"][i][param] + (1 - self._gamma) * (grad**2)
						layer._updates[param] = - (torch.sqrt(self._params["delta"][i][param] + self._eps) / torch.sqrt(self._params["sqr"][i][param] + self._eps) ) * grad
						self._params["delta"][i][param] = self._gamma * self._params["delta"][i][param] + (1 - self._gamma) * layer._updates[param] * layer._updates[param]
						layer._updates[param] = self._lr * layer._updates[param]
						layer._params[param] = layer._params[param] + layer._updates[param]


class ADAM(Optimizer):
	"""Implementation of ADAM update rule."""

	def __init__(self, learning_rate= 0.01, beta1 = 0.9, beta2 = 0.999, eps= 1.0e-8):
		"""Initialize an ADAM optimizer.

		Args:
			learning_rate: float, learning rate.
			beta1: float, decay rate
			beta2: float, decay rate
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		self._beta1 = beta1
		self._beta2 = beta2
		self._eps = eps


	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["t"] = 0
		self._params["m"] = {}
		self._params["v"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["m"][i] = {}
			self._params["v"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["v"][i][p] = torch.zeros_like(layer._params[p])
					self._params["m"][i][p] = torch.zeros_like(layer._params[p])
				
	def update(self, time=1):
		"""Updates the network parameters using the gradients."""
		self._params["t"] += 1
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						grad = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param]
						
						self._params["m"][i][param] = self._beta1 * self._params["m"][i][param] + (1 - self._beta1) * (grad)					
						self._params["v"][i][param] = self._beta2 * self._params["v"][i][param] + (1 - self._beta2) * (grad **2)
						bias_correction1 = 1. - self._beta1**self._params["t"]
						bias_correction2 = 1. - self._beta2**self._params["t"]
						step_size = self._lr * math.sqrt(bias_correction2) / bias_correction1
						layer._updates[param] = - (step_size * self._params["m"][i][param] / (torch.sqrt(self._params["v"][i][param]) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["m"][i][param] = self._beta1 * self._params["m"][i][param] + (1 - self._beta1) * (grad)					
						self._params["v"][i][param] = self._beta2 * self._params["v"][i][param] + (1 - self._beta2) * (grad **2)
						bias_correction1 = 1. - self._beta1**self._params["t"]
						bias_correction2 = 1. - self._beta2**self._params["t"]
						step_size = self._lr * math.sqrt(bias_correction2) / bias_correction1
						layer._updates[param] = - (step_size * self._params["m"][i][param] / (torch.sqrt(self._params["v"][i][param]) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]


########################### Weight sharing

class torch_WA_RMSprop(Optimizer):
	"""Implementation of RMSprop update rule with weight sharing awareness."""

	def __init__(self, learning_rate=0.01, alpha=0.99, eps=1.0e-8):
		"""Initialize an RMSprop optimizer.

		Args:
			learning_rate: float, learning rate.
			alpha: float, smoothing constant.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		self._alpha = alpha
		self._eps = eps
		
	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["g2"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["g2"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["g2"][i][p] = torch.zeros_like(layer._params[p])

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""

		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad = layer._grads[t][param] 
								self._params["g2"][i][param] = self._alpha * self._params["g2"][i][param] + (1 - self._alpha) * (grad**2)
								layer._updates[param] += - (self._lr/(torch.sqrt(self._params["g2"][i][param])+self._eps)) * grad
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]					
						self._params["g2"][i][param] = self._alpha * self._params["g2"][i][param] + (1 - self._alpha) * (grad**2)
						layer._updates[param] = - (self._lr/(torch.sqrt(self._params["g2"][i][param])+self._eps)) * grad
						layer._params[param] = layer._params[param] + layer._updates[param]



class torch_WA_ADAM(Optimizer):
	"""Implementation of ADAM update rule with weight sharing."""

	def __init__(self, learning_rate= 0.01, beta1 = 0.9, beta2 = 0.999, eps= 1.0e-8):
		"""Initialize an ADAM optimizer

		Args:
			learning_rate: float, learning rate.
			beta1: float, decay rate
			beta2: float, decay rate
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		self._beta1 = beta1
		self._beta2 = beta2
		self._eps = eps


	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		
		self._model = model
		self._params = {}
		self._params["t"] = 0
		self._params["m"] = {}
		self._params["v"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["m"][i] = {}
			self._params["v"][i] = {}
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for p in layer._params.keys():
						self._params["v"][i][p] = {}
						self._params["m"][i][p] = {}
						for t in range(0, self._max_steps):
							self._params["v"][i][p][t] = torch.zeros_like(layer._params[p])
							self._params["m"][i][p][t] = torch.zeros_like(layer._params[p])

				else:
					for p in layer._params.keys():
						self._params["v"][i][p] = torch.zeros_like(layer._params[p])
						self._params["m"][i][p] = torch.zeros_like(layer._params[p])

					
	def update(self, time=1):
		"""Updates the network parameters using the gradients."""
		self._params["t"] += 1
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad += layer._grads[t][param]
						
								self._params["m"][i][param] = self._beta1 * self._params["m"][i][param] + (1 - self._beta1) * (grad)					
								self._params["v"][i][param] = self._beta2 * self._params["v"][i][param] + (1 - self._beta2) * (grad **2)
								m_corrected = self._params["m"][i][param] / (1. - self._beta1**self._params["t"])
								v_corrected = self._params["v"][i][param] / (1. - self._beta2**self._params["t"])		
								layer._updates[param] += - (self._lr * m_corrected / (torch.sqrt(v_corrected) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["m"][i][param] = self._beta1 * self._params["m"][i][param] + (1 - self._beta1) * (grad)					
						self._params["v"][i][param] = self._beta2 * self._params["v"][i][param] + (1 - self._beta2) * (grad **2)
						m_corrected = self._params["m"][i][param] / (1. - self._beta1**self._params["t"])
						v_corrected = self._params["v"][i][param] / (1. - self._beta2**self._params["t"])		
						layer._updates[param] = - (self._lr * m_corrected / (torch.sqrt(v_corrected) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]


class torch_WA_ADAGRAD(Optimizer):
	"""Implementation of ADAGRAD update rule"""

	def __init__(self, learning_rate=0.01, eps=1.0e-8):
		"""Initialize an ADAGRAD optimizer

		Args:
			learning_rate: float, learning rate.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		
		self._eps = eps

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""	
		self._model = model
		self._params = {}
		self._params["sqrs"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["sqrs"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["sqrs"][i][p] = torch.zeros_like(layer._params[p])

	def update(self, time=1):
		"""Updates the network parameters using the gradients."""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad = layer._grads[t][param] 
								self._params["sqrs"][i][param] += ( grad ** 2) 
								layer._updates[param] += -(self._lr * grad / (torch.sqrt(self._params["sqrs"][i][param] ) + self._eps ))
						layer._params[param] = layer._params[param] + layer._updates[param]
					
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["sqrs"][i][param] += ( grad ** 2) 
						layer._updates[param] = -(self._lr * grad / (torch.sqrt(self._params["sqrs"][i][param]) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]		


class torch_WA_ADADELTA(Optimizer):
	"""Implementation of SGD update rule"""

	def __init__(self, learning_rate=0.1 , gamma=0.9, eps=1.0e-8):
		"""Initialize an SGD optimizer

		Args:
			learning_rate: float, learning rate.
			alpha: float, smoothing constant.
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		
		self._eps = eps
		self._gamma = gamma

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["sqr"] = {}
		self._params["delta"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["sqr"][i] = {}
			self._params["delta"][i] = {}
			if len(layer._params) > 0:
				for p in layer._params.keys():
					self._params["sqr"][i][p] = torch.zeros_like(layer._params[p])
					self._params["delta"][i][p] = torch.zeros_like(layer._params[p])

				
	def update(self, time=1):
		"""Updates the network parameters using the gradients."""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = torch.zeros_like(layer._params[param])
						for t in layer._grads.keys():
							if t <= time:
								grad = layer._grads[t][param]

								self._params["sqr"][i][param] = self._gamma * self._params["sqr"][i][param] + (1 - self._gamma) * (grad**2)
								layer._updates[param] += -(torch.sqrt(self._params["delta"][i][param] + self._eps) / torch.sqrt(self._params["sqr"][i][param] + self._eps) ) * grad
								self._params["delta"][i][param] = self._gamma * self._params["delta"][i][param] + (1 - self._gamma) * layer._updates[param] * layer._updates[param]
						layer._params[param] = layer._params[param] + layer._updates[param]

				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["sqr"][i][param] = self._gamma * self._params["sqr"][i][param] + (1 - self._gamma) * (grad**2)
						layer._updates[param] = -(torch.sqrt(self._params["delta"][i][param] + self._eps) / torch.sqrt(self._params["sqr"][i][param] + self._eps) ) * grad
						self._params["delta"][i][param] = self._gamma * self._params["delta"][i][param] + (1 - self._gamma) * layer._updates[param] * layer._updates[param]
						layer._params[param] = layer._params[param] + layer._updates[param]




class TA_RMSprop(Optimizer):
	"""Implementation of time-aware RMSprop update rule"""

	def __init__(self, learning_rate=0.01, alpha=0.99, eps=1.0e-8, max_steps=150):
		"""Initialize a time-aware RMSprop optimizer.

		Args:
			learning_rate: float, learning rate.
			alpha: float, smoothing constant.
			eps: float, term added for numerical stability.
			max_steps: int, maximum time steps for recurrence.
		"""
		super().__init__(learning_rate)
		self._alpha = alpha
		self._eps = eps
		self._max_steps = max_steps

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		self._model = model
		self._params = {}
		self._params["g2"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["g2"][i] = {}
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for p in layer._params.keys():
						self._params["g2"][i][p] = {}
						for t in range(0, self._max_steps):
							self._params["g2"][i][p][t] = np.zeros(shape=layer._params[p].shape).astype("float32")
				else:
					for p in layer._params.keys():
						self._params["g2"][i][p] = np.zeros(shape=layer._params[p].shape).astype("float32")
				

	def update(self, time=1):
		"""Updates the network parameters using the gradients.

		Args:
			time: int, time step
		"""
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = np.zeros(shape=layer._params[param].shape).astype("float32")
						for t in range(time, 0, -1):

							grad = layer._grads[t][param]
							ta = time-t 
							#print(ta)
							self._params["g2"][i][param][ta] = self._alpha * self._params["g2"][i][param][ta] + (1 - self._alpha) * (grad**2)
							layer._updates[param] += - (self._lr/(np.sqrt(self._params["g2"][i][param][ta])+self._eps)) * grad ## the addition operation is not consistent

						layer._params[param] = layer._params[param] + layer._updates[param] ## is the difference intentional 
				else:							
					for param in layer._params.keys():
						grad = layer._grads[time][param]					
						self._params["g2"][i][param] = self._alpha * self._params["g2"][i][param] + (1 - self._alpha) * (grad**2)
						layer._updates[param] = - (self._lr/(np.sqrt(self._params["g2"][i][param])+self._eps)) * grad
						layer._params[param] = layer._params[param] + layer._updates[param]


class TA_ADAM(Optimizer):
	"""Implementation of ADAM update rule"""

	def __init__(self, learning_rate=0.01, beta1 = 0.9, beta2 = 0.999, eps= 1.0e-8, max_steps = 150):
		"""Initialize an ADAM optimizer

		Args:
			learning_rate: float, learning rate.
			beta1: float, decay rate
			beta2: float, decay rate
			eps: float, term added for numerical stability.
		"""
		super().__init__(learning_rate)
		self._beta1 = beta1
		self._beta2 = beta2
		self._eps = eps
		self._max_steps = max_steps

	def register_model(self, model):
		"""register a model to the optimizer.
		
		Args:
			model: a model object.
		"""
		super().register_model(model)
		self._params["t"] = 0
		self._params["m"] = {}
		self._params["v"] = {}
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			self._params["m"][i] = {}
			self._params["v"][i] = {}
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for p in layer._params.keys():
						self._params["v"][i][p] = {}
						self._params["m"][i][p] = {}
						for t in range(0, self._max_steps):
							self._params["v"][i][p][t] = np.zeros(shape=layer._params[p].shape).astype("float32")
							self._params["m"][i][p][t] = np.zeros(shape=layer._params[p].shape).astype("float32")

				else:
					for p in layer._params.keys():
						self._params["v"][i][p] = np.zeros(shape=layer._params[p].shape).astype("float32")
						self._params["m"][i][p] = np.zeros(shape=layer._params[p].shape).astype("float32")

					
	def update(self, time=1):
		"""Updates the network parameters using the gradients."""
		
		self._params["t"] += 1
		for i in range(0, len(self._model._layer_list)):
			layer = self._model._layer_list[i]
			if len(layer._params) > 0:
				if layer.is_recurrent():
					for param in layer._params.keys():
						layer._updates[param] = np.zeros(shape=layer._params[param].shape).astype("float32")
						for t in range(time, 0, -1):
							grad = layer._grads[t][param]
							ta = time-t 
							self._params["m"][i][param][ta] = self._beta1 * self._params["m"][i][param][ta] + (1 - self._beta1) * (grad)					
							self._params["v"][i][param][ta] = self._beta2 * self._params["v"][i][param][ta] + (1 - self._beta2) * (grad**2)
							m_corrected = self._params["m"][i][param][ta] / (1. - self._beta1**self._params["t"])
							v_corrected = self._params["v"][i][param][ta] / (1. - self._beta2**self._params["t"])		
							layer._updates[param] += - (self._lr * m_corrected / (np.sqrt(v_corrected) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]
				else:
					for param in layer._params.keys():
						grad = layer._grads[time][param]
						self._params["m"][i][param] = self._beta1 * self._params["m"][i][param] + (1 - self._beta1) * (grad)					
						self._params["v"][i][param] = self._beta2 * self._params["v"][i][param] + (1 - self._beta2) * (grad**2)
						m_corrected = self._params["m"][i][param] / (1. - self._beta1**self._params["t"])
						v_corrected = self._params["v"][i][param] / (1. - self._beta2**self._params["t"])
						layer._updates[param] = - (self._lr * m_corrected / (np.sqrt(v_corrected) + self._eps))
						layer._params[param] = layer._params[param] + layer._updates[param]
