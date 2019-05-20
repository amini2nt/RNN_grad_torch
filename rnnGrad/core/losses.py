import torch


class l2_loss(object):
	"""Implementation of L2 loss."""

	def __init__(self, average='mean'):
		"""Initializes the L2 loss.

		Args:
			average: str, 'none' for not averaging the loss over minibatch,
					 'mean' for averaging, and 'sum' for summing.
		"""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}
		self._average = average
		
	def forward(self, pred, target, time=1):
		"""forward prop through the loss.

		Args:
			pred: torch array of size (batch size, num_outputs), model's prediction.
			target: torch array of size (batch size, num_outputs), target prediction.
			time: int, time step

		returns:
			loss which is a matrix if average is 'none', a scalar otherwise.
		"""
		self._pred[time] = pred
		self._target[time] = target
		loss = (pred - target)**2
		if self._average == 'none':
			self._output[time] = loss
		elif self._average == 'sum':
			self._output[time] = torch.sum(loss)
		elif self._average == 'mean':
			self._output[time] = torch.mean(loss)
		return self._output[time]

	def backward(self, time=1):
		"""backward prop through the loss.
		
		Args:
			time: int, time step

		returns:
			gradient matrix of size (num_outputs, batch_size)
		"""
		self._grad[time] = torch.t((self._pred[time] - self._target[time]))*2
		if self._average == 'mean':
			self._grad[time] /= self._pred[time].numel()
		return self._grad[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}


class bce_loss(object):
	"""Implementation of binary cross entropy loss."""

	def __init__(self, average='mean'):
		"""Initializes the BCE loss.

		Args:
			average: str, 'none' for not averaging the loss over minibatch,
					 'mean' for averaging, and 'sum' for summing.
		"""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}
		self._average = average
		
	def forward(self, pred, target, time=1):
		"""forward prop through the loss.

		Args:
			pred: torch array of size (batch size, num_outputs), model's prediction.
			target: torch array of size (batch size, num_outputs), target prediction.
			time: int, time step

		returns:
			loss which is a matrix if average is 'none', a scalar otherwise.
		"""
		self._pred[time] = pred
		self._target[time] = target
		loss = -target * torch.log(pred) - (1-target) * torch.log(1-pred)
		if self._average == 'none':
			self._output[time] = loss
		elif self._average == 'sum':
			self._output[time] = torch.sum(loss)
		elif self._average == 'mean':
			self._output[time] = torch.mean(loss)
		return self._output[time]

	def backward(self, time=1):
		"""backward prop through the loss.

		Args:
			time: int, time step

		returns:
			gradient matrix of size (num_outputs, batch_size)
		"""
		self._grad[time] = (-self._target[time]/(self._pred[time])) 
		self._grad[time] += (1-self._target[time])/(1-self._pred[time])
		self._grad[time] = torch.t(self._grad[time])
		if self._average == 'mean':
			self._grad[time] /= self._pred[time].numel()
		return self._grad[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}
		

class bce_with_logits_loss(object):
	"""Implementation of binary cross entropy loss with logits."""

	def __init__(self, average='mean'):
		"""Initializes the BCE loss with logits.

		Args:
			average: str, 'none' for not averaging the loss over minibatch,
					 'mean' for averaging, and 'sum' for summing.
		"""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}
		self._average = average
		
	def forward(self, pred, target, time=1):
		"""forward prop through the loss.

		Args:
			pred: torch array of size (batch size, num_outputs), model's prediction.
			target: torch array of size (batch size, num_outputs), target prediction.
			time: int, time step

		returns:
			loss which is a matrix if average is 'none', a scalar otherwise.
		"""
		self._pred[time] = pred
		self._target[time] = target
		loss = target * (torch.log(torch.exp(torch.zeros_like(pred)) + torch.exp(-pred))) 
		loss += (1-target) * (torch.log(torch.exp(torch.zeros_like(pred)) + torch.exp(pred)))
		if self._average == 'none':
			self._output[time] = loss
		elif self._average == 'sum':
			self._output[time] = torch.sum(loss)
		elif self._average == 'mean':
			self._output[time] = torch.mean(loss)
		return self._output[time]

	def backward(self, time=1):
		"""backward prop through the loss.

		Args:
			time: int, time step

		returns:
			gradient matrix of size (num_outputs, batch_size)
		"""
		self._grad[time] = torch.t((torch.sigmoid(self._pred[time]) - self._target[time]))
		if self._average == 'mean':
			self._grad[time] /= self._pred[time].numel()
		return self._grad[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}


class ce_with_logits_loss(object):
	"""Implementation of cross entropy loss with logits."""

	def __init__(self, average='mean'):
		"""Initializes the CE loss with logits.
		Args:
			average: str, 'none' for not averaging the loss over minibatch,
					 'mean' for averaging, and 'sum' for summing.
		"""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}
		self._average = average
		
	def forward(self, pred, target, time=1):
		"""forward prop through the loss.
		Args:
			pred: torch array of size (batch size, num_outputs), model's prediction.
			target: torch array of size (batch size), target prediction ids.
			time: int, time step
		returns:
			loss which is a matrix if average is 'none', a scalar otherwise.
		"""
		self._pred[time] = pred
		self._target[time] = target
		logits_for_answers = pred[torch.arange(len(self._pred[time])), target]
		loss = - logits_for_answers + torch.log(torch.sum(torch.exp(pred), dim=-1))
		if self._average == 'none':
			self._output[time] = loss
		elif self._average == 'sum':
			self._output[time] = torch.sum(loss)
		elif self._average == 'mean':
			self._output[time] = torch.mean(loss)
		return self._output[time]

	def backward(self, time=1):
		"""backward prop through the loss.
		Args:
			time: int, time step
		returns:
			gradient matrix of size (num_outputs, batch_size)
		"""
		ones_for_answers = torch.zeros_like(self._pred[time])
		ones_for_answers[torch.arange(len(self._pred[time])), self._target[time]] = 1
		softmax = torch.exp(self._pred[time])/ torch.exp(self._pred[time]).sum(dim=1, keepdim=True)
		self._grad[time] = torch.t(-ones_for_answers + softmax)
		if self._average == 'mean':
			self._grad[time] /= self._pred[time].shape[0]
		return self._grad[time]

	def reset_computation(self):
		"""resets the layer computations."""
		self._pred = {}
		self._target = {}
		self._output = {}
		self._grad = {}