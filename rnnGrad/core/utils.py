from rnnGrad.core.optimizers import *

def get_optimizer(config):
	"""Returns an optimizer object.

	Args:
		config: config dictionary.
	"""
	if config.optim_name == "SGD":
		return SGD(learning_rate=config.lr, momentum=config.momentum)
	elif config.optim_name == "ADAGRAD":
		return ADAGRAD(learning_rate=config.lr, eps=config.eps)
	elif config.optim_name == "RMSprop":
		return RMSprop(learning_rate=config.lr, alpha=config.alpha, eps=config.eps)
	elif config.optim_name == "ADADELTA":
		return ADADELTA(learning_rate=config.lr, gamma=config.gamma, eps=config.eps)
	elif config.optim_name == "ADAM":
		return ADAM(learning_rate=config.lr, beta1 = config.beta1, beta2 = config.beta2, eps=config.eps)

	elif config.optim_name == "WA_RMSprop":
		return WA_RMSprop(learning_rate=config.lr, alpha=config.alpha, eps=config.eps)	
	elif config.optim_name == "torch_WA_RMSprop":
		return torch_WA_RMSprop(learning_rate=config.lr, alpha=config.alpha, eps=config.eps)	
	elif config.optim_name == "torch_ADAM":
		return torch_ADAM(learning_rate=config.lr, beta1 = config.beta1, beta2 = config.beta2, eps=config.eps)
	elif config.optim_name == "WA_ADAM":
		return WA_ADAM(learning_rate=config.lr, beta1 = config.beta1, beta2 = config.beta2, eps=config.eps)
	elif config.optim_name == "torch_WA_ADAM":
		return torch_WA_ADAM(learning_rate=config.lr, beta1 = config.beta1, beta2 = config.beta2, eps=config.eps)

	elif config.optim_name == "WA_ADADELTA":
		return WA_ADADELTA(learning_rate=config.lr, gamma=config.gamma, eps=config.eps)	
	elif config.optim_name == "torch_WA_ADADELTA":
		return torch_WA_ADADELTA(learning_rate=config.lr, gamma=config.gamma, eps=config.eps)			
	elif config.optim_name == "WA_ADAGRAD":
		return WA_ADAGRAD(learning_rate=config.lr, eps=config.eps)
	elif config.optim_name == "torch_WA_ADAGRAD":
		return torch_WA_ADAGRAD(learning_rate=config.lr, eps=config.eps)
	elif config.optim_name == "TA_ADAM":
		return TA_ADAM(learning_rate=config.lr, beta1 = config.beta1, beta2 = config.beta2, eps=config.eps, max_steps=config.max_steps)	
	elif config.optim_name == "TA_RMSprop":
		return TA_RMSprop(learning_rate=config.lr, alpha=config.alpha, eps=config.eps, max_steps=config.max_steps)
	else:
		optimizers = "SGD, ADAGRAD, RMSprop, ADADELTA, ADAM, TA_RMSprop, TA_ADAM ."
		assert("Unsupported optimizer : {}. Valid optimizers : {}".format(config.optim_name, optimizers))

