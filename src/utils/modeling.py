import torch.nn.init as init
from torch import nn
import math
from functools import reduce

def weights_init(init_type='gaussian'):
	def init_fun(m):
		classname = m.__class__.__name__
		if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
			if init_type == 'gaussian':
				init.normal_(m.weight.data, 0.0, 0.02)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=math.sqrt(2))
			elif init_type == 'default':
				pass
			else:
				assert 0, "Unsupported initialization: {}".format(init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)

	return init_fun

def freeze(module: nn.Module):
	for param in module.parameters():
		param.requires_grad = False

def unfreeze(module: nn.Module):
	for param in module.parameters():
		param.requires_grad = True
	
def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)