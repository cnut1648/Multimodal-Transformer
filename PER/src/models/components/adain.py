import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm2d(nn.Module):
	def __init__(self, num_features, affine=True):
		super(AdaptiveInstanceNorm2d, self).__init__()

		self.num_features = num_features
		self.norm = nn.LayerNorm(num_features)
		# gamma and beta are dynamically assigned
		self.gamma = None
		self.beta = None

	def forward(self, x):
		assert self.gamma is not None and self.beta is not None, 'Please assign gamma and beta before calling AdaIn!'

		x = self.norm(x)
		x = x*self.gamma+self.beta

		return x
