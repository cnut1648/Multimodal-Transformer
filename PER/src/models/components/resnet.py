import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18


class ResNet2Plus1D(nn.Module):
	def __init__(self, dropout_rate=0.1):
		super(ResNet2Plus1D, self).__init__()

		r21d = r2plus1d_18(pretrained=True)
		r21d_layers = list(r21d.children())[:-1]
		self.model = nn.Sequential(*r21d_layers)

	def forward(self, x): # [B, F, C, H, W]
		x = x.permute(0,2,1,3,4) # [B, C, F, H, W]
		x = self.model(x)
		x = x.view(x.size(0), -1)

		return x
