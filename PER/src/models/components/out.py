import torch
import torch.nn as nn
import torch.nn.functional as F

from .adain import AdaptiveInstanceNorm2d

class RegOut(nn.Module):
	def __init__(self, dim, dropout):
		super().__init__()

		self.layernorm = nn.LayerNorm(dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(dim, dim)
		self.fc2 = nn.Linear(dim, 2)

	def forward(self, x):
		x = self.layernorm(x)
		x = self.dropout(self.fc1(x))
		x = self.relu(x)
		x = self.dropout(self.fc2(x))
		x = torch.tanh(x)

		return x

class ClfOut(nn.Module):
	def __init__(self, dim, dropout, num_classes):
		super().__init__()

		self.layernorm = nn.LayerNorm(dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(dim, dim)
		self.fc2 = nn.Linear(dim, num_classes)

	def forward(self, x):
		x = self.layernorm(x)
		x = self.dropout(self.fc1(x))
		x = self.relu(x)
		x = self.dropout(self.fc2(x))

		return x

class AdaInRegOut(nn.Module):
	def __init__(self, dim, dropout):
		super().__init__()

		self.norm1 = nn.AdaptiveInstanceNorm2d(dim)
		self.norm2 = nn.AdaptiveInstanceNorm2d(dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(dim, dim)
		self.fc2 = nn.Linear(dim, 2)

	def forward(self, x):
		x = self.norm1(x)
		x = self.dropout(self.fc1(x))
		x = self.relu(x)
		x = self.norm2(x)
		x = self.dropout(self.fc2(x))
		x = torch.tanh(x)

		return x

class AdaInClfOut(nn.Module):
	def __init__(self, dim, dropout, num_classes):
		super().__init__()

		self.norm1 = AdaptiveInstanceNorm2d(dim)
		self.norm2 = AdaptiveInstanceNorm2d(dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(dim, dim)
		self.fc2 = nn.Linear(dim, num_classes)

	def forward(self, x):
		x = self.norm1(x)
		x = self.dropout(self.fc1(x))
		x = self.relu(x)
		x = self.norm2(x)
		x = self.dropout(self.fc2(x))

		return x
