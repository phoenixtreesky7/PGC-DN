from torch import nn
import torch
from torch.nn import functional as F 

class RB(nn.Module):
	def __init__(self, inchannel, outchannel):
		super(RB, self).__init__()
		module = []
		module += [nn.Conv2d(inchannel, outchannel//4, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(outchannel//4), nn.ReLU(True)]
		module += [nn.Conv2d(outchannel//4, outchannel//4, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(outchannel//4), nn.ReLU(True)]
		module += [nn.Conv2d(outchannel//4, outchannel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(outchannel)]

		self.body = nn.Sequential(*module)

		self.relu = nn.ReLU(True)

	def forward(self, input):
		out = self.body(input)
		out = self.relu(out + input)

		return out

class ResNetBlock(nn.Module):
	def __init__(self, inchannel, outchannel):
		super(ResNetBlock, self).__init__()
		model = []
		model += [nn.Conv2d(inchannel, outchannel//4, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(outchannel//4), nn.ReLU(True)]
		model += [nn.Conv2d(outchannel//4, outchannel//4, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(outchannel//4), nn.ReLU(True)]
		model += [nn.Conv2d(outchannel//4, outchannel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(outchannel)]

		self.body = nn.Sequential(*model)
		self.relu = nn.ReLU(True)

	def forward(self, input):
		out = self.body(input)
		out = self.relu(out + input)
		return out


class ResNetGroup(nn.Module):
	def __init__(self, inchannel, outchannel, n_block):
		super(ResNetGroup, self).__init__()
		resnetblock = []
		for nb in range(n_block):
			resnetblock += [ResNetBlock(inchannel, outchannel)]

		self.rng = nn.Sequential(*resnetblock)

	def forward(self, input):
		out = rng(input)
		return out


