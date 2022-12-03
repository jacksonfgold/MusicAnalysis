import torch
import torchaudio
import os


class Egg(torch.nn.Module):

	def __init__(self):
		super(Egg, self).__init__()

		self.conv1 = torch.nn.Conv1d(1, 4, kernal_size=1024)

		self.conv2 = torch.nn.Conv1d(1, 4, kernal_size=1024)

		self.conv2 = torch.nn.Conv1d(50, 50, )


