import torch
import torchaudio
import os

class Egg(torch.nn.Module):

	def __init__(self):

		super(Egg, self).__init__()

		self.conv1 = torch.nn.Conv1d(2, 4, 1024)
		self.conv2 = torch.nn.Conv1d(4, 8, 1024)
		self.conv3 = torch.nn.Conv1d(8, 12, 512)
		self.conv4 = torch.nn.Conv1d(12, 16, 512)
		self.conv5 = torch.nn.Conv1d(16, 20, 512)
		self.conv6 = torch.nn.Conv1d(20, 24, 256)
		self.conv7 = torch.nn.Conv1d(24, 32, 256)
		self.conv8 = torch.nn.Conv1d(32, 40, 256)
		self.conv9 = torch.nn.Conv1d(40, 44, 256)
		self.conv10 = torch.nn.Conv1d(44, 48, 256)
		self.conv11 = torch.nn.Conv1d(48, 50, 157)
		self.pool2 = torch.nn.MaxPool1d(2)
		self.pool3 = torch.nn.MaxPool1d(3)
		self.pool5 = torch.nn.MaxPool1d(5)
		self.pool28 = torch.nn.MaxPool1d(28)
		self.dropout = torch.nn.Dropout(0.8)

	def forward(self, x):

		x = self.conv1(x)
		x = self.pool5(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.pool5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.pool3(x)
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.pool2(x)
		x = self.conv11(x)
		x = self.pool28(x)

		return x


egg = Egg()


