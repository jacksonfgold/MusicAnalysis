import torch
import torchaudio
import os

class Egg(torch.nn.Module):

	def __init__(self):

		super(Egg, self).__init__()

		self.conv1 = torch.nn.Conv1d(2, 2, 1024)
		self.conv2 = torch.nn.Conv1d(2, 4, 1024)
		self.conv3 = torch.nn.Conv1d(4, 6, 512)
		self.conv4 = torch.nn.Conv1d(6, 8, 512)
		self.conv5 = torch.nn.Conv1d(8, 10, 512)
		self.conv6 = torch.nn.Conv1d(10, 12, 256)
		self.linear6 = torch.nn.Linear(10390, 10390)
		self.conv7 = torch.nn.Conv1d(12, 14, 256)
		self.linear7 = torch.nn.Linear(10135, 10135)
		self.conv8 = torch.nn.Conv1d(14, 16, 256)
		self.linear8 = torch.nn.Linear(9880, 9880)
		self.conv9 = torch.nn.Conv1d(16, 18, 256)
		self.linear9 = torch.nn.Linear(3038, 3038)
		self.conv10 = torch.nn.Conv1d(18, 19, 256)
		self.linear10 = torch.nn.Linear(2783, 2783)
		self.conv11 = torch.nn.Conv1d(19, 20, 192)
		self.linear11 = torch.nn.Linear(1200, 1200)
		self.pool2 = torch.nn.AvgPool1d(2)
		self.pool3 = torch.nn.AvgPool1d(3)
		self.pool5 = torch.nn.AvgPool1d(5)
		self.pool28 = torch.nn.AvgPool1d(4)
		self.dropout = torch.nn.Dropout(0.8)
		self.batch16 = torch.nn.BatchNorm1d(8)
		self.batch40 = torch.nn.BatchNorm1d(16)
		self.batch48 = torch.nn.BatchNorm1d(19)

	def forward(self, x):

		x = self.conv1(x)
		x = self.pool5(x)
		x = self.dropout(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.batch16(x)
		x = self.pool5(x)
		x = self.conv5(x)
		x = self.pool5(x)
		x = self.conv6(x)
		x = self.linear6(x)
		x = self.dropout(x)
		x = self.conv7(x)
		x = self.linear7(x)
		x = self.conv8(x)
		x = self.linear8(x)
		x = self.batch40(x)
		x = self.pool3(x)
		x = self.conv9(x)
		x = self.linear9(x)
		x = self.conv10(x)
		x = self.linear10(x)
		x = self.pool2(x)
		x = self.batch48(x)
		x = self.conv11(x)
		x = self.linear11(x)
		x = self.pool28(x)

		return x

