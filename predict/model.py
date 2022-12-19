import torch
import torchaudio
import os

class Egg(torch.nn.Module):

	def __init__(self):

		super(Egg, self).__init__()

		self.conv1 = torch.nn.Conv1d(2, 2, 1024)
		#self.linear1 = torch.nn.Linear(1353729, 1353729)
		self.conv2 = torch.nn.Conv1d(2, 4, 1024)
		#self.linear2 = torch.nn.Linear(1352706, 1352706)
		self.conv3 = torch.nn.Conv1d(4, 6, 512)
		#self.linear3 = torch.nn.Linear(6, 6)
		self.conv4 = torch.nn.Conv1d(6, 8, 512)
		#self.linear4 = torch.nn.Linear(8, 8)
		self.conv5 = torch.nn.Conv1d(8, 10, 512)
		#self.linear5 = torch.nn.Linear(53229, 53229)
		self.conv6 = torch.nn.Conv1d(10, 12, 256)
		self.linear6 = torch.nn.Linear(10439, 10439)
		self.conv7 = torch.nn.Conv1d(12, 14, 256)
		self.linear7 = torch.nn.Linear(10184, 10184)
		self.conv8 = torch.nn.Conv1d(14, 16, 256)
		self.linear8 = torch.nn.Linear(9929, 9929)
		self.conv9 = torch.nn.Conv1d(16, 18, 256)
		self.linear9 = torch.nn.Linear(3054, 3054)
		self.conv10 = torch.nn.Conv1d(18, 19, 256)
		self.linear10 = torch.nn.Linear(2799, 2799)
		self.conv11 = torch.nn.Conv1d(19, 20, 200)
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
	#	x = self.linear1(x)
		x = self.dropout(x)
		x = self.conv2(x)
	#	x = self.linear2(x)
		x = self.conv3(x)
		x = self.pool5(x)
		#x = self.linear3(x)
		x = self.conv4(x)
		#x = self.linear4(x)
		x = self.batch16(x)
		x = self.pool5(x)
		x = self.conv5(x)
		#x = self.linear5(x)
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


egg = Egg()


