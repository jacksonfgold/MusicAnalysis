import torch
import torchaudio

class Dataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, index):
		return self.x[index], torch.stack(self.y[index])

	def __len__(self):
		return len(self.x)


