import torch
import torchaudio
from model import Egg
import math
class Song():

	def __init__(self, path, name):
		self.path = path
		self.name = name

		with torch.no_grad():

			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

			model = Egg()
			model.to(self.device)
			model.load_state_dict(torch.load("weights/egg_weights"))

			audio, _ = torchaudio.load(path)

			self.audio = audio

			#n = math.ceil(audio.size(dim=1) / 1354752)
			#print(n)

			inputs = list(torch.split(audio, 1354752, 1))

			n_len = 1354752 - inputs[-1].size(dim=1)

			#m = torch.nn.ConstantPad1d(n_len / 2, 0)
			#del inputs[-1]
			#print(inputs[-1].size())
			#inputs[-1] = m(inputs[-1])
			#print(inputs[-1].size())
			del inputs[-1]
			for x in inputs:
				print(x.size())
			outputs = []

			for i in inputs:
				print(i.size())
				i = torch.reshape(i, (1, 2, 1354752))
				prediction = model(i.to(self.device))
				prediction = prediction.to("cpu")

				outputs = outputs + [prediction]
			print(outputs)
			self.embeddings = torch.stack(outputs).resize(len(outputs), 20, 300)

	def distance(self, emb):
		values = []
		for x in self.embeddings:
			x_values = []
			for y in emb:
				x_values.append((x - y).pow(2).sum().sqrt())
			values.append(x_values)
		x = 0
		sum_col = 0
		while x < len(values):
			column = [col[x] for col in values]
			sum_col += min(column)
			del values[column.index(min(column))]
			x += 1
		return sum_col