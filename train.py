import pickle
from loadData import Dataset
import math
import torch
import numpy as np
from model import Egg
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

with open("data/song_tensors.pkl", "rb") as f:
	x = pickle.load(f)

with open("data/song_labels.pkl", "rb") as f:
	y = pickle.load(f)

batch_size = 32
train_ratio = 0.8
max_words = 20
load_model = False


data_len = len(x)

x_train = x[:math.floor(data_len * train_ratio)]
x_test = x[math.floor(data_len * train_ratio):]

y_train = y[:math.floor(data_len * train_ratio)]
y_test = y[math.floor(data_len * train_ratio):]

train_data = Dataset(x_train, y_train)
test_data = Dataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

print('Training set has {} instances'.format(len(train_data)))
print('Test set has {} instances'.format(len(test_data)))

mse = torch.nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Egg()

if load_model:
	model.load_state_dict("weights/egg_weights")

model.to(device)

def loss_func(predict, label):

	def single_loss(p, l):
		values = []
		for x in p:
			x_values = []
			for y in l:
				x_values.append(mse(x,y))
			values.append(x_values)

		x = 0
		sum_col = 0
		while x < max_words:
			column = [col[x] for col in values]
			col_min = min(column)
			sum_col += col_min
			del values[column.index(col_min)]
			x += 1
			#print(sum_col)
		return sum_col


	losses = []

	i = 0

	while i < len(predict):
		losses.append(single_loss(predict[i], label[i]))
		i += 1

	return torch.stack(losses, dim=0).sum(dim=0).sum(dim=0)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        #print(len(inputs), len(labels))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_func(outputs.float(), labels.float())
        #print(loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/music_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000

for epoch in range(EPOCHS):
	with torch.cuda.device(0):
		print('EPOCH {}:'.format(epoch_number + 1))

	    # Make sure gradient tracking is on, and do a pass over the data
		model.train(True)
		avg_loss = train_one_epoch(epoch_number, writer)
		torch.cuda.empty_cache()

	    # We don't need gradients on to do reporting
		model.train(False)

		with torch.no_grad():
			running_vloss = 0.0
			for i, vdata in enumerate(test_loader):
				vinputs, vlabels = vdata
				vinputs = vinputs.to(device=device)
				vlabels = vlabels.to(device=device)
				voutputs = model(vinputs)
				vloss = loss_func(voutputs.float(), vlabels.float())
				running_vloss += vloss

		avg_vloss = running_vloss / (i + 1)
		print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

	    #Log the running loss averaged per batch
	    # for both training and validation
		writer.add_scalars('Training vs. Validation Loss',
						{ 'Training' : avg_loss, 'Validation' : avg_vloss },
						epoch_number + 1)
		writer.flush()

	    # Track best performance, and save the model's state
		if avg_vloss < best_vloss:
			best_vloss = avg_vloss
		model_path = 'weights/model_{}_{}'.format(timestamp, epoch_number)
		torch.save(model.state_dict(), model_path)

		epoch_number += 1