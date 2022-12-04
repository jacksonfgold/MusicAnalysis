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

batch_size = 1
train_ratio = 0.8

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

model = Egg()

def loss_func(predict, label):
	print(len(predict), len(label))
	values = []
	for x in predict[0]:
		x_values = []
		for y in label:
			x_values.append(mse(x,y))
		values.append(x_values)

	x = 0
	sum_col = 0
	while x < 50:
		column = [col[x] for col in values]
		sum_col += min(column)
		del values[column.index(min(column))]
		x += 1
	return torch.tensor(sum_col)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.requires_grad = True
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/music_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000

for epoch in range(EPOCHS):
	with torch.cuda.device(0):
	    print('EPOCH {}:'.format(epoch_number + 1))

	    # Make sure gradient tracking is on, and do a pass over the data
	    model.train(True)
	    avg_loss = train_one_epoch(epoch_number, writer)

	    # We don't need gradients on to do reporting
	    model.train(False)

	    running_vloss = 0.0
	    for i, vdata in enumerate(test_loader):
	        vinputs, vlabels = vdata
	        voutputs = model(vinputs)
	        vloss = loss_fn(voutputs, vlabels)
	        running_vloss += vloss

	    avg_vloss = running_vloss / (i + 1)
	    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

	    # Log the running loss averaged per batch
	    # for both training and validation
	    writer.add_scalars('Training vs. Validation Loss',
	                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
	                    epoch_number + 1)
	    writer.flush()

	    # Track best performance, and save the model's state
	    if avg_vloss < best_vloss:
	        best_vloss = avg_vloss
	        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
	        torch.save(model.state_dict(), model_path)

	    epoch_number += 1