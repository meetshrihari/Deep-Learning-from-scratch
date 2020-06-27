import torch as t
import torchvision as tv
from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import resnet
from checkpoints import *

# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
train_dl = t.utils.data.DataLoader(get_train_dataset(), batch_size=200, shuffle=True)
val_dl =  t.utils.data.DataLoader(get_validation_dataset(), batch_size=200)

# set up your model
model = resnet.ResNet()

# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
# set up optimizer (see t.optim); 
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
posweight = t.squeeze(get_train_dataset().pos_weight())
lossfunc = t.nn.BCEWithLogitsLoss(pos_weight=posweight)
Optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.135)
earlystopper = EarlyStoppingCallback(70)

trainerob = Trainer(model, lossfunc, Optimizer, train_dl, val_dl, True, earlystopper)

# go, go, go... call fit on trainer
#trainerob.restore_checkpoint(7)
res = trainerob.fit(800)  #TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

