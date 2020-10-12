import torch as t
from trainer import Trainer
import sys
import torchvision as tv

epoch = int(sys.argv[1])
print(epoch)
import model
model = model.ResNet()

# #TODO: Enter your model here

crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
