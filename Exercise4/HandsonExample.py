import torch
# -------------------------------------------------------
# Part 1
x = torch.ones(2, 2, requires_grad=True)
y = x / 2
L = torch.sum(y)
L.backward()
print(x.grad)

# -------------------------------------------------------
# Part 2
x = torch.tensor([[2, 2],[1, 1]], requires_grad=True, dtype=torch.float32)
y = x * x + 1
L = y.sum()
L.backward()
print(x.grad)

# -------------------------------------------------------
# Part 3
X = torch.rand(10000,10000)
W = torch.rand(10000,50, requires_grad=True)
Y = X@W
L = Y.sum()
print(L)
# optimization
L.backward()
W = W - W.grad * 0.000001
L = (X@W).sum()
print(L)
# -------------------------------------------------------
# Part 4
# running things on gpu

import time
class Timer:
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


with Timer() as t:
    c = torch.device('cuda') # set 'cuda' or 'cpu'
    X = torch.rand(100000,10000, device= c)
    W = torch.rand(10000,500, requires_grad=True, device= c)
    Y = X@W
    L = Y.sum()
    L.backward()
    W = W - W.grad * 0.000001
    L = (X@W).sum()

# print(t.interval)

# -------------------------------------------------------
# Part 5 - Inventing new Layers
import numpy as np
class OurSin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        # ineficient but torch supports multi processing or cuda
        ctx.sin_input = tensor.cpu().detach().numpy()
        tmp = np.sin(ctx.sin_input)
        return torch.tensor(tmp, dtype=torch.double).cuda()

    @staticmethod
    def backward(ctx, error_tensor):
        tmp_grad = np.cos(ctx.sin_input)
        return error_tensor * torch.tensor(tmp_grad, dtype=torch.double).cuda()

from torch.autograd import gradcheck
input = torch.randn(20,20,dtype=torch.double, requires_grad=True)
test = gradcheck(OurSin.apply, input.cuda(), eps=1e-5, atol=1e-6)
print(test)

# -------------------------------------------------------
# Part 6
# Here we go!
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision as tv
import operator

data = tv.datasets.FashionMNIST(root="data", download=True)
batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(data))), 100, False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv2 = nn.Conv2d(100, 50, 5)
        self.conv3 = nn.Conv2d(50, 5, 5)
        self.fc1 = nn.Linear(5 * 256, 100)
        self.fc2 = nn.Linear(100, 10)
        self.sin = OurSin.apply # if in use, OurSin must be changed into float32

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.fc1(x.flatten(start_dim = 1)))
        return self.fc2(x)

model = Model()
model.cuda()
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

for e in range(100):
    for idx in batch_sampler:
        batch = operator.itemgetter(*idx)(data)
        x = []
        y_true = []
        for image, label in batch:
            x.append(np.asarray(image))
            y_true.append(label)
        x = np.stack(x)[:, np.newaxis]
        y_true = np.stack(y_true)
        y_pred = model(torch.tensor(x, dtype=torch.float).cuda())
        l = loss(y_pred, torch.tensor(y_true).cuda())
        optim.zero_grad()
        l.backward()
        optim.step()