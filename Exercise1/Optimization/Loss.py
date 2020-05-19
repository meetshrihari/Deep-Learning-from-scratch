import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        temp = np.log(input_tensor + np.finfo('float').eps)
        loss = np.sum(-temp*label_tensor)
        self.input = input_tensor
        return loss

    def backward(self, label_tensor):
        loss = -(np.divide(label_tensor, self.input))
        return loss

