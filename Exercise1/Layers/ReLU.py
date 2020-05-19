import numpy as np


class ReLU:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        out = input_tensor * (input_tensor > 0)  # only greater than 0
        self.input_tensor = input_tensor
        return out

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)  # > 0


