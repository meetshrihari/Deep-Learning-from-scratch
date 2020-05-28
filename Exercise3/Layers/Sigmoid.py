import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input_tensor):
        self.fwd_act = 1/(1 + np.exp(-input_tensor))
        return self.fwd_act

    def backward(self, error_tensor):
        self.back_act = self.fwd_act * (1-self.fwd_act) * error_tensor
        return self.back_act