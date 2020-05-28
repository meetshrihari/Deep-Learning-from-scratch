import numpy as np


class TanH:
    def __init__(self):
        self.fwd_act = None
        self.back_act = None
    
    def forward(self, input_tensor):
        self.fwd_act = np.tanh(input_tensor)
        return self.fwd_act
    
    def backward(self, error_tensor):
        self.back_act = (1 - np.power(self.fwd_act, 2)) * error_tensor
        return self.back_act






