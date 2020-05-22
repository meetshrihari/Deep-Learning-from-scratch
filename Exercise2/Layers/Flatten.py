import numpy as np


class Flatten:
    def __init__(self):
        pass

    def forward(self, input_tensor):   # input_tensor.shape = (9, 3, 4, 11)
        # convert (9, 3, 4, 11) to (9, 132)
        self.input = input_tensor
        self.fwd_1d = np.reshape(input_tensor, (input_tensor.shape[0], -1))  # (#,-1) take any column with # rows
        #print(self.fwd_1d)
        return self.fwd_1d   #.shape = (9, 132)

    def backward(self, error_tensor):  # error_tensor.shape = (9, 132)
        # convert (9, 132) size to (9, 3, 4, 11)
        self.bkwd_2d = np.reshape(error_tensor, self.input.shape)  # (self.fwd_1d.shape[0], -1)
        #print(self.bkwd_2d.shape)
        return self.bkwd_2d
