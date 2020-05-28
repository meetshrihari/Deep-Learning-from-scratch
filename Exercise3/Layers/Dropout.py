import numpy as np
from Layers.Base import Phase
import itertools

"""
Dropout most often used to regularize fully connected layers.
It enforces independent weights, reducing the effect of co-adaptation.
"""


class Dropout:
    def __init__(self, probability):
        self.probability = probability
        self.phase = Phase.train
        self.dropout = None

    def forward(self, input_tensor):
        input_size = list(input_tensor.shape)
        # enable all the connections
        self.dropout = np.ones(input_tensor.shape)
        for i, j in itertools.product(range(input_size[0]), range(input_size[1])):
            random_p = np.random.random(1)
            # drop the connects w.r.t probability
            if random_p >= self.probability:
                self.dropout[i, j] = 0

        output_tensor = self.dropout * input_tensor   #do dropout
        if self.phase is Phase.train:
            output_tensor = output_tensor/self.probability
        else:  # test
            output_tensor = input_tensor

        return output_tensor

    def backward(self, error_tensor):
        if self.phase is Phase.train:
            et = self.dropout * error_tensor
        else:
            et = error_tensor

        return et       # error tensor output




