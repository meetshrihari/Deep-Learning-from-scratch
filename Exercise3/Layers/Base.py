import numpy as np
from enum import Enum as E
"""from Layers import FullyConnected
from Layers import Conv
from Layers import LSTM

All layers need to inherit
from this ”base-layer” so refactor them accordingly
"""

class Base_Layer():
    def __init__(self):
        self.regularizer = None
        self.phase = Phase.train    # train as default

    def calculate_regularization_loss(self):
        reg_loss = 0
        if self.weights is not None:
            if self.regularizer is not None:
                reg_loss = self.regularizer.norm(self.weights)
        return reg_loss
        
    
class Phase(E):
    train = 'some'
    test = 'asdf'
    validation = 'sadf'





