
import numpy as np
import Optimization.Optimizers as opt
from copy import *


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size  # 4   self.batch_size = 9
        self.output_size = output_size  # 3
        self.input_tensor = None  #
        self.weights = np.random.rand(input_size+1, output_size)  # +1 for bias
        self._optimizer = None

    def forward(self, input_tensor):
        bias = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.concatenate((input_tensor, bias), axis=1)
        self.next = np.dot(input_tensor, self.weights)
        self.input_tensor = input_tensor
        return self.next   # shape = (9,3)

    def setter(self, optimizer):
        self._optimizer = optimizer

    def getter(self):
        return self._optimizer

    optimizer = property(getter, setter)

    # gradient with respect to input:  En-1 = En*W.T
    # Update W using gradient with respect to W:  Wt+1 = W.T + n*En*X.T
    def backward(self, error_tensor):
        self.error = error_tensor
        # refer chapter 2 slide 39
        correct_weight = self.weights[0: self.weights.shape[0]-1, :].T
        gradient_input = np.dot(error_tensor, correct_weight)
        gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self.gradient_wt = gradient_weights
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_weights)

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(np.shape(self.weights[:-1, :]), np.shape(self.weights[:-1, :])[0],
                                                 np.shape(self.weights[:-1, :])[1])
        bias = np.expand_dims(self.weights[-1, :], axis=0)
        bias = bias_initializer.initialize(bias.shape, bias.shape[0], bias.shape[1])
        self.weights = np.concatenate((weights, bias), axis=0)
        return self.weights, bias

    @property
    def gradient_weights(self):
        self.gradient = np.dot(self.input_tensor.T, self.error)
        return self.gradient

    def set_optimizer(self, optimizer):
        self.optimizer = deepcopy(optimizer)
        #return self.optimizer
