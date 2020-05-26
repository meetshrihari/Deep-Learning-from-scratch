from Layers import *

from copy import *
import pickle

from Optimization import *
import os
import numpy as np
import copy
from Layers.Base import Phase

"""
FOR RNN
1.Implement a property phase in the NeuralNetwork class setting each layer’s phase.
Use this method to set the phase in the train and test methods.
2.Refactor the NeuralNetwork class to add the regularization loss to the data loss. Use
the method norm(weights) and sum up the regularization loss created by the weights of
all layers. Hint: You might want to refactor more classes to get the necessary information
using base-classes.

"""


class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = list()
        self.layers = list()
        self.data_layer = None  # provide input data and labels
        self.loss_layer = None  # softmax layer provides loss and prediction
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        #self.phase(Phase.train)
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        current = self.input_tensor
        for layer in self.layers:
            current = layer.forward(current)

        out = self.loss_layer.forward(current, self.label_tensor)
        return out

    def backward(self):
        e = self.loss_layer.backward(self.label_tensor)     # from Softmax
        for layer in reversed(self.layers):
            e = layer.backward(e)   # FC

    def train(self, iterations):
        self.phase(Phase.train)
        
        for wt in np.arange(iterations):
            self.loss.append(self.forward())
            self.backward()
        return self.loss

    def append_trainable_layer(self, layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def test(self, input_tensor):
        self.phase(Phase.test)
        
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
    
    def phase(self, phase):
        for layers in self.layers:
            layers.phase = phase

    def get_phase(self):
        return self._phase

    def set_phase(self, ph):
        self._phase = ph