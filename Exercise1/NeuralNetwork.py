import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None  # provide input data and labels
        self.loss_layer = None  # softmax layer provides loss and prediction
        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
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
        for wt in np.arange(iterations):
            #self.forward()
            self.loss.append(self.forward())
            self.backward()

    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
        # layer. np.copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        # layer.set_optimizer(copy.deepcopy(self.optimizer))

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

#        predict = self.loss_layer.forward(input_tensor)
        return input_tensor#predict
