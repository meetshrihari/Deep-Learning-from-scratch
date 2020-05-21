
import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer, ):
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
            self.loss.append(self.forward())
            self.backward()
        return self.loss

    def append_trainable_layer(self, layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)


    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)


#        predict = self.loss_layer.forward(input_tensor)
        return input_tensor#predict
        

"""
class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer, ):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()
            self.loss.append(self.loss_layer.loss)
        return self.loss

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.predict(input_tensor)

    def append_trainable_layer(self, layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)
"""