import numpy as np
from Layers.Base import Base_Layer
from Layers.Base import Phase
from Layers.Helpers import *
from Layers.TanH import *


class BatchNormalization(Base_Layer):
    def __init__(self, channels=0):
        
        super().__init__()
        self.channels = channels
        
        self.weights = []
        self.bias = []
        self.alpha = 0.8
        self.mean = 0
        self.variance = 1
        self.test_mean = 0
        self.test_variance = 1
        
        self.optimizer = None
        self.weights_initializer = None
        self.bias_initializer = None
    
    def initialize(self, weights_initializer, bias_initializer):
        # initializing weights and biases
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
    
    def reformat(self, tensor):  # (5, 3, 6, 4) ; (120, 3)
        # image2vector
        if len(tensor.shape) == 4:
            # setting the channels as the first dimension
            # np.transpose((1, 2, 3), (1, 0, 2)).shape -->   (2, 1, 3)
            # B × H × M × N tensor to B × M × N × H to B · M · N × H
            tensor = np.transpose(tensor, [0, 2, 3, 1])
            # reshaping to a vector
            new_ten = np.reshape(tensor, (tensor.shape[0] * tensor.shape[1] * tensor.shape[2],
                                          tensor.shape[3]), order="C")
        
        # vector2image --> useless approach
        elif (self.input_shape[0] == 200 or self.input_shape[0] == 50 or self.input_shape[0] == 150 or self.input_shape[0] == 148 or self.input_shape[0] == 599) and len(self.input_shape) == 2:  # tensor shape --> (120,3), (200,18)
            # print(self.input_shape)
            shape = (self.input_shape[0], self.input_shape[1])
            new_tensor = np.reshape(tensor, shape, order="C")
            new_ten = np.transpose(new_tensor, [0, 1])
        
        else:
            # print(self.input_shape)
            shape = (self.input_shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1])
            new_tensor = np.reshape(tensor, shape, order="C")
            # print(new_tensor.shape)
            new_ten = np.transpose(new_tensor, [0, 3, 1, 2])
        
        return new_ten
    
    # *********************************************************
    def forward(self, input_tensor):
        # print(input_tensor.shape)
        epsilon = 1e-20
        self.input_shape = input_tensor.shape  # (200,18)
        
        # reshaping for the convolutional case
        if self.channels > 0:
            input_tensor = self.reformat(input_tensor)
        
        # initializing parameters the first forward pass
        self.initialize_wt_bias(input_tensor)
        
        # forward pass for the training
        if self.phase == Phase.train:
            # calculating new train mean and variance
            train_mean = np.mean(input_tensor, axis=0)
            train_var = np.var(input_tensor, axis=0)
            
            # computing test mean and variance through MA
            self.test_mean = self.alpha * self.mean + (1 - self.alpha) * train_mean
            self.test_variance = self.alpha * self.variance + (1 - self.alpha) * train_var
            
            # applying forward pass
            self.mean = train_mean
            self.variance = train_var
            new_input = (input_tensor - self.mean) / np.sqrt(self.variance + epsilon)
        
        # forward pass for test and validation
        else:
            new_input = (input_tensor - self.test_mean) / np.sqrt(self.test_variance + epsilon)
        
        output = self.weights * new_input + self.bias
        
        # reconverting back to image in the convolutional case
        if self.channels > 0:
            output = self.reformat(output)
        
        # for BP
        self.input_tensor = input_tensor
        self.normalized_input = new_input
        return output
    
    def initialize_wt_bias(self, input_tensor):
        if len(self.weights) == 0 or len(self.bias) == 0:
            if self.weights_initializer is not None:
                self.weights = self.weights_initializer.initialize([input_tensor.shape[1]], input_tensor.shape[1],
                                                                   input_tensor.shape[1])
                self.bias = self.bias_initializer.initialize([input_tensor.shape[1]], 1, input_tensor.shape[1])
            else:
                self.weights = np.ones(input_tensor.shape[1])
                self.bias = np.zeros(input_tensor.shape[1])
            
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
    
    # *********************************************************
    def backward(self, error_tensor):
        
        # reshaping for the convolutional case
        if self.channels > 0:
            error_tensor = self.reformat(error_tensor)
        # elif len(error_tensor.shape) == 4:
        #    error_tensor = error_tensor #[:, :, 0, 0]
        
        self.grad_wt = np.sum(self.normalized_input * error_tensor, axis=0)
        self.grad_bias = np.sum(error_tensor, axis=0)
        
        # update weights and bias
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_wt)
            self.bias = self.optimizer.calculate_update(self.bias, self.grad_bias)
        
        BP_output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.test_mean,
                                         self.test_variance)
        
        # converting back to image in the convolutional case
        if self.channels > 0:
            BP_output = self.reformat(BP_output)
        
        return BP_output