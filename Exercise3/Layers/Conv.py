"""
self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

            scipy.ndimage.correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0)
"""


import copy
from scipy.signal import correlate2d, convolve2d
import numpy as np


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.conv_shape = convolution_shape # 1D = [c,m], 2D = [c,m,n]
        self.num_kernels = num_kernels
        self.stride_row = self.stride_shape[0]
        self.conv_row = convolution_shape[1]
        self.bias=None
        self.weights = None

        if len(convolution_shape) == 3:  # for 3d convolution
            self.weights = np.random.rand(num_kernels, *convolution_shape)
            self.bias = np.random.rand(num_kernels)  # 1 bias per kernel
            self.stride_col = self.stride_shape[1]
            self.conv_col = convolution_shape[2]
            self.dim1 = False
        else:  # distinction for 2d convlution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], 1)
            self.bias = np.random.rand(num_kernels)
            self.stride_col = 1
            self.conv_col = 1
            self.dim1 = True  # boolean for the 2d case

        self._weightsOptimizer = None
        self._biasOptimizer = None

    def forward(self, input_tensor):  # input dimensions batch,c,y,x
        if len(self.conv_shape) == 2:
            self.input_tensor = input_tensor.reshape(*input_tensor.shape, 1)  # convert 3D to 4D (b,c,h,1)
        else:
            self.input_tensor = input_tensor.reshape(*input_tensor.shape) # already have (b,c,h,w)

        ##set the output shape
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))

        ###we stride
        strideRow = int(np.ceil(output_tensor.shape[2] / self.stride_row))
        strideCol = int(np.ceil(output_tensor.shape[3] / self.stride_col))
        strided_out = np.zeros((input_tensor.shape[0], self.num_kernels, strideRow, strideCol))

        for batch in range(input_tensor.shape[0]):  # batches
            for ker_num in range(self.num_kernels):  # kernels (chanels of the output)
                for j in range(self.input_tensor.shape[1]):  # chanels of the input
                    output_tensor[batch, ker_num, :] += correlate2d(self.input_tensor[batch, j, :], self.weights[ker_num, j, :], 'same')  # we correlate each channel
        # add the bias to each element
                for ht in range(output_tensor.shape[2]):
                    for wdt in range(output_tensor.shape[3]):
                        output_tensor[batch, ker_num, ht, wdt] += self.bias[ker_num]

            #for i in range(self.num_kernels):
                for row in range(strideRow):
                    for col in range(strideCol):
                        strided_out[batch, ker_num, row, col] = output_tensor[
                            batch, ker_num, row * self.stride_row, col * self.stride_col]

        self.output_shape = np.shape(strided_out)  # store the output shape

        # again distinction between 2d and 3d
        if len(self.conv_shape) == 2:
            strided_out = strided_out.reshape(strided_out.shape[0],strided_out.shape[1],strided_out.shape[2])

        return strided_out

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
        fan_out = self.num_kernels * self.weights.shape[2] * self.weights.shape[3]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.weights.shape[0])
        return self.weights, self.bias

    def backward(self, error_tensor):

        ######## Initilize
        self.error_T = error_tensor.reshape(self.output_shape)

        # upsampling
        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))  # num_ker=num chanels
        next_error = np.zeros(self.input_tensor.shape)  # we have the same size of the input
        # For Padded input image
        self.padding_X = np.zeros((*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.conv_row - 1,
                                   self.input_tensor.shape[3] + self.conv_col - 1))
        # Bias
        self.grad_bias = np.zeros(self.num_kernels)
        # gradient with respect to the weights
        self.grad_weights = np.zeros(self.weights.shape)

        #########################

        # Padding
        # input padding we pad with half of the kernel size
        pad_up = int(np.floor(self.conv_col / 2))  # (3, 5, 8)
        pad_left = int(np.floor(self.conv_row / 2))

        for batch in range(self.up_error_T.shape[0]):
            for ker_num in range(self.up_error_T.shape[1]):
                # gradient with respect to the bias
                self.grad_bias[ker_num] += np.sum(error_tensor[batch, ker_num, :])

                for ht in range(self.error_T.shape[2]):
                    for wdt in range(self.error_T.shape[3]):
                        self.up_error_T[batch, ker_num, ht * self.stride_row, wdt * self.stride_col] = self.error_T[
                            batch, ker_num, ht, wdt]  # we fill up with the strided error tensor

                for ch in range(self.input_tensor.shape[1]):  # channel num
                    next_error[batch, ch, :] += convolve2d(self.up_error_T[batch, ker_num, :], self.weights[ker_num, ch, :],'same')  # same

            # Referred from
            for num in range(self.input_tensor.shape[1]):
                for ht in range(self.padding_X.shape[2]):
                    for wdt in range(self.padding_X.shape[3]):
                        if (ht > pad_left - 1) and (ht < self.input_tensor.shape[2] + pad_left):
                            if (wdt > pad_up - 1) and (wdt < self.input_tensor.shape[3] + pad_up):
                                self.padding_X[batch, num, ht, wdt] = self.input_tensor[batch, num, ht - pad_left, wdt - pad_up]

            for ker_num in range(self.num_kernels):
                for ch in range(self.input_tensor.shape[1]):
                    self.grad_weights[ker_num, ch, :] += correlate2d(self.padding_X[batch, ch, :],
                                                              self.up_error_T[batch, ker_num, :],'valid')  # convolution of the error tensor with the padded input tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_weights)

        if self._biasOptimizer is not None:
            self.bias = self._biasOptimizer.calculate_update(self.bias, self.grad_bias)

        # again distinction between 2d and 3d
        if len(self.conv_shape) == 2:   #if self.dim1:
            next_error = next_error.reshape(next_error.shape[0],next_error.shape[1], next_error.shape[2])

        return next_error

    def getter(self):
        return self._weightsOptimizer

    def set_optimizer(self, optimizer):
        self._weightsOptimizer = copy.deepcopy(optimizer)
        self._biasOptimizer = copy.deepcopy(optimizer)
        return

    optimizer = property(getter, set_optimizer)

    @property
    def gradient_weights(self):
        return self.grad_weights
    @property
    def gradient_bias(self):
        return self.grad_bias
