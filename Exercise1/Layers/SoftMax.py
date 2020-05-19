import numpy as np


class SoftMax:
    def __init__(self):
        pass

    def forward(self, input_tensor):

        maximum = np.expand_dims(np.max(input_tensor, 1), 1)  # similar to x[:,np.newaxis]

        temp = input_tensor - maximum   # to increase stability
        # denominator = np.expand_dims(np.sum(np.exp(temp), 1), 1)
        self.yk_hat = np.exp(temp) / np.expand_dims(np.sum(np.exp(temp), axis=1), 1)
        return self.yk_hat

    def backward(self, label_tensor):
        scalar_rows = np.sum(np.multiply(label_tensor,self.yk_hat), axis=1)
        error = np.multiply(self.yk_hat, np.subtract(label_tensor[0:label_tensor.shape[0], :].T, scalar_rows).T)
        return error

