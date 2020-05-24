import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        update_weights = weight_tensor - (self.learning_rate * gradient_tensor)
        return update_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = None
        self.weight = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v_k is None:
            #use np.zeros.like ####################
            self.v_k = np.zeros_like(weight_tensor)  #(weight_tensor.shape[0], weight_tensor.shape[1])
        self.v_k = (self.momentum_rate * self.v_k) - (self.learning_rate * gradient_tensor)
        self.weight = weight_tensor + self.v_k
        return self.weight


class Adam:
    def __init__(self, learning_rate=0.001, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = None # weight_tensor
        self.r_k = None  # gradient_tensor
        self.k = 1   # iteration

    def calculate_update(self, weight_tensor, gradient_tensor):
        # self.g_k = gradient_tensor
        if self.v_k is None:  # First order momentum
            self.v_k = np.zeros_like(weight_tensor)
        self.v_k = (self.mu * self.v_k) + ((1-self.mu)*gradient_tensor)
        if self.r_k is None:    # Second order momentum
            self.r_k = np.zeros_like(gradient_tensor)
        self.r_k = (self.rho * self.r_k) + ((1-self.rho) * gradient_tensor) * gradient_tensor
        # Bias Correction   # refer slide 10
        self.v_hat = self.v_k / (1 - np.power(self.mu, self.k))
        self.r_hat = self.r_k / (1 - np.power(self.rho, self.k))
        # We normalize the gradient with the std deviation
        temp = (self.v_hat + np.finfo(float).eps)/(np.sqrt(self.r_hat) + np.finfo(float).eps)  #np.finfo('float').eps)
        self.weights = weight_tensor - self.learning_rate * temp
        self.k += 1
        return self.weights

