import numpy as np

# One object for the bias and one for the other weights


class Constant:
    def __init__(self, const=0.1):
        self.const = const
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.zeros(weights_shape) + self.const
        return self.weights
        #print(self.weights)


class UniformRandom:
    def __init__(self):
        self.weights = None
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.uniform(0, 1, weights_shape)    #np.random.rand(weights_shape[0], weights_shape[1])
        return self.weights


class Xavier:
    def __init__(self):
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        # Zero-mean Gaussian
        self.weights = np.random.normal(0, np.sqrt(2/(fan_in+fan_out)), weights_shape)
        return self.weights


class He:
    def __init__(self):
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.normal(0, np.sqrt(2/fan_in), weights_shape)
        return self.weights

