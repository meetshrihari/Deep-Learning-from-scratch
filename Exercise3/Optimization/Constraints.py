import numpy as np
from Optimization.Optimizers import Base_optimizer


class L2_Regularizer(Base_optimizer):
    def __init__(self, alpha=0.1):  # alpha = lambda
        self.alpha = alpha  # representing the regularization weight
        # TODO: Refactor the optimizers to apply the new regularizer
        Base_optimizer.__init__(self)  # = alpha
    
    def calculate_gradient(self, weights):
        self.sub_gradient = self.alpha * weights
        return self.sub_gradient
    
    def norm(self, weights):  # Norm enhanced Loss
        self.L2 = self.alpha * np.sqrt(np.sum(np.square(np.abs(weights))))
        # self.L2 = np.linalg.norm(weights.flatten(), ord=2, keepdims=True) * self.alpha
        return self.L2


class L1_Regularizer(Base_optimizer):
    def __init__(self, alpha):
        Base_optimizer.__init__(self)
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        self.sub_gradient = self.alpha * np.sign(weights)
        return self.sub_gradient
    
    def norm(self, weights):
        # self.L1 = np.sum(np.abs(weights)) * self.alpha
        self.L1 = np.linalg.norm(weights.flatten(), ord=1, keepdims=True) * self.alpha
        return self.L1
