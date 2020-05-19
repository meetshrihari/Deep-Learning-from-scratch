
class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weigth_tensor, gradient_tensor):
        update_weights = weigth_tensor - (self.learning_rate * gradient_tensor)
        return update_weights