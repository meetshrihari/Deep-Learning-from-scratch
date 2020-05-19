import numpy as np
import Layers.FullyConnected as FC

batch_size = 9
input_size = 4
output_size = 3
input_tensor = np.random.rand(batch_size, input_size)
layer = FC.FullyConnected(input_size, output_size)
output_tensor = layer.forward(input_tensor)
error_tensor = layer.backward(output_tensor)