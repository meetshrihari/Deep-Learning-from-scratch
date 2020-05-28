import numpy as np
from Layers.FullyConnected import FullyConnected
from Layers.Base import Base_Layer
import copy


class LSTM(Base_Layer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size      # 13
        self.hidden_size = hidden_size    # 7
        self.output_size = output_size    # 5
        
        self.hidden_state = None        # h_t
        self.cell_state = None          # c_t
        self.memory = False
        self.optimizer = None
        
        self.fc_f = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_i = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_c_hat = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_o = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_y = FullyConnected(hidden_size, output_size)
        
    def initialize(self, weights_initializer, bias_initializer):
        self.fc_y.initialize(weights_initializer, bias_initializer)
        self.fc_f.initialize(weights_initializer, bias_initializer)
        self.fc_i.initialize(weights_initializer, bias_initializer)
        self.fc_c_hat.initialize(weights_initializer, bias_initializer)
        self.fc_o.initialize(weights_initializer, bias_initializer)
        
    # *********************** ALL PROPERTIES
    @property
    def memorize(self):
        return self.memory

    @memorize.setter
    def memorize(self, value):
        self.memory = value

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @property
    def weights(self):
        wf = self.fc_f.weights  # getter()     # get_weights()
        wi = self.fc_i.weights  # get_weights()
        wc_hat = self.fc_c_hat.weights  # .getter()  # get_weights()
        wo = self.fc_o.weights  # .getter()     # get_weights()
    
        weights = np.hstack((wf, wi, wc_hat, wo))
        return weights

    @weights.setter
    def weights(self, weights):
        set_weights = weights
        # self.fc_f.(set_weights[:,0:self.hidden_size])
        self.fc_f.weights = set_weights[:, 0:self.hidden_size]
        self.fc_i.weights = set_weights[:, self.hidden_size:self.hidden_size * 2]
        self.fc_c_hat.weights = set_weights[:, self.hidden_size * 2:self.hidden_size * 3]
        self.fc_o.weights = set_weights[:, self.hidden_size * 3:self.hidden_size * 4]

    @property
    def gradient_weights(self):
        # print('grad_wt_hx', self.grad_wt_hx.shape)
        return self.grad_wt_hx

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def setter(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    def getter(self):
        return self._optimizer

    optimizer = property(getter, setter)
    
    # *******************************************************************************
    # refer lecture slide 29
    def forward(self, input_tensor):
        self.loop = input_tensor.shape[0]
        self.input_tensor = input_tensor  # 9*13
        
        self.i_t = np.zeros((self.loop, self.hidden_size))
        self.o_t = np.zeros((self.loop, self.hidden_size))
        self.f_t = np.zeros((self.loop, self.hidden_size))
        self.c_hat_t = np.zeros((self.loop, self.hidden_size))
        self.cell_state = np.zeros((self.loop+1, self.hidden_size))     # +1 to store next state

        if self.memory:
            if self.hidden_state is None:
                self.hidden_state = np.zeros((self.loop + 1, self.hidden_size))
            else:
                # update with the previous value
                self.hidden_state[0] = self.hidden_state[-1]
        else:
            self.hidden_state = np.zeros((self.loop + 1, self.hidden_size))

        y_out = np.zeros((self.loop, self.output_size))

        for time in range(self.loop):   # current state = time -> t-1; time+1 -> t
            new_hidden_state = self.hidden_state[time][np.newaxis, :]
            new_input_tensor = self.input_tensor[time][np.newaxis, :]
            new_input = np.concatenate((new_hidden_state, new_input_tensor), axis=1)
            
            self.f_t[time] = self.sigmoid(self.fc_f.forward(new_input))
            self.i_t[time] = self.sigmoid(self.fc_i.forward(new_input))
            self.c_hat_t[time] = np.tanh(self.fc_c_hat.forward(new_input))
            # output cell state
            self.cell_state[time+1] = (self.f_t[time] * self.cell_state[time]) + (self.i_t[time] * self.c_hat_t[time])
            self.o_t[time] = self.sigmoid(self.fc_o.forward(new_input))
            self.hidden_state[time+1] = self.o_t[time] * np.tanh(self.cell_state[time+1])
            # output sigmoid not needed
            y_out[time] = self.fc_y.forward(self.hidden_state[time+1][np.newaxis, :])
        #print(new_input.shape)
        return y_out

    # *********************************************************************
    # https://towardsdatascience.com/back-to-basics-deriving-back-propagation-on-simple-rnn-lstm-feat-aidan-gomez-c7f286ba973d
    # h
    
    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        gra_cell = np.zeros((1, self.hidden_size))
        
        self.grad_wt_i = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))   # (7+13+1, 7)
        self.grad_wt_f = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.grad_wt_c_hat = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.grad_wt_y = np.zeros((self.hidden_size+1, self.output_size))
        self.grad_wt_o = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))
        self.hx = np.zeros((self.loop, self.hidden_size + self.input_size))

        hidden = np.zeros((1, self.hidden_size))

        for time in reversed(range(self.loop)):       # 9 -> 0
            self.fc_y.input_tensor = np.hstack((self.hidden_state[time + 1], 1))[np.newaxis, :]
            print((np.hstack((self.hidden_state[time + 1], 1))[np.newaxis, :]).shape)
            self.fc_i.input_tensor = np.hstack((self.hidden_state[time], self.input_tensor[time], 1))[np.newaxis, :]
            self.fc_o.input_tensor = np.hstack((self.hidden_state[time], self.input_tensor[time], 1))[np.newaxis, :]
            self.fc_f.input_tensor = np.hstack((self.hidden_state[time], self.input_tensor[time], 1))[np.newaxis, :]
            self.fc_c_hat.input_tensor = np.hstack((self.hidden_state[time], self.input_tensor[time], 1))[np.newaxis,:]
            
            delta_y_h = self.fc_y.backward(self.error_tensor[time][np.newaxis, :]) + hidden
            o_error = delta_y_h * (np.tanh(self.cell_state[time+1])) * (self.o_t[time]-self.o_t[time]**2)

            delta_h_ct = (self.o_t[time] * (1 - np.tanh(self.cell_state[time + 1]) ** 2)) * delta_y_h + gra_cell
            gra_cell = self.f_t[time] * delta_h_ct
            i_error = delta_h_ct * (self.c_hat_t[time]) * (self.i_t[time]-self.i_t[time]**2)
            f_error = delta_h_ct * (self.cell_state[time]) * (self.f_t[time] - self.f_t[time] ** 2)
            c_hat_error = delta_h_ct * self.i_t[time] * (1 - self.c_hat_t[time] ** 2)
            
            # see the diagram
            eo_hx = self.fc_o.backward(o_error)
            ec_hat_hx = self.fc_c_hat.backward(c_hat_error)
            ei_hx = self.fc_i.backward(i_error)
            eo_fx = self.fc_f.backward(f_error)
            self.hx[time] = eo_hx + ec_hat_hx + ei_hx + eo_fx
            
            hidden = self.hx[time, 0:self.hidden_size]

            self.grad_wt_y += self.fc_y.gradient_weights
            self.grad_wt_f += self.fc_f.gradient_weights
            self.grad_wt_i += self.fc_i.gradient_weights
            self.grad_wt_c_hat += self.fc_c_hat.gradient_weights
            self.grad_wt_o += self.fc_o.gradient_weights

            self.grad_wt_hx = np.hstack((self.grad_wt_f, self.grad_wt_i, self.grad_wt_c_hat, self.grad_wt_o))
            
        if self.optimizer is not None:
            # call the hstacked weights from getter
            # update the hstacked weights using gradweights
            # use setter to set the weights
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_wt_hx)
            self.fc_y.weights = self.optimizer.calculate_update(self.fc_y.weights, self.grad_wt_y)
            
        output = self.hx[:, self.hidden_size:self.hidden_size+self.input_size+1]

        return output




"""

    def calculate_regularization_loss(self):
        
        pass
"""