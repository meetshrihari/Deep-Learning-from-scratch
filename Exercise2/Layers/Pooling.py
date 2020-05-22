"""     self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_tensor = np.abs(np.random.random((self.batch_size, *self.input_shape)))

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

layer = Pooling.Pooling((2, 2), (2, 2))
"""

import numpy as np
#   https://medium.com/dataseries/basic-overview-of-convolutional-neural-network-cnn-4fcc7dbb4f17

# https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
# Figure 13


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape        #(2,2) or (3,2)
        self.pooling_shape = pooling_shape      # (2,2)

    def forward(self, input_tensor):   # (2,2,4,7)
        # get input_tensor parameters
        batch_size, channel, img_height, img_width = input_tensor.shape
        # Sliding window height and width
        ker_height = int(np.floor((img_height - self.pooling_shape[0]) / self.stride_shape[0]) + 1)
        ker_width = int(np.floor((img_width - self.pooling_shape[1]) / self.stride_shape[1]) + 1)

        self.pool_output = np.zeros((batch_size, channel, ker_height, ker_width))
        # we need to store the fucking locations of maxima
        self.pool_index = np.zeros((batch_size, channel, ker_height, ker_width))


        for batch in np.arange(batch_size):
            for ch in np.arange(channel):
                # ############  Pooling   #######################################
                img = input_tensor[batch, ch, :]
                im_row, im_col = img.shape
                #self.stride_shape[0] & [1]
                max_pool_row = im_row - self.pooling_shape[0] + 1
                max_pool_col = im_col - self.pooling_shape[0] + 1
                max_pool = np.zeros((max_pool_row, max_pool_col))
                max_pool_idx = np.zeros((max_pool_row, max_pool_col))
                ############
                for vert_start in np.arange(max_pool_row):
                    for horiz_start in np.arange(max_pool_col):
                        pool_block = img[vert_start:(vert_start + self.pooling_shape[0]),
                                    horiz_start:(horiz_start + self.pooling_shape[0])]
                        max_pool[vert_start, horiz_start] = np.max(pool_block)
                        # fucking indices for maxima
                        x, y = (np.argmax(pool_block) // self.pooling_shape[0]),\
                                (np.argmax(pool_block) % self.pooling_shape[0])  # to find the position  first / & %
                        img_row_index = vert_start + x  # locate idx in img
                        img_col_index = horiz_start + y
                        max_pool_idx[vert_start, horiz_start] = img_row_index * im_col + img_col_index

                # Now Striding in pooling WTF
                pool_row_stride = int(np.ceil(max_pool_row / self.stride_shape[0]))
                pool_col_stride = int(np.ceil(max_pool_col / self.stride_shape[1]))
                max_pool_stride = np.zeros((pool_row_stride, pool_col_stride))
                max_index_stride = np.zeros((pool_row_stride, pool_col_stride))

                for r in range(pool_row_stride):
                    for c in range(pool_col_stride):
                        max_pool_stride[r, c] = max_pool[r * self.stride_shape[0], c * self.stride_shape[1]]
                        max_index_stride[r, c] = max_pool_idx[r * self.stride_shape[0], c * self.stride_shape[1]]

                self.pool_output[batch, ch, :] = max_pool_stride
                self.pool_index[batch, ch, :] = max_index_stride
                ######################

        # for backward
        self.input_tensor = input_tensor

        return self.pool_output     # (2,2,2,3) or overlap (2,2,2,6)

    def backward(self, error_tensor):

        error_next_layer = np.zeros(np.shape(self.input_tensor))
        batch_size, channel, img_height, img_width = self.pool_output.shape

        self.error_tensor = error_tensor.reshape(self.pool_output.shape)

        for batch in range(batch_size):
            for ch in range(channel):  # num of slides
                for ht in range(img_height):
                    for wdt in range(img_width):
                        #TODO: 1) by trial and error -> input_tensor.shape[3] ?  2)
                        idx_x = int(np.floor(self.pool_index[batch, ch, ht, wdt] / self.input_tensor.shape[3]))
                        idx_y = int(np.mod(self.pool_index[batch, ch, ht, wdt], self.input_tensor.shape[3]))
                        error_next_layer[batch, ch, idx_x, idx_y] += self.error_tensor[batch, ch, ht, wdt]
        return error_next_layer

