import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution    #int(input('figure resolution = '))
        self.tile_size = tile_size  #int(input('Tile size in pixels = '))
        self.output = []
        self.black_white = []

    def draw(self):
        self.black_white = np.zeros((2 * self.tile_size, 2 * self.tile_size), dtype='uint8')
        self.black_white[:, self.tile_size : 2 * self.tile_size] = 1  # make white    [ 0 1 ]
        self.black_white[self.tile_size : 2 * self.tile_size, :] = 1  # make white    [ 1 0 ]
        self.black_white[self.tile_size : 2 * self.tile_size, self.tile_size : 2 * self.tile_size] = 0
        tot_Blocks = int(self.resolution / (2 * self.tile_size))
        self.output = np.tile(self.black_white, [tot_Blocks, tot_Blocks])  # replicate the blocks for checkboard
        return np.copy(self.output)

    def show(self):
        #f = plt.figure(1)
        plt.imshow(self.img, cmap='gray', interpolation = 'hamming')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution    #int(input('figure resolution = '))
        self.output = []

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3))
        #if y axis is from 0-> resolution
        #self.img[..., 0] = np.linspace(0, 1, 500)   # 0 -> 1 same matrix
        #self.img[..., 1] = np.linspace(1, 0, 500)[:, np.newaxis]    # 0-> 1 next matrix
        #self.img[..., 2] = np.linspace(1, 0, 500)   # 0->1 same matrix

        # keeping axis default
        self.output[..., 0] = np.linspace(0, 1, self.resolution)[np.newaxis, :]
        self.output[..., 1] = np.linspace(0, 1, self.resolution)[:, np.newaxis]
        self.output[..., 2] = np.linspace(1, 0, self.resolution)  # [:, np.newaxis]

        return np.copy(self.output)

    def show(self):
        #g = plt.figure(2)
        plt.imshow(self.img, cmap='gray', interpolation = 'hamming')
        #plt.ylim(0, self.resolution)
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, *args):
        self.resolution = resolution    #int(input('figure resolution = '))
        self.radius = radius    #int(input('Circles radius = '))
        self.center = args[0]  #tuple([circle_x, circle_y])
        # pos = tuple(map(int,input('Circles x & y position = ').split(',')))
        self.output = []

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution), dtype = 'uint8')
        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        circle = np.sqrt((x-self.center[0])**2 + (y-self.center[1])**2)  #np.hypot
        self.output[np.where(circle < self.radius)] = 1  #kind of slicing
        return np.copy(self.output)

    def show(self):
        #h = plt.figure(3)
        plt.imshow(self.output, cmap='gray', interpolation = 'hamming')
        plt.show()
