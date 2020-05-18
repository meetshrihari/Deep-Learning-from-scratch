import math
import json
import random
from scipy import ndimage
from skimage.transform import rotate, resize
import numpy as np
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, *image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size[0]
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.iteration = 0
        self.batch_matrix = []
        with open(self.label_path) as json_file:
            self.json_data = json.load(json_file)

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
                           6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

        seq_json = np.arange(len(self.json_data))  # 0 to 99
        self.rows = int(len(self.json_data) / self.batch_size)  # 8+1
        #print(self.rows)
        self.batch_matrix = np.zeros((0, self.batch_size))
        for i in np.arange(0, (self.rows * self.batch_size), self.batch_size):
            self.batch_matrix = np.vstack([self.batch_matrix, seq_json[i:i + self.batch_size]])

        # Generate a matrix which includes all the json images number
        if self.batch_matrix[self.rows - 1, self.batch_size - 1] != seq_json[len(self.json_data) - 1]:
            rem = self.batch_size - (int(len(seq_json) - self.batch_matrix[self.rows - 1, self.batch_size - 1]))
            reuse = np.concatenate([seq_json[int(self.batch_matrix[self.rows - 1, self.batch_size - 1] + 1):len(seq_json)],
                                    self.batch_matrix[0, 0:rem + 1]])
            #print(seq_json[int(self.batch_matrix[self.rows - 1, self.batch_size - 1] + 1):len(seq_json)])
            self.batch_matrix = np.vstack([self.batch_matrix, reuse])
        #print(self.batch_matrix)
        self.batch_matrix = self.batch_matrix.astype(int)

    def next(self):
        if self.shuffle:
            flat = np.ndarray.flatten(self.batch_matrix)
            #print(self.batch_matrix)
            random.shuffle(flat)
            row = math.ceil(len(self.json_data) / self.batch_size)

            shuffle_matrix = np.reshape(flat, (row, self.batch_size))
            self.batch_matrix = shuffle_matrix

        batch = []
        labels = '0'
        temp = []
        if self.iteration > self.rows:
            self.iteration = 0
        else:
            # all for images in batch
            batch = self.batch_matrix[self.iteration, :]
            for no in batch:
                im_shape = np.load(self.file_path + '{}'.format(no) + '.npy')
                if im_shape.shape != (self.image_size[0], self.image_size[1], 3):  # check for size
                    im_shape = resize(im_shape, self.image_size, anti_aliasing=True)
                    temp.append(im_shape)
                    img = np.array(temp)
                elif (np.random.randint(0,len(batch))%2 ==0) and (self.mirroring or self.rotation):
                    #count = np.random.random_integers(len(batch))
                    temp.append(self.augment(im_shape))
                    img = np.array(temp)

                else:
                    temp.append(im_shape)
                    img = np.array(temp)
            batch = img
            print(batch.shape)
            labels = self.batch_matrix[self.iteration, :]
            self.iteration += 1
        return np.copy(batch), np.copy(labels)

    def augment(self, img):
        if self.mirroring:
            img = np.flipud(img)
            #plt.imshow(img, interpolation='hamming')
            #plt.show()

        elif self.rotation:
            #plt.imshow(img, interpolation='hamming')
            img = np.rot90(img)
        return np.copy(img)

    def class_name(self, x):
        return self.class_dict[self.json_data['{}'.format(x)]]

    def show(self):
        batch, label = self.next()
        #self.iteration = 0
        plt.figure(figsize=(8,8))
        for no in np.arange(len(batch)):
#TODO: how to calculate the subplot in the image
            plt.subplot(int(self.rows/2), 3, no + 1)
            plt.imshow(self.augment(batch[no]),aspect='equal', interpolation='bicubic' )
            plt.title(label[no], fontsize = 12)
            plt.axis('off')
        plt.show()



