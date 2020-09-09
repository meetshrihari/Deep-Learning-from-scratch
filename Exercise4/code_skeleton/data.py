from torch.utils.data import Dataset
import os
import torch as t
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.DF = data
        self.mode = mode    # val or train
        # Compose is the callable class which does chain of transformations on the data
        self.to_tensor = tv.transforms.Compose([tv.transforms.ToTensor()])

        # Consider creating two different transforms based on whether you are in the training or validation dataset.
        self.val_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self.train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(),tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std), ])

        self.images= self.DF.iloc[:, 0]
        self.labels = self.DF.iloc[:, 1:]
        self.count = 0


    def __len__(self):
        #if self.mode == 'train':
        return len(self.DF)
        #if self.mode == 'val':
        #    return len(self.images.index)

    """The file names and the corresponding labels are listed in the csv-file data.csv. 
    Each row in the csv-file contains the path to an image in our dataset and two numbers indicating 
    if the solar cell shows a 'crack' and if the solar cell can be considered 'inactive'"""

    def __getitem__(self, index): # used for accessing list items

        if self.mode == 'val':
            #self.count += 1
            #print(self.count)
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)
            img = self.val_transform(img)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1,2))
            #img = self.to_tensor(img)
            return (img, labels)
        if self.mode == 'train':
            #self.count += 1
            #print(self.count)
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)

            img = self.train_transform(img)
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1, 2))
            #img = self.to_tensor(img)
            return (img, labels)



'''
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
tab = pd.read_csv(csv_path, sep=';')
temp = ChallengeDataset(tab, 'val')

val_dl = t.utils.data.DataLoader(temp, batch_size=1)
print(val_dl)
for x,y in val_dl:
    print(y)
    break'''

