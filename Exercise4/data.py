from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    #TODO implement the Dataset class according to the description
    def __init__(self, mode, path, split_paramtet, compose = tv.transforms.Compose([tv.transforms.ToTensor()])):
        self.mode = mode
        self.path = path
        self.split_parameter = split_paramtet
        self.compose = compose
        self.df = pd.read_csv(self.path, sep=';')
        self.image = self.df.iloc[0:, 0]
        self.labels = self.df.iloc[0:, 2:]

        self.im_train, self.im_val, self.label_train, self.label_val = train_test_split(self.image, self.labels, train_size=self.split_parameter, random_state = 3)

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = gray2rgb(imread(str(self.im_train.iloc[index])))
            t_image = self.compose(sample)
            label = np.asarray(self.label_train.iloc[index]).reshape(1, 2)
            label_compose = tv.transforms.Compose([tv.transforms.ToTensor()])
            t_label = label_compose(label)
            return (t_image,t_label)
        if self.mode == 'val':
            sample = gray2rgb(imread(str(self.im_val.iloc[index])))
            t_image = self.compose(sample)
            label = np.asarray(self.label_val.iloc[index]).reshape(1, 2)
            label_compose = tv.transforms.Compose([tv.transforms.ToTensor()])
            t_label = label_compose(label)
            return (t_image,t_label)

    def __len__(self):
        if self.mode == 'train':
            return len(self.im_train.index)
        if self.mode == 'val':
            return len(self.im_val.index)


    def pos_weight(self):
        if self.mode == 'train':
            weight_crack = sum(1-self.label_train.crack)/sum(self.label_train.crack)
            weight_inactive = sum(1-self.label_train.inactive)/sum(self.label_train.inactive)
            weight_tensor = np.asarray([weight_crack, weight_inactive]).reshape(1,2)
            compose = tv.transforms.Compose([tv.transforms.ToTensor()])
            return compose(weight_tensor)
        else:
            return None

'''
df = pd.read_csv('./train.csv', sep=';')
image = df.iloc[0:, 0]
label = df.iloc[0:,2:]

t,v, lt, lv = train_test_split(image,label)
weight_crack = sum(1 - lt.crack) / sum(lt.crack)
weight_inactive = sum(1 - lt.inactive) / sum(lt.inactive)
weight_tensor = np.asarray([weight_crack, weight_inactive]).reshape(1, 2)

one_image = df.iloc[1,0]
img = imread(str(one_image))
compose = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize()])



df = pd.read_csv('./train.csv', sep=';')
image = df.iloc[0:, 0]
label = df.iloc[0:,2:]

t,v, lt, lv = train_test_split(image,label, train_size= 0.8, random_state=1)
print(len(v.index))
sample = gray2rgb(imread(str(t.iloc[1])))
compose = tv.transforms.Compose([tv.transforms.ToTensor()])

t_image = compose(sample)
label = np.asarray(lt.iloc[1]).reshape(1,2)
t_label = compose(label)

print(t_image, t_label)
'''

def get_train_dataset():
    #TODO
    #no augmentation was done
    compose_train = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
    trainob = ChallengeDataset('train', './train.csv', 0.8, compose_train)
    return trainob

# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    #TODO
    compose_val = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
    valob = ChallengeDataset('val', './train.csv', 0.8, compose_val)
    return valob

