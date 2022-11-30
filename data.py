import torchvision.transforms.functional as F
from os import listdir
from os.path import join
from torch.utils.data import Dataset

from PIL import Image, ImageFilter
from torchvision.transforms import Compose, ToTensor
import torch
import numpy as np
import nni

from config import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.class_b

def get_image(data_path:    str = DATASET_ROOT_DIR,
              class_symbol: str = 'B', # B - I - L - N - O
              phase:        str = 'train'):
    data_list = []
    class_dir = join(data_path, phase, class_symbol.upper())
    for filename in listdir(class_dir):
        if is_image_file(filename):
            data_list.append(join(class_dir, filename))

    return data_list
    
def to_onehot_tensor(class_num, index):
    label = [0 for _ in range(class_num)]
    label[index] = 1
    return torch.Tensor(label)


class LoadDataset(Dataset):
    def __init__(self, data_path:    str = DATASET_ROOT_DIR, 
                       phase:        str = 'train'
                       ):  # lr_size must be valid
        super(LoadDataset, self).__init__()
        self.num_class = 5
        self.images = []
        self.phase = phase
        
        for class_symbol in CLASSES.keys():
            self.images.extend(
                get_image(
                    data_path    = data_path,
                    class_symbol = class_symbol,
                    phase        = self.phase
                )
            )

        self.transform = Compose([
            ToTensor()
        ])


        
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        np_temp = np.load(self.images[index][:-3] + 'npy')

        temp = Image.fromarray(np_temp)

        class_symbol = self.images[index].split('/')[-2]
        label = to_onehot_tensor(self.num_class, CLASSES[class_symbol][0])

        image = self.transform(image)
        temp = self.transform(temp).squeeze(2)


        return self.__len__(), image, temp, label


    def __len__(self):
        return len(self.images)