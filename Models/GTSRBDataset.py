import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class GTSRBDataset(Dataset):

    def __init__(self, indices=None, train=True, transform=None):
        dataType = 'Train.csv' if train else 'Test.csv'
        path = os.getcwd() + '\\data\\' + dataType

        self.csv_data = pd.read_csv(path)
        if indices:
            self.size = len(indices)
        else:
            self.size = len(self.csv_data)
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.indices:
            idx = self.indices[idx]
        label = self.csv_data['ClassId'][idx]
        relativePath = 'data\\' + self.csv_data['Path'][idx]
        imgPath = os.path.join(os.getcwd(), relativePath)

        img = Image.open(imgPath)
        if self.transform:
            img = self.transform(img)

        return img, label
