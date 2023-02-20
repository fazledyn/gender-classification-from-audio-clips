import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


import os
import cv2
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

alex = models.alexnet()

"""
Write AlexNet model that reads images from CSv and trains
"""

class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
class ImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.image_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.image_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):


