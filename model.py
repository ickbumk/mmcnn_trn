import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((1,32,32))
        label = self.labels[idx]
        return image, label

class MultiInput(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.dropout = 0.5  #randomly setting a fraction of input units to zero during training. Prevent overfitting.
        
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4), 
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(32*8*8+64*2*2, 512),  
            nn.ReLU(inplace=True),
            nn.Linear(512, self.out_dim),
            nn.Softmax(1)
        )

    def forward(self, depth, img):
        x1 = self.img_conv(img)
        x2 = self.depth_conv(depth)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1,x2), dim = 1)
        x = self.classifier(x)
        return x
        
class CNN_one_channel_depth(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.dropout = 0.5  #randomly setting a fraction of input units to zero during training. Prevent overfitting.
        
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(32 * 8*8 , 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN_one_channel_int(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.dropout = 0.5  #randomly setting a fraction of input units to zero during training. Prevent overfitting.
        
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4), 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(64 * 2*2 , 128),  
            nn.ReLU(inplace=True),
            nn.Linear(128, self.out_dim),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
