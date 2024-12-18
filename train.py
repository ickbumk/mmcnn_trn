# -*- coding: utf-8 -*-
"""
Sample code for training level 1
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from model import MultiInput
from mmtraining import train, test
import numpy as np
import time



class CustomDataset(Dataset):
    def __init__(self, depth, images, labels):
        self.images = images
        self.depth = depth
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((1,32,32))
        depth = self.depth[idx].reshape((1,32,32))
        label = self.labels[idx]
        return depth, image, label
    
device = torch.device('cuda')

raw_imgs = np.load('dataset/features_with_intensity_128_lvl1.npy')
labels = np.load('dataset/labels__with_intensity_128_lvl1.npy')

normalized_imgs = raw_imgs.copy().astype(np.float32)

#add noise to depth
noise = np.random.normal(loc = 0, scale = 1, size = normalized_imgs[:,:,0].shape)
normalized_imgs[:,:,0] += noise
normalized_imgs[:,:,0] -= -200
normalized_imgs[:,:,0] /= 800


normalized_imgs = torch.from_numpy(normalized_imgs).float()

correct_labels = labels[:,0] #lvl1
correct_labels = torch.from_numpy(correct_labels).int()

train_imgs, train_labels = normalized_imgs, correct_labels

train_dataset = CustomDataset(train_imgs[:,:,0],train_imgs[:,:,1], train_labels)

# validation set

realistic_imgs = np.load('dataset/random_features_with_intensity_128_samples_20.npy')
realistic_labels = np.load('dataset/random_labels__with_intensity_128_samples_20.npy')

normalized_realistic_imgs = realistic_imgs.copy().astype(np.float32)

# add noise to the validation data

noise_test = np.random.normal(loc = 0, scale = 3, size = normalized_realistic_imgs[:,:,0].shape)
normalized_realistic_imgs[:,:,0] += noise_test
normalized_realistic_imgs[:,:,0] -= -200
normalized_realistic_imgs[:,:,0] /= 800

normalized_realistic_imgs = torch.from_numpy(normalized_realistic_imgs).float()

correct_realistic_labels = realistic_labels[:,0] # lvl1 validation
correct_realistic_labels = torch.from_numpy(correct_realistic_labels).int()

test_imgs, test_labels = normalized_realistic_imgs, correct_realistic_labels

test_dataset = CustomDataset(test_imgs[:,:,0],test_imgs[:,:,1], test_labels) # change when needed for not testing



train_batch_size = 128
test_batch_size = 128
n_epochs = 100
learning_rate = 1e-2
seed = 100
input_dim = 1
out_dim = 4
momentum = 0.7
img_size = int(np.sqrt(normalized_imgs.shape[1]))
pool_size = 4


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)# , pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)#, pin_memory=True)

network = MultiInput(in_dim=input_dim, out_dim=out_dim)
network = network.to(device)

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

start_time = time.time()
best_model = 40
train_accuracies = []
test_accuracies = []

for epoch in range(1, n_epochs + 1):
    acc_train = train(network, train_loader, optimizer, epoch, device)
    acc_test = test(network, test_loader, device)
    
    train_accuracies.append(acc_train)
    test_accuracies.append(acc_test)
    
    if acc_test > best_model:
        best_model = acc_test
        model_path = 'models/mmcnn.pth'
        torch.save(network.state_dict(), model_path)    
        
  

end_time = time.time()
print('time taken = ', end_time-start_time)
print('maximum val acc:', np.array(test_accuracies).max())