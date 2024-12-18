import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np


def split_train_val(org_train_set, valid_ratio=0.1):
    num_train = len(org_train_set)

    split = int(np.floor(valid_ratio * num_train))

    indices = list(range(num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)

    return new_train_set, val_set

def test(net, loader, device):
    net.eval()

    correct = 0

    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            img, depth, target = batch  # Assuming the first two elements are data and target
            
            img, depth, target = img.to(device), depth.to(device), target.to(device)

            output = F.log_softmax(net(img, depth), dim=1)

            target = target.to(torch.long)

            test_loss += F.nll_loss(output, target, size_average=True).item()
            
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loader)
    
    accuracy = 100.0 * correct / len(loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(loader.dataset),
                                                                          (100. * correct / len(loader.dataset))), flush=True)
    return accuracy


def train(net, loader, optimizer, epoch, device):
    
    net.train()

    correct = 0

    for batch_idx, batch in enumerate(loader):
        img, depth, target = batch  # Assuming the first two elements are data and target
        
        img, depth, target = img.to(device), depth.to(device), target.to(device)

        optimizer.zero_grad()
        output = F.log_softmax(net(img, depth), dim=1)
        target = target.to(torch.long) 

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(loader.dataset)
    print('Epoch:{} Accuracy: {:.2f}%'.format(epoch, accuracy), flush=True)
    
    return accuracy

