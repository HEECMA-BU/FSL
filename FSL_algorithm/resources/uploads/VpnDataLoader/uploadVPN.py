from .mnist import MNIST
from torchvision import datasets, transforms
import random
import torch
from torch.utils.data import DataLoader

def uploadVPN(seed_set, device, batch_size):    
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
                                    transforms.ToTensor()                              
                                ])
    trainset = MNIST(root='/home/cc/DeepTraffic/2.encrypted_traffic_classification/3.PerprocessResults/6class/NovpnSessionAllLayers', train=True, download=True, transform=transform )
    valset = MNIST(root='/home/cc/DeepTraffic/2.encrypted_traffic_classification/3.PerprocessResults/6class/NovpnSessionAllLayers', train=False, download=True, transform=transform )
    
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True)


    #trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
    #valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
    return train_dataloader, test_dataloader