from torchvision import datasets, transforms
import random
import torch

def uploadCIFAR10(seed_set, device, batch_size):    
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()                              
                                ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"), drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"))
    return trainloader, valloader