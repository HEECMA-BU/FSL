from torchvision import datasets, transforms
import random
import torch

def uploadMNIST(seed_set, device, batch_size):    
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()                              
                                ])
    trainset = datasets.MNIST('mnist', download=True, train=True, transform=transform)
    valset = datasets.MNIST('mnist', download=True, train=False, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, drop_last=True, num_workers=0, pin_memory=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, num_workers=0, pin_memory=False)
    return trainloader, valloader
