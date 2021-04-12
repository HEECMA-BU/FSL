from torchvision import datasets, transforms
import random
import torch

class YourSampler(torch.utils.data.sampler.Sampler):
        def __init__(self, dataset, mask):
            self.mask = mask
            self.dataset = dataset

        def __iter__(self):
            #return iter([i.item() for i in torch.nonzero(self.mask)])
            return iter([i.item() for i in torch.nonzero(self.mask)])

        def __len__(self):
            return len(self.dataset)

def uploadEMNIST(seed_set, device, batch_size):    
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()                              
                                ])

    trainset = datasets.EMNIST('emnist', split='letters', download=True, train=True, transform=transform)
    valset = datasets.EMNIST('emnist', split='letters', download=False, train=False, transform=transform)

    # # Run all labels with a subset of n = 500 or 5000  
    # trainset = torch.utils.data.Subset(trainset, range(0,5000))
    # valset = torch.utils.data.Subset(valset, range(0,5000))
    
    # select only images A, B, C
    mask = [1 if trainset[i][1] == 1 else 0 for i in range(len(trainset))]
    mask = torch.tensor(mask)   
    sampler = YourSampler(trainset, mask)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = sampler, shuffle=True, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"), drop_last=True)
    
    mask1 = [1 if trainset[i][1] == 1 else 0 for i in range(len(valset))]
    mask1 = torch.tensor(mask1)   
    sampler1 = YourSampler(valset, mask1)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, sampler = sampler1, shuffle=True, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"))
    
    return trainloader, valloader
