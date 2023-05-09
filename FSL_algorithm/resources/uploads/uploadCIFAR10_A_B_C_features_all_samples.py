from torchvision import datasets, transforms
import random
import torch
from random import shuffle

class YourSampler(torch.utils.data.sampler.Sampler):
        def __init__(self, dataset, mask):
            self.mask = mask
            self.dataset = dataset

        def __iter__(self):
            #return iter([i.item() for i in torch.nonzero(self.mask)])
            selected_idx_l = [i.item() for i in torch.nonzero(self.mask)]
            shuffle(selected_idx_l)
            return iter(selected_idx_l)

        def __len__(self):
            return len(self.dataset)

def uploadCIFAR10(seed_set, device, batch_size, flag):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()                              
                                ])
    trainset = datasets.CIFAR10('cifar10', download=True, train=True, transform=transform)
    valset = datasets.CIFAR10('cifar10', download=True, train=False, transform=transform)
    
    if(flag == 'diffDistrmodel1'):
        A_list = [0,3,8,9,6]
        B_list = [1,2,4,7,5]

        # mask = [1 if trainset[i][1]%2 == 0 else 0 for i in range(len(trainset))]
        mask = [1 if trainset[i][1] in A_list else 0 for i in range(len(trainset))]
        for i in range(len(mask)):
            if mask[i] == 0:
                if(random.uniform(0,1)> 0.9):
                    mask[i] = 1
        mask = torch.tensor(mask)   
        sampler = YourSampler(trainset, mask)
        #trainloader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size,sampler = sampler,shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler = sampler, shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
        # mask = [1 if trainset[i][1]%2 == 1 else 0 for i in range(len(trainset))]
        mask = [1 if trainset[i][1] in B_list else 0 for i in range(len(trainset))]
        for i in range(len(mask)):
            if mask[i] == 0:
                if(random.uniform(0,1)> 0.9):
                    mask[i] = 1
        mask = torch.tensor(mask)   
        sampler = YourSampler(trainset, mask)
        trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler = sampler, shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
        #valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
        
        # mask1 = [1 if valset[i][1]%2 == 0 else 0 for i in range(len(valset))]
        mask1 = [1 if valset[i][1] in A_list else 0 for i in range(len(valset))]
        mask1 = torch.tensor(mask1)  
        sampler1 = YourSampler(valset, mask1)
        #trainloader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size,sampler = sampler,shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
        valloader0 = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler = sampler1, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
        # mask1 = [1 if valset[i][1]%2 == 1 else 0 for i in range(len(valset))]
        mask1 = [1 if valset[i][1] in B_list else 0 for i in range(len(valset))]
        mask1 = torch.tensor(mask1)   
        sampler1 = YourSampler(valset, mask1)
        valloader1 = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler = sampler1, shuffle=False, worker_init_fn=seed_set, pin_memory=False)    
        valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
        return trainloader, trainloader1, valloader, valloader0, valloader1
    else:
        if(flag == 'diffDistrmodel3'):
            mask = [1 if trainset[i][1]%2 == 0 else 0 for i in range(len(trainset))]
            for i in range(len(mask)):
                if mask[i] == 0:
                    if(random.uniform(0,1)> 0.93):
                        mask[i] = 1
            mask = torch.tensor(mask)   
            sampler = YourSampler(trainset, mask)
            #trainloader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size,sampler = sampler,shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler = sampler, shuffle=False)#, num_workers=workers)
            mask = [1 if trainset[i][1]%2 == 1 else 0 for i in range(len(trainset))]
            for i in range(len(mask)):
                if mask[i] == 0:
                    if(random.uniform(0,1)> 0.93):
                        mask[i] = 1
            mask = torch.tensor(mask)   
            sampler = YourSampler(trainset, mask)
            trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler = sampler, shuffle=False, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
            valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
            return trainloader, trainloader1, valloader
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, drop_last=True, num_workers=0, pin_memory=False)
            valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, num_workers=0, pin_memory=False) # pin_memory=(device.type == "cuda")
            return trainloader, valloader
