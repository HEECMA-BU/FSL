from torchvision import datasets, transforms
import random
import torch

def uploadCIFAR10(seed_set, device, batch_size):    
    transform = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    # transforms.RandAugment(),
                                    # transforms.RandomResizedCrop(32),
                                    # transforms.RandomGrayscale(0.3),
                                    # transforms.RandomHorizontalFlip(0.1),
                                    # transforms.RandomPerspective(0.1),
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    # transforms.RandomInvert(p=0.5),
                                    # transforms.RandomPosterize(bits=2),
                                    # transforms.RandomSolarize(threshold=192.0),
                                    # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                    # transforms.RandomAutocontrast(),
                                    # transforms.RandomEqualize(),
                                    transforms.ToTensor()                              
                                ])
    transform_val = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    # transforms.RandAugment(),
                                    # transforms.RandomResizedCrop(32),
                                    # transforms.RandomGrayscale(0.3),
                                    # transforms.RandomHorizontalFlip(0.5),
                                    # transforms.RandomPerspective(0.2),
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    # transforms.RandomInvert(p=0.5),
                                    # transforms.RandomPosterize(bits=2),
                                    # transforms.RandomSolarize(threshold=192.0),
                                    # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                    # transforms.RandomAutocontrast(),
                                    # transforms.RandomEqualize(),
                                    transforms.ToTensor()                              
                                ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, pin_memory=False, drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=False)
    return trainloader, valloader