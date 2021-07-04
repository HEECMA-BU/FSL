import torchvision
from torchvision import datasets, transforms
import random
import torch

def uploadCOVID(seed_set, device, colab, batch_size):
    # data Transformation
    transform_train = transforms.Compose([
                      transforms.RandomHorizontalFlip(0.5),
                      transforms.RandomRotation(10.0), 
                      transforms.ColorJitter(brightness=0.2, contrast=0.2),
                      transforms.Resize((256,256)),
                      transforms.ToTensor()])
                      #fd

    transform_test = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    # Transform Images
    if colab ==True:
      trainset = torchvision.datasets.ImageFolder('/content/drive/My Drive/images_mod/train', transform=transform_train, target_transform=None)
      valset = torchvision.datasets.ImageFolder('/content/drive/My Drive/images_mod/val', transform=transform_test, target_transform=None)
      #testset = torchvision.datasets.ImageFolder('/content/drive/My Drive/images_mod/test', transform=transform_test, target_transform=None)
    else:
      trainset = torchvision.datasets.ImageFolder('Data/images_mod/train', transform=transform_train, target_transform=None)
      valset = torchvision.datasets.ImageFolder('Data/images_mod/val', transform=transform_test, target_transform=None)
      #testset = torchvision.datasets.ImageFolder('Data/images_mod/test', transform=transform_test, target_transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"), drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle=False, worker_init_fn=seed_set, pin_memory=(device.type == "cuda"))
    return trainloader, valloader
