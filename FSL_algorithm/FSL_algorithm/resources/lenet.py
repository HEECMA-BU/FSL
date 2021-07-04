import torch
from torch import nn
from torch import functional as F

def get_modelMNIST_nosyft(n_classes):
    model =  nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=n_classes),
        nn.Softmax(dim = 1))

    return model


def get_modelMNIST(n_classes):
    model =  nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=n_classes),
        nn.Softmax(dim = 1))

    return model

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tan1 = nn.Tanh()
        self.avg1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tan2 = nn.Tanh()
        self.avg2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tan3 = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tan4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tan1(x)
        x = self.avg1(x)
        x = self.conv2(x)
        x = self.avg2(x)
        x = self.conv3(x)
        x = self.tan3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.tan4(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x