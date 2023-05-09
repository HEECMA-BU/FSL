import torch
from torch import nn
from torch import functional as F


class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()

        self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        # nn.ReLU(nn.Parameter(torch.zeros(32), requires_grad=True)),
        self.relu_0 = nn.ReLU()
        self.maxpooling_0 = nn.MaxPool2d(kernel_size=2)

        self.conv2d_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        # nn.ReLU(nn.Parameter(torch.zeros(64), requires_grad=True)),
        self.relu_1 = nn.ReLU()
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc_0 = nn.Linear(in_features=7*7*64, out_features=1024) # input shape [28, 28, 1]

        self.dropout = nn.Dropout(p=0.5)

        self.fc_1 = nn.Linear(in_features=1024, out_features=n_classes)
        self.softmax = nn.Softmax(dim = 1)    


        # ReLU
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, n_classes):
        # x = x + self.bias
        # x = self.relu(x) 

        x = self.conv2d_0(x)
        x = self.relu_0(x)
        x = self.maxpooling_0(x)

        x = self.conv2d_1(x)
        # nn.ReLU(nn.Parameter(torch.zeros(64), requires_grad=True)),
        x = self.relu_1(x)
        x = self.maxpooling_1(x)

        x = self.flatten(x)
        x = self.fc_0(x)

        x = self.dropout(x)

        x = self.fc_1(x)
        x = self.softmax(x)


        # # ReLU
        # self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.relu = nn.ReLU(inplace=True)

        
        return x

class BiasedReLU(nn.Module):
    def __init__(self, size):
        super(BiasedReLU, self).__init__()
        self.size = size
        # ReLU
        # temp_bias = nn.Parameter(torch.zeros([size]), requires_grad=True)
        # self.bias = torch.reshape(temp_bias, (-1, size, 1, 1))
        self.bias = nn.Parameter(torch.zeros(size, requires_grad=True))
        # self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = x + torch.reshape(self.bias, (-1, self.size, 1, 1))
        x = x + self.bias
        x = self.relu(x)        
        return x


def get_modelVPN(n_classes):
    model =  nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
        # nn.ReLU(nn.Parameter(torch.zeros(32), requires_grad=True)),
        # BiasedReLU(32),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        # nn.ReLU(nn.Parameter(torch.zeros(64), requires_grad=True)),
        # BiasedReLU(64),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(in_features=7*7*64, out_features=1024), # input shape [28, 28, 1]

        nn.Dropout(p=0.5),

        nn.Linear(in_features=1024, out_features=n_classes),
        nn.Softmax(dim = 1)
    )

    return model

