import torch
from torch import nn
from torch import functional as F
import torchvision.models as models 

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

def get_modelCIFAR(n_classes):
    model =  nn.Sequential(vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2),
        vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2),
        vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2),
        vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2),
        vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2),
        nn.Flatten([512,512,512],-1),
        vgg_fc_layer(7*7*512, 4096),
        vgg_fc_layer(4096, 4096),
        nn.Linear(4096, n_classes))
    return model

def get_modelCIFAR(n_classes):
    vgg16 = models.vgg16(pretrained=True)
    model =  vgg16_seq = nn.Sequential(*(
        list(list(vgg16.children())[0]) + 
        [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()] + 
        list(list(vgg16.children())[2]))
        )
    return model

class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


if __name__ == "__main__":
    mod = get_modelCIFAR(10)