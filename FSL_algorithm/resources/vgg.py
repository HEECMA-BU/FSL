import torch
from torch import nn
from torch import functional as F
import torchvision.models as models 
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}
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
# def get_modelCIFAR(n_classes):
#     model =  nn.Sequential(vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2),
#         vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2),
#         vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2),
#         vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2),
#         vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2),
#         nn.Flatten([512,512,512],-1),
#         vgg_fc_layer(7*7*512, 4096),
#         vgg_fc_layer(4096, 4096),
#         nn.Linear(4096, n_classes))
#     return model
def get_modelCIFAR(n_classes, device):
    model = models.vgg16(pretrained=False)
    state_dict = torch.load("FSL_algorithm/resources/vgg16-397923af.pth", map_location=device)
    # state_dict = torch.load("vgg16-397923af.pth", map_location="cuda:1")
    # state_dict = torch.load("vgg16-397923af.pth")
    model.load_state_dict(state_dict)
    # model = model.cuda()
    model = nn.Sequential(*(
        list(list(model.children())[0]) + 
        [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()] + 
        list(list(model.children())[2][:-1]) + 
        [nn.Linear(4096, n_classes)])
        )
    model = model.to(device)
    return model
def add_small_filter(model, CUT_idxs):
    modules=[]
    for idx, layer in model.named_children():
        idx = int(idx)
        modules.append(layer)
        if idx+1 in CUT_idxs:
            modules.append(
                nn.Conv2d(last_output_size, 1, kernel_size=3, padding=1)
            )
            modules.append(
                nn.ReLU()
            )
            modules.append(
                nn.Conv2d(1, last_output_size, kernel_size=3, padding=1)
            )
            modules.append(
                nn.ReLU()
            )
        if 'weight' in layer._parameters:
            last_output_size = layer._parameters['weight'].size()[0]
    sequential = nn.Sequential(*modules)
    sequential = sequential.cuda()
    return sequential


def add_batchnorm(model, device):
    modules=[]
    # list(model_all.named_children())[0][1]._get_name()
    length = len(model)
    for idx, layer in model.named_children():
        idx = int(idx)
        if  idx == length-1:
            modules.append(layer)
        else:
            if layer._get_name()=="Conv2d":
                modules.append(layer)
                modules.append(nn.BatchNorm2d(layer._parameters['weight'].size()[0]))
            elif layer._get_name()=="Linear":
                modules.append(layer)
                modules.append(nn.BatchNorm1d(layer._parameters['weight'].size()[0]))
            else:
                modules.append(layer)
    sequential = nn.Sequential(*modules)
    sequential = sequential.cuda()
    return sequential

def remove_dropout(model):
    modules=[]
    for idx, module in enumerate(model):
        if idx!=35 and idx!=38:
            modules.append(module)
    sequential = nn.Sequential(*modules)
    return sequential
    
# >>> summary(mod.cuda(), (3, 32, 32))
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 32, 32]           1,792
#        BatchNorm2d-2           [-1, 64, 32, 32]             128
#               ReLU-3           [-1, 64, 32, 32]               0
#             Conv2d-4           [-1, 64, 32, 32]          36,928
#        BatchNorm2d-5           [-1, 64, 32, 32]             128
#               ReLU-6           [-1, 64, 32, 32]               0
#          MaxPool2d-7           [-1, 64, 16, 16]               0
#             Conv2d-8          [-1, 128, 16, 16]          73,856
#        BatchNorm2d-9          [-1, 128, 16, 16]             256
#              ReLU-10          [-1, 128, 16, 16]               0
#            Conv2d-11          [-1, 128, 16, 16]         147,584
#       BatchNorm2d-12          [-1, 128, 16, 16]             256
#              ReLU-13          [-1, 128, 16, 16]               0
#         MaxPool2d-14            [-1, 128, 8, 8]               0
#            Conv2d-15            [-1, 256, 8, 8]         295,168
#       BatchNorm2d-16            [-1, 256, 8, 8]             512
#              ReLU-17            [-1, 256, 8, 8]               0
#            Conv2d-18            [-1, 256, 8, 8]         590,080
#       BatchNorm2d-19            [-1, 256, 8, 8]             512
#              ReLU-20            [-1, 256, 8, 8]               0
#            Conv2d-21            [-1, 256, 8, 8]         590,080
#       BatchNorm2d-22            [-1, 256, 8, 8]             512
#              ReLU-23            [-1, 256, 8, 8]               0
#         MaxPool2d-24            [-1, 256, 4, 4]               0
#            Conv2d-25            [-1, 512, 4, 4]       1,180,160
#       BatchNorm2d-26            [-1, 512, 4, 4]           1,024
#              ReLU-27            [-1, 512, 4, 4]               0
#            Conv2d-28            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-29            [-1, 512, 4, 4]           1,024
#              ReLU-30            [-1, 512, 4, 4]               0
#            Conv2d-31            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-32            [-1, 512, 4, 4]           1,024
#              ReLU-33            [-1, 512, 4, 4]               0
#         MaxPool2d-34            [-1, 512, 2, 2]               0
#            Conv2d-35            [-1, 512, 2, 2]       2,359,808
#       BatchNorm2d-36            [-1, 512, 2, 2]           1,024
#              ReLU-37            [-1, 512, 2, 2]               0
#            Conv2d-38            [-1, 512, 2, 2]       2,359,808
#       BatchNorm2d-39            [-1, 512, 2, 2]           1,024
#              ReLU-40            [-1, 512, 2, 2]               0
#            Conv2d-41            [-1, 512, 2, 2]       2,359,808
#       BatchNorm2d-42            [-1, 512, 2, 2]           1,024
#              ReLU-43            [-1, 512, 2, 2]               0
#         MaxPool2d-44            [-1, 512, 1, 1]               0
# AdaptiveAvgPool2d-45            [-1, 512, 7, 7]               0
#           Flatten-46                [-1, 25088]               0
#            Linear-47                 [-1, 4096]     102,764,544
#       BatchNorm1d-48                 [-1, 4096]           8,192
#              ReLU-49                 [-1, 4096]               0
#           Dropout-50                 [-1, 4096]               0
#            Linear-51                 [-1, 4096]      16,781,312
#       BatchNorm1d-52                 [-1, 4096]           8,192
#              ReLU-53                 [-1, 4096]               0
#           Dropout-54                 [-1, 4096]               0
#            Linear-55                   [-1, 10]          40,970
# ================================================================
# Total params: 134,326,346
# Trainable params: 134,326,346
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 7.20
# Params size (MB): 512.41
# Estimated Total Size (MB): 519.63
# ----------------------------------------------------------------
# >>> summary(mod.cuda(), (3, 32, 32))
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 32, 32]           1,792
#               ReLU-2           [-1, 64, 32, 32]               0
#             Conv2d-3           [-1, 64, 32, 32]          36,928
#               ReLU-4           [-1, 64, 32, 32]               0
#          MaxPool2d-5           [-1, 64, 16, 16]               0
#             Conv2d-6          [-1, 128, 16, 16]          73,856
#               ReLU-7          [-1, 128, 16, 16]               0
#             Conv2d-8          [-1, 128, 16, 16]         147,584
#               ReLU-9          [-1, 128, 16, 16]               0
#         MaxPool2d-10            [-1, 128, 8, 8]               0
#            Conv2d-11            [-1, 256, 8, 8]         295,168
#              ReLU-12            [-1, 256, 8, 8]               0
#            Conv2d-13            [-1, 256, 8, 8]         590,080
#              ReLU-14            [-1, 256, 8, 8]               0
#            Conv2d-15            [-1, 256, 8, 8]         590,080
#              ReLU-16            [-1, 256, 8, 8]               0
#         MaxPool2d-17            [-1, 256, 4, 4]               0
#            Conv2d-18            [-1, 512, 4, 4]       1,180,160
#              ReLU-19            [-1, 512, 4, 4]               0
#            Conv2d-20            [-1, 512, 4, 4]       2,359,808
#              ReLU-21            [-1, 512, 4, 4]               0
#            Conv2d-22            [-1, 512, 4, 4]       2,359,808
#              ReLU-23            [-1, 512, 4, 4]               0
#         MaxPool2d-24            [-1, 512, 2, 2]               0
#            Conv2d-25            [-1, 512, 2, 2]       2,359,808
#              ReLU-26            [-1, 512, 2, 2]               0
#            Conv2d-27            [-1, 512, 2, 2]       2,359,808
#              ReLU-28            [-1, 512, 2, 2]               0
#            Conv2d-29            [-1, 512, 2, 2]       2,359,808
#              ReLU-30            [-1, 512, 2, 2]               0
#         MaxPool2d-31            [-1, 512, 1, 1]               0
# AdaptiveAvgPool2d-32            [-1, 512, 7, 7]               0
#           Flatten-33                [-1, 25088]               0
#            Linear-34                 [-1, 4096]     102,764,544
#              ReLU-35                 [-1, 4096]               0
#           Dropout-36                 [-1, 4096]               0
#            Linear-37                 [-1, 4096]      16,781,312
#              ReLU-38                 [-1, 4096]               0
#           Dropout-39                 [-1, 4096]               0
#            Linear-40                 [-1, 1000]       4,097,000
# ================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 5.03
# Params size (MB): 527.79
# Estimated Total Size (MB): 532.84
# ----------------------------------------------------------------
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
    k_size = 3
    p_size = 1
    mod = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=4),    #           AdaptiveAvgPool2d-32
            nn.ConvTranspose2d(512, 512, kernel_size=4, padding=1),
            nn.ReLU(),                                                                      #           ReLU-30
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=p_size),
            nn.ConvTranspose2d(512, 512, kernel_size=5, padding=p_size),    #           MaxPool2d-24
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=k_size, padding=p_size),
            nn.ConvTranspose2d(256, 256, kernel_size=7, padding=p_size),    #           MaxPool2d-17
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=k_size, padding=p_size),
            nn.ConvTranspose2d(128, 128, kernel_size=11, padding=p_size),    #           MaxPool2d-10
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=k_size, padding=p_size),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=k_size, padding=p_size),
            nn.ConvTranspose2d(64, 64, kernel_size=19, padding=p_size),    #           MaxPool2d-5
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        )
    from torchsummary import summary
    summary(mod.cuda(), (512, 7, 7))