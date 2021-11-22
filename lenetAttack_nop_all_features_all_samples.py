from uploads.uploadMNIST import uploadMNIST
from uploads.uploadEMNIST_all_features_all_samples import uploadEMNIST
from uploads.uploadCIFAR10 import uploadCIFAR10
from uploads.uploadCIFAR100 import uploadCIFAR100
import random
import constant
import torch
import torchvision
from torchvision import transforms
import torchvision.models as models 
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
import os 
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from diffprivlib.mechanisms import Laplace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.measure import block_reduce
import torch
from torch.nn.functional import avg_pool2d
from torchsummary import summary
# torch.manual_seed(100)
# np.random.seed(100)

from pathlib import Path

import sys

import lenet
import vgg
import logging

class Resize:

    @staticmethod
    def crop_image(img, target_h, target_w):
        shape = img.shape
        
        # Estimate the filter dimensions to get the desired shape
        f_h = shape[0] // (target_h )
        f_w = shape[1] // (target_w )
        
        # Area of the image that will be processed with given filter dimensions
        actual_h = target_h * f_h
        actual_w = target_w * f_w
        
        # Number of pixels that will be cropped in both dimensions
        crop_h = shape[0] - actual_h
        crop_w = shape[1] - actual_w
        
        # Image that will be left after evenly cropping both dimensions
        cropped_img = img[crop_h//2 : crop_h//2 + actual_h, crop_w//2 : crop_w//2 + actual_w]
        return cropped_img, f_h, f_w


    @staticmethod
    def pad_image(img, target_h, target_w):
        shape = img.shape

        # pixels left out from each dimension
        extra_h = shape[1] % target_h
        extra_w = shape[2] % target_w

        # padding required so that dimensions are evenly divisible by target dimensions 
        pad_h = target_h - extra_h if extra_h != 0 else 0
        pad_w = target_w - extra_w if extra_w != 0 else 0

        # Evenly pad in both dimensions
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        padded_img = np.pad(img, ((pad_h_before, pad_h_after),(pad_w_before, pad_w_after), (0,0)), mode='constant')

        new_shape = padded_img.shape
        f_h = new_shape[1] // target_h 
        f_w = new_shape[2] // target_w

        return padded_img, f_h, f_w

class Noise:

    @staticmethod
    def add_gaussian_noise(img, mean, stdev, noise_factor=1):
        img = img / 255.0
        noise = noise_factor * np.random.normal(mean, stdev, img.shape)
        noisy_image = np.clip(img + noise, 0, 1) * 255
        return noisy_image

    @staticmethod
    def add_laplace_noise(img, loc, scale, noise_factor=1):
        img = img / 255.0
        noise = noise_factor *  np.random.laplace(loc, scale, img.shape)
        noisy_image = np.clip(img + noise, 0, 1) * 255
        return noisy_image

class Pixelate:
    
    @staticmethod
    def sequential(img, f_h, f_w):
        target_h = img.shape[0] // f_h
        target_w = img.shape[1] // f_w 
        px = np.zeros((target_h, target_w, img.shape[2]))
        for i in range(target_h):
            row = i * f_h
            for j in range(target_w):
                col = j * f_w
                grid = img[row : row + f_h, col : col + f_w]
                m = np.mean(grid, axis=(0,1))
                px[i,j,:] = m
        return px

    @staticmethod
    def skimage(img, f_h, f_w):
        px = block_reduce(img, (f_h, f_w, 1), func=np.mean)
        return px

    @staticmethod
    def pytorch(img, f_h, f_w):
        img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
        px = avg_pool2d(img, (f_h, f_w))
        px = px.permute(0,2,3,1).squeeze(0).numpy()
        return px

resize_f = Resize.pad_image
pixelate_f = Pixelate.pytorch




epsilon = 1
min_diff = 1e-5

def numpy_to_pillow(img):
    I = Image.fromarray(img.astype(np.uint8))
    return I

def display_image_grid(images, size=(12,12), titles=None, num_cols=4):
    images = list(map(numpy_to_pillow, images))
    fig = plt.figure(figsize=size)
    fig.tight_layout(pad=0)
    N = len(images)
    cols = num_cols
    rows = N/cols if N%cols == 0 else (N//cols + 1)
    for i in range(N):
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[i])
    #plt.show()

def dp_pixelate_images(images, target_h, target_w, m, eps):
    noisy_images = [dp_pixelate(I,target_h, target_w, m, eps, resize_f=resize_f, 
                                  pixelate_f=pixelate_f) for I in images]
    
    #display_image_grid(noisy_images)
    return noisy_images
    

def dp_pixelate(img, target_h, target_w, m, eps, 
                noise_factor = 1, 
                resize_f = Resize.pad_image, 
                pixelate_f = Pixelate.pytorch):
    """
    Input:
        img: numpy array of your image
        target_h: required height of the pixelated image
        target_w: required width of the pixelated image
        eps: privacy parameter
        m: number of pixels to add noise to (see paper)
        noise_factor: scale the noise by this factor (default: 1 i.e., don't scale).
        resize_f: Function to use in order to fit the target dimensions correctly. 
            Resize.pad_image: This function pads 0's at the image boundary. (default)
            Resize.crop_image: This function crops boundary pixels.
        pixelate_f: Function to use for pixelating the image. 
            All the methods below compute same result. They just differ in performance.
                Pixelate.sequential: Slowest
                Pixelate.skimage: Okay
                Pixelate.pytorch: Fastest (default)
    Output:
        Return DP pixelated image with dimension (target_h, target_w, input_channels)
    """

    flag = False
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        flag = True
    
    num_channels = img.shape[0]
    resized_img, f_h, f_w = resize_f(img, target_h, target_w)
    px_img = pixelate_f(resized_img, f_h, f_w)
    
    # distributing eps among channels by eps/num_channels
    scale = (1 * m * num_channels) / (f_h * f_w * eps) 
    dp_px_img = np.zeros(px_img.shape)
    for i in range(num_channels):
        dp_px_img[:,:,i] = Noise.add_laplace_noise(px_img[:,:,i], 0, scale, noise_factor=noise_factor)
    
    if flag:
        dp_px_img = np.squeeze(dp_px_img)

    return dp_px_img.tolist()

target_h = 32 # f_h = 4
target_w = 32 # f_w = 4
m = 10
eps = 10



class myReshape(nn.Module):
    def __init__(self, *args):
        super(myReshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class autoencoder_lenet(nn.Module):
    def __init__(self, n_classes=10, CUTS=[0,3]):
        super(autoencoder_lenet, self).__init__()
        self.CUTS = CUTS
        self.encoder_full = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # [-1, 1, 32, 32] -> [-1, 6, 28, 28] 
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2), # [-1, 6, 28, 28] -> [-1, 6, 14, 14]

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # [-1, 6, 14, 14] -> [-1, 16, 10, 10]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),    # [-1, 16, 10, 10] -> [-1, 16, 5, 5]
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),   # [-1, 16, 5, 5] -> [-1, 120, 1, 1]
            nn.Tanh()
            # nn.Flatten(),
            # nn.Linear(in_features=120, out_features=84),
            # nn.Tanh(),
            # nn.Linear(in_features=84, out_features=n_classes),
            # nn.Softmax(dim = 1)
        )
        modules = []
        for module in self.encoder_full[:CUTS[1]]:
            modules.append(module)
        self.encoder = nn.Sequential(*modules)
                
        self.decoder_full = nn.Sequential(
            # nn.Linear(in_features=n_classes, out_features=84),      # [-1, 10] -> [-1, 84]
            # nn.Tanh(),
            # nn.Linear(in_features=84, out_features=120),            # [-1, 84] -> [-1, 120]
            # myReshape((120, 1, 1)),                           # [-1, 120] -> [-1, 120, 1, 1]
            nn.Tanh(),
            nn.ConvTranspose2d(120, 16, 5, 2),                     # [-1, 120, 1, 1] -> [-1, 16, 5, 5]
            nn.ConvTranspose2d(16, 16, 2, 2),                       # [-1, 16, 5, 5] -> [-1, 16, 10, 10]
            nn.Tanh(),
            nn.ConvTranspose2d(16, 6, 5, 1),                       # [-1, 16, 10, 10] -> [-1, 6, 14, 14]

            nn.ConvTranspose2d(6, 6, 2, 2),                         # None, 6, 28, 28 
            nn.Tanh(),
            nn.ConvTranspose2d(6, 1, 5, 1),                         # None, 1, 32, 32
            nn.ReLU()
        )
        modules = []
        for module in self.decoder_full[-(CUTS[1]+1):]:
            modules.append(module)
        self.decoder = nn.Sequential(*modules)

    def forward(self, x, epoch):
        x = self.encoder(x)
        #print(summary(self.encoder, (1, 32, 32)))
        #print(summary(self.decoder, (6, 14, 14)))
        # pic = to_img(x.data, x.data.shape[2])
        # save_image(pic, './dc_img/encoder_image_{}.png'.format(epoch))
        x = self.decoder(x)
        return x


class autoencoder_vgg16(nn.Module):
    def __init__(self, n_classes=10, CUTS=[0,3]):
        super(autoencoder_vgg16, self).__init__()
        self.CUTS = CUTS
        self.encoder_full = models.vgg16(pretrained=False)
        state_dict = torch.load("vgg16-397923af.pth", map_location="cuda:0")
        self.encoder_full.load_state_dict(state_dict)
        self.encoder_full = self.encoder_full.cuda()
        self.encoder_full = nn.Sequential(*(
            list(list(self.encoder_full.children())[0]) + 
            [nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten()] + 
            list(list(self.encoder_full.children())[2]))
            )
        modules = []
        for module in self.encoder_full[:CUTS[1]]:
            modules.append(module)
        self.encoder = nn.Sequential(*modules)
                
        

#(env) cc@fsl-journal:~$ /home/cc/split-learning/env/bin/python3 /home/cc/split-learning/vgg.py
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#    ConvTranspose2d-1            [-1, 512, 1, 1]       2,359,808
#    ConvTranspose2d-2            [-1, 512, 2, 2]       4,194,816
#               ReLU-3            [-1, 512, 2, 2]               0
#    ConvTranspose2d-4            [-1, 512, 2, 2]       2,359,808
#               ReLU-5            [-1, 512, 2, 2]               0
#    ConvTranspose2d-6            [-1, 512, 2, 2]       2,359,808
#               ReLU-7            [-1, 512, 2, 2]               0
#    ConvTranspose2d-8            [-1, 512, 2, 2]       2,359,808
#    ConvTranspose2d-9            [-1, 512, 4, 4]       6,554,112
#              ReLU-10            [-1, 512, 4, 4]               0
#   ConvTranspose2d-11            [-1, 512, 4, 4]       2,359,808
#              ReLU-12            [-1, 512, 4, 4]               0
#   ConvTranspose2d-13            [-1, 512, 4, 4]       2,359,808
#              ReLU-14            [-1, 512, 4, 4]               0
#   ConvTranspose2d-15            [-1, 256, 4, 4]       1,179,904
#   ConvTranspose2d-16            [-1, 256, 8, 8]       3,211,520
#              ReLU-17            [-1, 256, 8, 8]               0
#   ConvTranspose2d-18            [-1, 256, 8, 8]         590,080
#              ReLU-19            [-1, 256, 8, 8]               0
#   ConvTranspose2d-20            [-1, 256, 8, 8]         590,080
#              ReLU-21            [-1, 256, 8, 8]               0
#   ConvTranspose2d-22            [-1, 128, 8, 8]         295,040
#   ConvTranspose2d-23          [-1, 128, 16, 16]       1,982,592
#              ReLU-24          [-1, 128, 16, 16]               0
#   ConvTranspose2d-25          [-1, 128, 16, 16]         147,584
#              ReLU-26          [-1, 128, 16, 16]               0
#   ConvTranspose2d-27           [-1, 64, 16, 16]          73,792
#   ConvTranspose2d-28           [-1, 64, 32, 32]       1,478,720
#              ReLU-29           [-1, 64, 32, 32]               0
#   ConvTranspose2d-30           [-1, 64, 32, 32]          36,928
#              ReLU-31           [-1, 64, 32, 32]               0
#   ConvTranspose2d-32            [-1, 3, 32, 32]           1,731
# ================================================================
# Total params: 34,495,747
# Trainable params: 34,495,747
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.10
# Forward/backward pass size (MB): 4.48
# Params size (MB): 131.59
# Estimated Total Size (MB): 136.17
# ----------------------------------------------------------------

        k_size = 3
        p_size = 1
        self.decoder_full = nn.Sequential(
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
        modules = []
        for module in self.decoder_full[-(CUTS[1]):]:
            modules.append(module)
        self.decoder = nn.Sequential(*modules)

    def forward(self, x, epoch):
        x = self.encoder(x)
        #print(summary(self.encoder, (1, 32, 32)))
        #print(summary(self.decoder, (6, 14, 14)))
        # pic = to_img(x.data, x.data.shape[2])
        # save_image(pic, './dc_img/encoder_image_{}.png'.format(epoch))
        x = self.decoder(x)
        return x
    
    def conv_layer(self, chann_in, chann_out, k_size, p_size):
        layer = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        )
        return layer

    def vgg_conv_block(self, in_list, out_list, k_list, p_list, pooling_k, pooling_s):

        layers = [ self.conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
        layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
        return nn.Sequential(*layers)

    def vgg_fc_layer(self, size_in, size_out):
        layer = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.ReLU()
        )
        return layer



def to_img(x, dim):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 3, dim, dim)
        return x

def train_autoencoder(dataloaders, logger, device, CUTS, data):
    if data=="mnist":
        model = autoencoder_lenet(CUTS=CUTS)
    if data=="cifar10":
        model = autoencoder_vgg16(CUTS=CUTS)
    model = model.to(device)
    # summary(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=constant.LR,
                                weight_decay=1e-5)
    # if not os.path.exists('./dc_img'):
    #     os.mkdir('./dc_img')

    

    for epoch in range(constant.ATTACKER_EPOCHS):
        for data in dataloaders['train']:
            img, _ = data
            img = img.to(device)
            
            # img = dp_pixelate_images(img, target_h,target_w, m, eps)
            # img = torch.FloatTensor(img)
            #img = Variable(img)
            # ===================forward=====================
            #temp =  np.stack(temp).toTensor
            output = model.forward(img, epoch)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
        logger.debug('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))

        #if epoch == constant.ATTACKER_EPOCHS - 1:
        picbefore = to_img(img.data, img.data.shape[2])
        picafter = to_img(output.data, img.data.shape[2])

        save_image(picbefore, wd+'/reconstrucion/attacker/image_{}_before.png'.format(epoch))
        save_image(picafter, wd+'/reconstrucion/attacker/image_{}_after.png'.format(epoch))
    return model


# encoder = model.encoder
# for param in encoder.parameters():
#     param.requires_grad = False

# class Myencoder(nn.Module):
#     def __init__(self, my_pretrained_model):
#         super(Myencoder, self).__init__()
#         self.pretrained = my_pretrained_model
#         self.my_new_layers = nn.Sequential(
#         nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
#         nn.Tanh(),
#         nn.AvgPool2d(kernel_size=2),
#         nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
#         nn.Tanh(),
#         nn.Flatten(),
#         nn.Linear(in_features=120, out_features=84),
#         nn.Tanh(),
#         nn.Linear(in_features=84, out_features=10),
#         nn.Softmax(dim = 1))
        
    
#     def forward(self, x):
#         x = self.pretrained(x)
#         x = self.my_new_layers(x)
#         return x
# my_extended_model = Myencoder(my_pretrained_model=encoder)
# print(my_extended_model)
# summary(my_extended_model, (1, 32, 32))

def train_my_extended_model(dataloaders_, logger, device, data):
    if data=="mnist":
        my_extended_model = lenet.get_modelMNIST(10)
    if data=="cifar10":
        my_extended_model = vgg.get_modelCIFAR(10)
    
    my_extended_model = my_extended_model.to(device)
    optimizer = torch.optim.Adam(my_extended_model.parameters(), lr=constant.LR,
                                weight_decay=1e-5)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for epoch in range(constant.ATTACKER_EPOCHS):
        my_extended_model.train()
        # my_extended_model.pretrained.requires_grad_(False)
        # my_extended_model.my_new_layers.requires_grad_(True)
        # summary(my_extended_model, (1, 32, 32))
        for img, labels in dataloaders_['train']:
            img = img.to(device)
            labels = labels.to(device)
            criterion = nn.CrossEntropyLoss()
            # ===================forward=====================
            #temp =  np.stack(temp).toTensor
            # torch.flatten(img)
            # print(img.size())
            output = my_extended_model.forward(img)
            # print(output.size())
            # print(labels.size())
            loss = criterion(output, labels)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # ===================log========================
            print('train epoch [{}/{}], loss:{:.4f}'
                    .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
            logger.debug('train epoch [{}/{}], loss:{:.4f}'
                    .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))

        my_extended_model.eval()
        # my_extended_model.pretrained.requires_grad_(False)
        # my_extended_model.my_new_layers.requires_grad_(False)
        # summary(my_extended_model, (1, 32, 32))
        with torch.no_grad():
            # summary(my_extended_model, (1, 32, 32))
            for img, labels in dataloaders_['val']:
                img = img.to(device)
                labels = labels.to(device)
                criterion = nn.CrossEntropyLoss()
                # ===================forward=====================
                #temp =  np.stack(temp).toTensor
                # print(img.size())
                output = my_extended_model.forward(img)
                loss = criterion(output, labels)
                # ===================backward====================
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
            else:
                # ===================log========================
                print('val   epoch [{}/{}], loss:{:.4f}'
                        .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
                logger.debug('val   epoch [{}/{}], loss:{:.4f}'
                        .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
    return my_extended_model






# path1 = wd+'/intermediate/BeforeTrainA/'
# Path(path1).mkdir(parents=True, exist_ok=True)
# path2 = wd+'/source/BeforeTrainA/'
# Path(path2).mkdir(parents=True, exist_ok=True)
# path3 = wd+'/labels/Train/'
# Path(path3).mkdir(parents=True, exist_ok=True)
# path4 = wd+'/intermediate/AfterTrainA/'
# Path(path4).mkdir(parents=True, exist_ok=True)
# path5 = wd+'/source/AfterTrainA/'
# Path(path5).mkdir(parents=True, exist_ok=True)
# path6 = wd+'/intermediate/Val/'
# Path(path6).mkdir(parents=True, exist_ok=True)
# path7 = wd+'/source/Val/'
# Path(path7).mkdir(parents=True, exist_ok=True)
# path8 = wd+'/labels/Val/'
# Path(path8).mkdir(parents=True, exist_ok=True)



def setup_logger(name, log_file):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

if __name__=="__main__":
    # wd = './m1_nop_reconstruction_equal_work_dataset_20_base_500_all_samples_all_features_attacker'
    wd = sys.argv[1]
    Path(wd).mkdir(parents=True, exist_ok=True)

    logs_dirpath = wd+'/logs/attack/'
    Path(logs_dirpath).mkdir(parents=True, exist_ok=True)

    attacker_reconstruction_dir = wd+'/reconstrucion/attacker'
    Path(attacker_reconstruction_dir).mkdir(parents=True, exist_ok=True)

    log_filename_idx = 1
    while os.path.isfile(logs_dirpath+str(log_filename_idx)+'.log'):
        log_filename_idx = log_filename_idx+1
    logger = setup_logger(str(log_filename_idx), logs_dirpath+str(log_filename_idx)+'.log')
    logger.debug("working_dir: " + wd)

    label_dirname =     wd+'/labels/Val/'
    input_dirname =     wd+'/intermediate/Val/'
    output_dir_name =   wd+'/reconstrucion/Val/'
    Path(output_dir_name).mkdir(parents=True, exist_ok=True)

    print("Is cuda available? " + str(torch.cuda.is_available()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if sys.argv[3] == 'cifar10':
        trainloader_, valloader_ = uploadCIFAR10(random.seed(constant.SEED), device, constant.BATCH_SIZE)
        trainloader, valloader = uploadCIFAR100(random.seed(constant.SEED), device, constant.BATCH_SIZE)
    if sys.argv[3] == 'mnist':
        trainloader_, valloader_ = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE, 'equDiff')
        trainloader, valloader = uploadEMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE)
    dataloaders_ = {'train': trainloader_, 'val': valloader_}
    dataloaders = {'train': trainloader, 'val': valloader}

    from os import walk
    equal_count_sum = 0
    used_file_count = 0
    (dirpath, dirname_l, filename_l) = next(walk(input_dirname))
    # filename_l = []
    trials = 5
    print("trials: " + str(trials))
    logger.debug("trials: " + str(trials))
    for i in range(trials):
        logger.debug("trial_: "+str(i))
        model = train_autoencoder(dataloaders, logger, device, CUTS=[0, int(sys.argv[2])], data=sys.argv[3])
        my_extended_model = train_my_extended_model(dataloaders_, logger, device, data=sys.argv[3])
        for filename in filename_l:
            if "19" not in filename.split("_")[0]:
                continue
            used_file_count = used_file_count+1
            intermediate = torch.load(input_dirname+filename, map_location=device)
            # print(intermediate)
            output = model.decoder(intermediate)
            pic = to_img(output.data, output.data.shape[2])
            save_image(pic, output_dir_name+filename+".png")

            output = my_extended_model.forward(output)
            # print("output: " + output.argmax(dim=1, keepdim=True).flatten().tolist())
            output_in_list = output.argmax(dim=1, keepdim=True).flatten().tolist()
            logger.debug("output: " + " ".join(str(x) for x in output_in_list))

            label = torch.load(label_dirname+filename, map_location=device)
            label_in_list = label.flatten().tolist()
            logger.debug("label:  " + " ".join(str(x) for x in label_in_list))

            equal_count = len(torch.nonzero(output.argmax(dim=1, keepdim=True).flatten() == label.flatten(), as_tuple=False).flatten())
            print("equal_count_trial_"+str(i)+"_with_"+filename+": "+str(equal_count))
            logger.debug("equal_count_trial_"+str(i)+"_with_"+filename+": "+str(equal_count))

            equal_count_sum = equal_count_sum + equal_count

    print("equal_count_sum: " + str(equal_count_sum))
    print("total_number_of_img: "+str(used_file_count*constant.BATCH_SIZE))
    accuracy_ratio = equal_count_sum/(used_file_count*constant.BATCH_SIZE)  # (i+1) trials * used_file_count batch count * 32 imgs per batch
    print("accuracy_ratio: " + str(accuracy_ratio))

    logger.debug("equal_count_sum: " + str(equal_count_sum))
    logger.debug("total_number_of_img: "+str((i+1)*used_file_count*constant.BATCH_SIZE))
    logger.info("accuracy_ratio: " + str(accuracy_ratio))