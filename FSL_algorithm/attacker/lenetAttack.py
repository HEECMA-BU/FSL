import torch
from torch import nn
from torchvision.utils import save_image
import torchvision.models as models 


# torch.manual_seed(100)
# np.random.seed(100)

import FSL_algorithm.resources.lenet as lenet
import FSL_algorithm.resources.vgg as vgg

target_h = 32 # f_h = 4
target_w = 32 # f_w = 4
m = 10
eps = 10

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
    def __init__(self, device, n_classes=10, CUTS=[0,3]):
        super(autoencoder_vgg16, self).__init__()
        self.CUTS = CUTS
        self.encoder_full = vgg.get_modelCIFAR(n_classes, device)
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

def train_autoencoder(dataloaders, logger, device, wd, constant):
    if constant.data=="mnist":
        model = autoencoder_lenet(CUTS=constant.CUTS)
    if constant.data=="cifar10":
        model = autoencoder_vgg16(device, CUTS=constant.CUTS)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=constant.LR,
                                weight_decay=1e-5)
    # if not os.path.exists('./dc_img'):
    #     os.mkdir('./dc_img')

    

    for epoch in range(constant.ATTACKER_EPOCHS):
        for data in dataloaders['train']:
            img, _ = data
            img = img.to(device)

            # label_in_list = label.flatten().tolist()
            # logger.debug("label:  " + " ".join(str(x) for x in label_in_list))
            # ===================forward=====================
            output = model.forward(img, epoch)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        # print('epoch [{}/{}], loss:{:.4f}'
        #     .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
        logger.debug('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))

        picbefore = to_img(img.data, img.data.shape[2])
        picafter = to_img(output.data, img.data.shape[2])

        save_image(picbefore, wd+'/reconstrucion/attacker_'+constant.INTERMEDIATE_DATA_DIR+'image_{}_before.png'.format(epoch))
        save_image(picafter, wd+'/reconstrucion/attacker_'+constant.INTERMEDIATE_DATA_DIR+'image_{}_after.png'.format(epoch))
    return model




def train_my_extended_model(dataloaders_, logger, device, constant):
    if constant.data=="mnist":
        my_extended_model = lenet.get_modelMNIST(10)
    if constant.data=="cifar10":
        my_extended_model = vgg.get_modelCIFAR(10, device)
    
    my_extended_model = my_extended_model.to(device)
    optimizer = torch.optim.Adam(my_extended_model.parameters(), lr=constant.LR,
                                weight_decay=1e-5)
    for epoch in range(constant.ATTACKER_EPOCHS):
        my_extended_model.train()
        for img, labels in dataloaders_['train']:
            img = img.to(device)
            labels = labels.to(device)
            criterion = nn.CrossEntropyLoss()
            # ===================forward=====================
            output = my_extended_model.forward(img)
            loss = criterion(output, labels)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # ===================log========================
            # print('train epoch [{}/{}], loss:{:.4f}'
            #         .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
            logger.debug('train epoch [{}/{}], loss:{:.4f}'
                    .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))

        my_extended_model.eval()
        with torch.no_grad():
            for img, labels in dataloaders_['val']:
                img = img.to(device)
                labels = labels.to(device)
                criterion = nn.CrossEntropyLoss()
                # ===================forward=====================
                output = my_extended_model.forward(img)
                loss = criterion(output, labels)
                # ===================backward====================
            else:
                # ===================log========================
                # print('val   epoch [{}/{}], loss:{:.4f}'
                #         .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
                logger.debug('val   epoch [{}/{}], loss:{:.4f}'
                        .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
    return my_extended_model
