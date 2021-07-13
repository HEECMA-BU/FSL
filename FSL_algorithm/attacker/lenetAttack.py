import torch
from torch import nn
from torchvision.utils import save_image

# torch.manual_seed(100)
# np.random.seed(100)

import FSL_algorithm.resources.lenet as lenet

target_h = 32 # f_h = 4
target_w = 32 # f_w = 4
m = 10
eps = 10

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # None, 6, 26, 26 
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2) # None, 6, 14, 14
        )
                
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(6, 3, 2, 2),  # None, 3, 26, 26 
            nn.Tanh(),
            nn.ConvTranspose2d(3, 1, 5, 1),  # None, 1, 32, 32
            nn.ReLU()
        )

    def forward(self, x, epoch):
        x = self.encoder(x)
        #print(summary(self.encoder, (1, 32, 32)))
        #print(summary(self.decoder, (6, 14, 14)))
        # pic = to_img(x.data, x.data.shape[2])
        # save_image(pic, './dc_img/encoder_image_{}.png'.format(epoch))
        x = self.decoder(x)
        return x

def to_img(x, dim):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, dim, dim)
        return x

def train_autoencoder(dataloaders, logger, device, wd, constant):
    model = autoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=constant.LR,
                                weight_decay=1e-5)
    # if not os.path.exists('./dc_img'):
    #     os.mkdir('./dc_img')

    

    for epoch in range(constant.ATTACKER_EPOCHS):
        for data in dataloaders['train']:
            img, label = data
            img = img.to(device)

            label_in_list = label.flatten().tolist()
            logger.debug("label:  " + " ".join(str(x) for x in label_in_list))
            # ===================forward=====================
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

        picbefore = to_img(img.data, img.data.shape[2])
        picafter = to_img(output.data, img.data.shape[2])

        save_image(picbefore, wd+'/reconstrucion/attacker/image_{}_before.png'.format(epoch))
        save_image(picafter, wd+'/reconstrucion/attacker/image_{}_after.png'.format(epoch))
    return model




def train_my_extended_model(dataloaders_, logger, device, constant):
    my_extended_model = lenet.get_modelMNIST(10)
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
            print('train epoch [{}/{}], loss:{:.4f}'
                    .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
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
                print('val   epoch [{}/{}], loss:{:.4f}'
                        .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
                logger.debug('val   epoch [{}/{}], loss:{:.4f}'
                        .format(epoch+1, constant.ATTACKER_EPOCHS, loss.item()))
    return my_extended_model
