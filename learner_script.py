import torch
import random
from FSL_algorithm.resources.config import Config as constant

from FSL_algorithm.resources.functions import set_seed
# from FSL_algorithm.resources.uploads.uploadMNIST import uploadMNIST
from FSL_algorithm.resources.uploads.uploadMNIST_A_B_C_features_all_samples import uploadMNIST
# from FSL_algorithm.resources.uploads.uploadCIFAR10 import uploadCIFAR10
from FSL_algorithm.resources.uploads.uploadCIFAR10_A_B_C_features_all_samples import uploadCIFAR10
from FSL_algorithm.resources.uploads.VpnDataLoader.uploadVPN import uploadVPN
import sys

def main():
   config = {}
   for idx in range(3, len(sys.argv)):
      if idx%2 == 0:
         continue
      if sys.argv[idx] == "CUTS":
         config[sys.argv[idx]] = [0, int(sys.argv[idx+1])]
      else:
         config[sys.argv[idx]] = sys.argv[idx+1]
   constant.load_config(config)
   # print(constant)

# # =========================
# # adding config's
#    constant.CUTS=[0,4]
# # =========================

   print("Is cuda available? " + str(torch.cuda.is_available()))
   device = torch.device('cuda:'+sys.argv[2] if torch.cuda.is_available() else 'cpu')

   print(sys.argv)
   if sys.argv[1] =='mnist':
      set_seed(constant.SEED)
      if constant.DATA_DIST == 'diffDistrmodel1':
         trainloader0, trainloader1, valloader, valloader0, valloader1 = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
         dataloaders = {'train': [trainloader0, trainloader1], 'val': [valloader, valloader0, valloader1]}
      else:
         if constant.DATA_DIST == 'diffDistrmodel3':
            trainloader, trainloader1, valloader = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
            dataloaders = {'train': [trainloader, trainloader1], 'val': [valloader]}
         else: # constant.DATA_DIST == 'equDiff'
            trainloader, valloader = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
            dataloaders = {'train': [trainloader], 'val': [valloader]}

   if sys.argv[1] == 'cifar10':
      set_seed(constant.SEED)
      if constant.DATA_DIST == 'diffDistrmodel1':
         trainloader0, trainloader1, valloader, valloader0, valloader1 = uploadCIFAR10(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
         dataloaders = {'train': [trainloader0, trainloader1], 'val': [valloader, valloader0, valloader1]}
      else:
         if constant.DATA_DIST == 'diffDistrmodel3':
            trainloader, trainloader1, valloader = uploadCIFAR10(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
            dataloaders = {'train': [trainloader, trainloader1], 'val': [valloader]}
         else:
            trainloader, valloader = uploadCIFAR10(random.seed(constant.SEED), device, constant.BATCH_SIZE, constant.DATA_DIST)
            dataloaders = {'train': [trainloader], 'val': [valloader]}
   if sys.argv[1] == 'vpn':
      set_seed(constant.SEED)
      trainloader, valloader = uploadCIFAR10(random.seed(constant.SEED), device, constant.BATCH_SIZE)
      dataloaders = {'train': trainloader, 'val': valloader}

   print('start')
   models = constant.MODELS
   for i, model in enumerate(models):
      print(i, model)
      model.run_model(device, dataloaders, sys.argv[1], constant)



# Execute main
if __name__=="__main__":
   main()
