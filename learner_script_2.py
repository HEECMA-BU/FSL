import torch
import random
from FSL_algorithm.resources.config import Config as constant

from FSL_algorithm.resources.functions import set_seed
from FSL_algorithm.resources.uploads.uploadMNIST import uploadMNIST
from FSL_algorithm.resources.uploads.uploadCIFAR10 import uploadCIFAR10
import sys

def main():
   config = {}
   for idx in range(3, len(sys.argv)):
      if idx%2 == 0:
         continue
      config[sys.argv[idx]] = sys.argv[idx+1]
   constant.load_config(config)
   # print(constant)

# =========================
# adding config's
   constant.CUTS=[0,16]
# =========================

   print("Is cuda available? " + str(torch.cuda.is_available()))
   device = torch.device('cuda:'+sys.argv[2] if torch.cuda.is_available() else 'cpu')

   print(sys.argv)
   if sys.argv[1] =='mnist':
      set_seed(constant.SEED)
      trainloader, valloader = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE)
      dataloaders = {'train': trainloader, 'val': valloader}
   if sys.argv[1] == 'cifar10':
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
