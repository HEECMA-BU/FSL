import torch
import random
from FSL_algorithm.resources.config import Config as constant

from FSL_algorithm.resources.functions import set_seed
from FSL_algorithm.resources.uploads.uploadMNIST import uploadMNIST
import sys

def main():
   config = {}
   constant.load_config(config)

   print("Is cuda available? " + str(torch.cuda.is_available()))
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   print(sys.argv)
   if sys.argv[1] =='mnist':
      set_seed(constant.SEED)
      trainloader, valloader = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE, 'equDiff')
      dataloaders = {'train': trainloader, 'val': valloader}

   print('start')
   models = constant.MODELS
   for i, model in enumerate(models):
      print(i, model)
      model.run_model(device,dataloaders, sys.argv[1], constant)



# Execute main
if __name__=="__main__":
   main()
