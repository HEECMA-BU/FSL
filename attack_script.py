import FSL_algorithm.attacker.lenetAttack as lenetAttack
from FSL_algorithm.resources.uploads.uploadMNIST import uploadMNIST
from FSL_algorithm.resources.uploads.uploadEMNIST_all_features_all_samples import uploadEMNIST
from pathlib import Path
import os
import logging 
import random
import torch 
from torchvision.utils import save_image
from FSL_algorithm.resources.config import Config as constant
def setup_logger(name, log_file):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


wd = constant.WD
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

label_dirname =     wd+'/labels/Train/'
input_dirname =     wd+'/intermediate/'+constant.INTERMEDIATE_DATA_DIR
output_dir_name =   wd+'/reconstrucion/Train/'
Path(output_dir_name).mkdir(parents=True, exist_ok=True)

print("Is cuda available? " + str(torch.cuda.is_available()))
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
trainloader_, valloader_ = uploadMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE)
trainloader, valloader = uploadEMNIST(random.seed(constant.SEED), device, constant.BATCH_SIZE)
dataloaders_ = {'train': trainloader_, 'val': valloader_}
dataloaders = {'train': trainloader, 'val': valloader}

from os import walk
equal_count_sum = 0
(dirpath, dirname_l, filename_l) = next(walk(input_dirname))
# filename_l = []
trials = 150
print("trials: " + str(trials))
logger.debug("trials: " + str(trials))
for i in range(trials):
    logger.debug("trial_: "+str(i))
    model = lenetAttack.train_autoencoder(dataloaders, logger, device, wd, constant)
    my_extended_model = lenetAttack.train_my_extended_model(dataloaders_, logger, device, constant)
    for filename in filename_l:
        if "19" not in filename.split("_")[0]:
            continue
        intermediate = torch.load(input_dirname+filename, map_location=device)
        # print(intermediate)
        output = model.decoder(intermediate)
        pic = lenetAttack.to_img(output.data, output.data.shape[2])
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
print("total_number_of_img: "+str((i+1)*1875*32))
accuracy_ratio = equal_count_sum/((i+1)*1875*32)  # (i+1) trials * 1875 batch count * 32 imgs per batch
print("accuracy_ratio: " + str(accuracy_ratio))

logger.debug("equal_count_sum: " + str(equal_count_sum))
logger.debug("total_number_of_img: "+str((i+1)*1875*32))
logger.info("accuracy_ratio: " + str(accuracy_ratio))