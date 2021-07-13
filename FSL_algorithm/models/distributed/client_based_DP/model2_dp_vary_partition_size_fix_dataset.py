#######################################################################################################################
# Federated Split Learning: vary partition size and fix dataset
# note: change constant.CLIENTS to control partition size, i.e., partition the fixed dataset to how many parts
#######################################################################################################################
import logging
import math
import torch
from time import time

from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
from torch import nn, optim
import syft as sy
import time
from sklearn.metrics import confusion_matrix, f1_score
from FSL_algorithm.resources.lenet import get_modelMNIST

from FSL_algorithm.resources.setup import setup2, average_weights
from FSL_algorithm.resources.functions import make_prediction, total_time_train

from opacus import PrivacyEngine

from FSL_algorithm.resources.classes import MultiSplitNN

from pathlib import Path
import os

DEBUG = False

# hook = sy.TorchHook(torch)


def setup_logger(name, log_file):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


def train(x, target, model):
    model.zero_grads()

    pred, forward_clients, forward_server, intermediate = model.forward(x)
    begin_server = time.time()
    # without noPeek
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, target)
    loss.backward()
    
    backward_server = time.time() - begin_server
    backward_client = model.backward()
    model.step()    
    return loss, pred, forward_clients+backward_client, forward_server+backward_server, intermediate

def run_model(device, dataloaders, data, constant):
    if(torch.cuda.is_available()== True):
        torch.cuda.reset_max_memory_allocated()
    wd = os.path.join(constant.PD, 'm2_dp_'+str(constant.PARAM)+'_reconstruction_vary_partition_size_fix_dataset_'+str(constant.CLIENTS)+"_base_"+str(constant.MAXCLIENTS))+"_dist"
    Path(wd).mkdir(parents=True, exist_ok=True)

    logs_dirpath = wd+'/logs/train/'
    Path(logs_dirpath).mkdir(parents=True, exist_ok=True)

    log_filename_idx = 1
    while os.path.isfile(logs_dirpath+str(log_filename_idx)+'.log'):
        log_filename_idx = log_filename_idx+1
    logger = setup_logger(str(log_filename_idx), logs_dirpath+str(log_filename_idx)+'.log')

    # print(wd, "  ", constant.CLIENTS, "    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.debug("Is cuda available? " + str(torch.cuda.is_available()))
    #File
    path1 = wd+'/intermediate/Train/'
    Path(path1).mkdir(parents=True, exist_ok=True)
    path2 = wd+'/source/Train/'
    Path(path2).mkdir(parents=True, exist_ok=True)
    path3 = wd+'/labels/Train/'
    Path(path3).mkdir(parents=True, exist_ok=True)
    # path4 = wd+'/intermediate/AfterTrainA/'
    # Path(path4).mkdir(parents=True, exist_ok=True)
    # path5 = wd+'/source/AfterTrainA/'
    # Path(path5).mkdir(parents=True, exist_ok=True)
    path6 = wd+'/intermediate/Val/'
    Path(path6).mkdir(parents=True, exist_ok=True)
    path7 = wd+'/source/Val/'
    Path(path7).mkdir(parents=True, exist_ok=True)
    path8 = wd+'/labels/Val/'
    Path(path8).mkdir(parents=True, exist_ok=True)
    
    hook = sy.TorchHook(torch)
    sy.local_worker.is_client_worker = False
    
    #Set up Model
    if (data == 'mnist'):
        model_all = get_modelMNIST(10)
    # if (data == 'covid'):
    #     model_all = get_modelCOVID()
   
    #Split Original Model
    local_models, models = setup2(model_all, device, constant)

    #Optimizer
    optimizers = {}
    for i in range(constant.CLIENTS):
        optimizers['optimizer{}'.format(i+1)] =  [ 
            torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
            if constant.OPTIMIZER == 'Adam'
            else
            torch.optim.SGD(model.parameters(), lr=0.03,)
            for model in models['models{}'.format(i+1)]]

    # Distributed Workers
    # addr_array = ["http://10.52.3.229:", "http://10.52.1.120:"]
    addr_array = ["http://localhost:", "http://localhost:"]
    port_array = [[7600, 7602], [7601, 7603]]
    # client_array = []
    client_array = [[] for i in range(constant.CLIENTS)]
    # clientname_array = []
    for k in range(constant.CLIENTS):
        for i in range(constant.THOR):
            # sy.VirtualWorker(hook, id="client{}{}".format(k+1, i))
            remote_client = DataCentricFLClient(hook, addr_array[k] + str(port_array[k][i]))
            client_array[k].append(remote_client)
            # clientname_array.append()

    flat_list = [item for sublist in client_array for item in sublist]
    my_grid = sy.PrivateGridNetwork(*flat_list)

    #Create splitNN model
    splitNNs = {}
    for i in range(constant.CLIENTS):
        splitNNs['splitNN{}'.format(i+1)] = MultiSplitNN(models['models{}'.format(i+1)], optimizers['optimizer{}'.format(i+1)])

    running_loss = []
    lun =[]
    for i in range(constant.CLIENTS):
            running_loss.append(0)
            lun.append(0)
    
    counter = 0
    epoch=0
    best_f1_score=0
    privacy_engines = []
    
    while(epoch < constant.EPOCHS):
        logger.debug('Epoch {}/{}'.format(epoch, constant.EPOCHS - 1))
        logger.debug('-' * 10)

        #Training
        target_tot = []
        pred_tot = []
        total_time_client = 0
        total_time_server = 0

        since = time.time()
        for i in range(constant.CLIENTS):
            running_loss[i]=0
            lun[i]=0

        if (epoch == 0):
            for k in range(constant.CLIENTS):
                splitNNs['splitNN{}'.format(k+1)].train()
                privacy_engines_line = []
                for i in range(constant.THOR):
                    optimizers['optimizer{}'.format(k+1)][i] = optim.AdamW(models['models{}'.format(k+1)][i].parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                    privacy_engine = PrivacyEngine(models['models{}'.format(k+1)][i],
                                    batch_size=constant.BATCH_SIZE, 
                                    sample_size=len(dataloaders['train'].dataset), 
                                    alphas=range(2,32), 
                                    noise_multiplier=constant.PARAM,
                                    max_grad_norm=1.0)
                    privacy_engine.attach(optimizers['optimizer{}'.format(k+1)][i])
                    privacy_engines_line.append(privacy_engine)
                    models['models{}'.format(k+1)][i] = models['models{}'.format(k+1)][i].send(client_array[k][i])
                privacy_engines.append(privacy_engines_line)

        for i in range(constant.THOR):
                models["models1"][i].train()

        expected_batches = constant.CLIENTS*len(dataloaders['train'])/constant.CLIENTS
        j=0
        for filling_data in range(math.ceil(constant.CLIENTS/constant.CLIENTS)):
            for idx, (images, labels) in enumerate(dataloaders['train']):
                images = images.to(device)
                labels = labels.to(device)
                images = images.send(models['models{}'.format((j%constant.CLIENTS)+1)][0].location)
                labels = labels.send(models['models{}'.format((j%constant.CLIENTS)+1)][constant.THOR-1].location)
                loss, output, end_clients, end_server, intermediate = train(images, labels, splitNNs['splitNN{}'.format((j%constant.CLIENTS)+1)])
                temp = loss.get()
                running_loss[j%constant.CLIENTS] += float(temp)
                labels = labels.get()
                output = output.get()
                target_tot, pred_tot =  make_prediction(output, labels.tolist(), target_tot, pred_tot)

                total_time_client += end_clients
                total_time_server += end_server

    ##############################################################
                intermediate = intermediate.get()
                images = images.get()
                if epoch==constant.EPOCHS-1:
                    torch.save(intermediate,   path1+str(epoch)+'_Client'+str(idx)+'.pt')
                    torch.save(labels,                      path3+str(epoch)+'_Client'+str(idx)+'.pt')
    ##############################################################
                lun[j%constant.CLIENTS]=lun[j%constant.CLIENTS]+len(images)

                del intermediate
                del images
                del labels
                sy.local_worker.clear_objects()

                j=j+1
                if j>=expected_batches:
                    logger.debug('Batches: {}, Expected_Batches: {}, constant.CLIENTS: {}, constant.MAXCLIENTS: {}, Total_Batches: {}'.format(j, expected_batches, constant.CLIENTS, constant.MAXCLIENTS, len(dataloaders['train'])))
                    break
        else:
            target_tot_ = sum(target_tot, [])
            pred_tot_ = sum(pred_tot, [])
            cm1 = confusion_matrix(target_tot_, pred_tot_)
            if data == 'mnist':
                preds = torch.FloatTensor(pred_tot_)
                targets = torch.FloatTensor(target_tot_)
                acc = preds.eq(targets).float().mean() 
                # print(cm1)
                logger.debug(cm1)
                for i in range(constant.CLIENTS):
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss[i%constant.CLIENTS]/lun[i%constant.CLIENTS], acc))
            if data == 'covid':
                preds = torch.FloatTensor(pred_tot_)
                targets = torch.FloatTensor(target_tot_)
                f1_score_value = f1_score(pred_tot_, target_tot_)
                # print(cm1)
                logger.debug(cm1)
                for i in range(constant.CLIENTS):
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss[i%constant.CLIENTS]/lun[i%constant.CLIENTS], f1_score_value))

            with torch.no_grad():
                #Average weights
                averaging_time = time.time()
                average_weights(models, local_models, 'False', constant)
                averaging_time = time.time() - averaging_time

                #Total Time for one epoch
                total_time_train(since, epoch, total_time_client, total_time_server+averaging_time, logger, "train")
                
                #Validation
                for k in range(constant.CLIENTS):
                    models['models{}'.format(k+1)][1].send(client_array[k+1][1])
                    # models['models{}'.format(k+1)][1] = models['models{}'.format(k+1)][1].send('client{}{}'.format(k+1, 1))
                    optimizers['optimizer{}'.format(k+1)][1] = optim.AdamW(models['models{}'.format(k+1)][1].parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                    # privacy_engine = PrivacyEngine(models['models{}'.format(k+1)][1],
                    #     batch_size=constant.BATCH_SIZE, 
                    #     sample_size=len(dataloaders['train'].dataset), 
                    #     alphas=range(2,32), 
                    #     noise_constant.PARAM=constant.PARAM,
                    #     max_grad_norm=1.0)
                    privacy_engines[k][1].detach()
                    privacy_engines[k][1].attach(optimizers['optimizer{}'.format(k+1)][1])

                for i in range(constant.THOR):
                    models["models1"][i].eval()
            
                target_tot=[]
                pred_tot=[]
                j = 0
                running_loss_val = 0
                with torch.set_grad_enabled(True):
                    expected_batches = constant.CLIENTS*len(dataloaders['val'])/constant.MAXCLIENTS
                    for filling_data in range(math.ceil(constant.CLIENTS/constant.CLIENTS)):
                        for idx, (images, labels) in enumerate(dataloaders['val']):
                            images = images.to(device)
                            labels = labels.to(device)
                            images = images.send(models['models{}'.format((j%constant.CLIENTS)+1)][0].location)
                            output, intermediate = splitNNs['splitNN{}'.format((j%constant.CLIENTS)+1)].forwardVal(images)
        ##############################################################
                            intermediate = intermediate.get()
                            images = images.get()
                            if epoch==constant.EPOCHS-1:
                                torch.save(intermediate,      path6+str(epoch)+'_Client'+str(idx)+'.pt')
                                # torch.save(images,            path7+str(epoch)+'_Client'+str(idx)+'.pt')
                                # images.get()
                                torch.save(labels,            path8+str(epoch)+'_Client'+str(idx)+'.pt')
        ##############################################################
                            output = output.get()
                            criterion = nn.CrossEntropyLoss()
                            loss = criterion(output, labels)
                            target_tot, pred_tot = make_prediction(output, labels.tolist(), target_tot, pred_tot)

                            temp = loss.item()
                            running_loss_val += temp
                            del intermediate
                            del images
                            del labels
                            sy.local_worker.clear_objects()
                            j = j+1
                            if j>=expected_batches:
                                logger.debug('Batches: {}, Expected_Batches: {}, constant.CLIENTS: {}, constant.MAXCLIENTS: {}, Total_Batches: {}'.format(j, expected_batches, constant.CLIENTS, constant.MAXCLIENTS, len(dataloaders['val'])))
                                break
                    else:
                        target_tot_ = sum(target_tot, [])
                        pred_tot_ = sum(pred_tot, [])
                        cm1 = confusion_matrix(target_tot_, pred_tot_)
                        if data == 'mnist':
                            preds = torch.FloatTensor(pred_tot_)
                            targets = torch.FloatTensor(target_tot_)
                            acc = preds.eq(targets).float().mean()
                            logger.debug(cm1)
                            epoch_loss = running_loss_val / len(dataloaders['val'].dataset)
                            logger.debug('Phase: {} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format('val', epoch, epoch_loss, acc))
                        if data == 'covid':
                            f1_score_value = f1_score(pred_tot_, target_tot_)
                            # print(cm1)
                            logger.debug(cm1)
                            epoch_loss = running_loss_val / len(dataloaders['val'].dataset)
                            logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', epoch, epoch_loss, f1_score_value))

                        if(torch.cuda.is_available()== True):
                            logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
                        
        epoch=epoch+1