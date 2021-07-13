#######################################################################################################################
# Parallel Split Learning: vary partition size and fix dataset
# note: change constant.CLIENTS to control partition size, i.e., partition the fixed dataset to how many parts
#######################################################################################################################
import logging
import torch
from time import time

from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

from torch import nn
import syft as sy
import time

from FSL_algorithm.resources.lenet import get_modelMNIST

from FSL_algorithm.resources.classes import SingleSplitNN
from FSL_algorithm.resources.setup import setup1
from FSL_algorithm.resources.functions import make_prediction, total_time_train
from sklearn.metrics import confusion_matrix, f1_score

from opacus import PrivacyEngine 

from pathlib import Path
import os
import math


def setup_logger(name, log_file):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


def train(x, target, splitNN, batch_size, batch_idx):
    splitNN.zero_grads()

    begin_clients = time.time()
    intermediate_list = splitNN.forwardA(x)
    forward_clients = time.time()-begin_clients

    begin_server = time.time()
    loss, inputsB, pred = splitNN.for_back_B(target, x)
    end_server = time.time()-begin_server

    begin2_clients = time.time()
    splitNN.backwardA(batch_size, inputsB)
    splitNN.step(batch_idx)
    backward_clients = time.time() - begin2_clients

    return loss, pred, forward_clients+backward_clients, end_server, intermediate_list

def run_model(device, dataloaders, data, constant):
    if(torch.cuda.is_available()== True):
        torch.cuda.reset_max_memory_allocated()
    wd = os.path.join(constant.PD, 'm1_dp_'+str(constant.PARAM)+'_reconstruction_vary_partition_size_fix_dataset_'+str(constant.CLIENTS)+"_base_"+str(constant.MAXCLIENTS))+"_dist"
    Path(wd).mkdir(parents=True, exist_ok=True)

    logs_dirpath = wd+'/logs/train/'
    Path(logs_dirpath).mkdir(parents=True, exist_ok=True)

    log_filename_idx = 1
    while os.path.isfile(logs_dirpath+str(log_filename_idx)+'.log'):
        log_filename_idx = log_filename_idx+1
    logger = setup_logger(str(log_filename_idx), logs_dirpath+str(log_filename_idx)+'.log')

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


    #Split Original Model
    if (data == 'mnist'):
        model = get_modelMNIST(10)  
    
    modelsA, modelsB = setup1(model, device, constant)
    num_of_batches = len(dataloaders["train"])

    #Optimizer
    optimizersA = [
        torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        if constant.OPTIMIZER == 'Adam'
        else
        torch.optim.SGD(model.parameters(), lr=0.03,)
        for model in modelsA 
    ]
    # privacy_engine_A = []
    # for idx, model in enumerate(modelsA):
    #     privacy_engine = PrivacyEngine(model,
    #                                 batch_size=constant.BATCH_SIZE, 
    #                                 sample_size=len(dataloaders['train'].dataset), 
    #                                 alphas=range(2,32), 
    #                                 noise_constant.PARAM=constant.PARAM,
    #                                 max_grad_norm=1.0)
    #     privacy_engine.attach(optimizersA[idx])
    #     privacy_engine.to(device)
    #     privacy_engine_A.append(privacy_engine)
    
    optimizersB = [
         torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        if constant.OPTIMIZER == 'Adam'
        else
        torch.optim.SGD(model.parameters(), lr=0.03,)
        for model in modelsB 
    ]
    # privacy_engine_B = []
    # for idx, model in enumerate(modelsB):
    #     privacy_engine = PrivacyEngine(model,
    #                                 batch_size=constant.BATCH_SIZE, 
    #                                 sample_size=len(dataloaders['train'].dataset), 
    #                                 alphas=range(2,32), 
    #                                 noise_constant.PARAM=constant.PARAM,
    #                                 max_grad_norm=1.0)
    #     privacy_engine.attach(optimizersB[idx])
    #     privacy_engine.to(device)
    #     privacy_engine_B.append(privacy_engine)


    #Distributed Workers
    # addr_array = ["http://10.52.3.229:", "http://10.52.1.120:", "http://10.52.3.22:"]
    addr_array = ["http://127.0.0.1:", "http://127.0.0.1:", "http://127.0.0.1:"]
    port_array = [7600, 7601, 7602]
    alice_array = []
    client_array = []
    for k in range(1):
        for i in range(constant.CLIENTS):
            # sy.VirtualWorker(hook, id="client{}{}".format(i+1, k))
            remote_client = DataCentricFLClient(hook, addr_array[i] + str(port_array[i]))
            client_array.append(remote_client)
            alice_array.append(remote_client)
            # alice_array.append("client{}{}".format(i+1, k))


    bob_array = []
    # remote_client = sy.VirtualWorker(hook, id="bob")
    remote_client = DataCentricFLClient(hook, addr_array[2] + str(port_array[2]))
    client_array.append(remote_client)
    bob_array = [remote_client]
    # alice_array.append("bob")

    my_grid = sy.PrivateGridNetwork(*client_array)

    #Create splitNN model
    splitNN = SingleSplitNN(modelsA, modelsB, optimizersA, optimizersB, num_of_batches)
    

    #Send Split Model to location
    for idx, (model, location) in enumerate(zip(modelsA, alice_array)):
        model = model.send(location)
        privacy_engine = PrivacyEngine(model,
                                    batch_size=constant.BATCH_SIZE, 
                                    sample_size=len(dataloaders['train'].dataset), 
                                    alphas=range(2,32), 
                                    noise_multiplier=constant.PARAM,
                                    max_grad_norm=1.0)
        privacy_engine.attach(optimizersA[idx])
        

    
    
    for idx, (model, location) in enumerate(zip(modelsB, bob_array)):
        model = model.send(location)
        privacy_engine = PrivacyEngine(model,
                                    batch_size=constant.BATCH_SIZE*constant.CLIENTS, 
                                    sample_size=len(dataloaders['train'].dataset), 
                                    alphas=range(2,32), 
                                    noise_multiplier=constant.PARAM,
                                    max_grad_norm=1.0)
        privacy_engine.attach(optimizersB[idx])
        
    
    counter=0
    best_f1_score=0
    #f1_history = {'train': [], 'val': []}
    #loss_history = {'train': [], 'val': []}
    epoch = 0
    
    while(epoch < constant.EPOCHS):
        
        logger.debug('Epoch {}/{}'.format(epoch, constant.EPOCHS - 1))
        logger.debug('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                since = time.time()
                splitNN.train()  # Set model to training mode
            else:
                splitNN.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            j=0
            k=0
            images_array=[]
            labels_array=[]
            labels_temp = []
            target_tot = []
            pred_tot = []
            all_output = []
            all_labels = []
            total_time_client = 0
            total_time_server = 0

            expected_batches = constant.CLIENTS*len(dataloaders[phase])/constant.CLIENTS
            batch_idx = 0
            for filling_data in range(math.ceil(constant.CLIENTS/constant.CLIENTS)):
                for images, labels in dataloaders[phase]:
                    batch_idx = batch_idx+1
                    images = images.to(device)
                    labels = labels.to(device)
                    if phase == 'train':
                        images = images.send(modelsA[j%constant.CLIENTS].location)
                        #ONLY for MNIST
                        images_array.append(images)
                        labels_temp.append(labels.tolist())
                        labels_array.append(labels.send(modelsB[-1].location))
                        j = j+1
                    if phase == 'val':
                        images = images.send(modelsA[k%constant.CLIENTS].location)
                        labels.send(modelsB[-1].location)
                    
                    with torch.set_grad_enabled(phase == 'train' or phase == 'val'):
                        if (j%constant.CLIENTS == 0 and phase == 'train'):
                            loss, output, end_clients, end_server, intermediate_list = train(images_array, labels_array, splitNN, constant.BATCH_SIZE, batch_idx)
                            temp = loss.get()
                            running_loss += float(temp)
                            labels_temp = sum(labels_temp,[])
                            output = output.get()
                            target_tot, pred_tot =  make_prediction(output, labels_temp, target_tot, pred_tot)

                            total_time_client += end_clients
                            total_time_server += end_server



    ##############################################################
                            for idx, (intermediate, images, labels) in enumerate(zip(intermediate_list, images_array, labels_array)):
                                intermediate = intermediate.get()
                                images = images.get()
                                if epoch==constant.EPOCHS-1:
                                    torch.save(intermediate,   path1+str(epoch)+"_"+str(batch_idx)+'_Client'+str(idx)+'.pt')
                                    torch.save(labels,         path3+str(epoch)+"_"+str(batch_idx)+'_Client'+str(idx)+'.pt')
                                del intermediate
                                del images
                                del labels
    ##############################################################
                            sy.local_worker.clear_objects()
                            del images_array
                            del labels_array
                            del labels_temp
                            images_array=[]
                            labels_array=[]
                            labels_temp =[]
                            if j>=expected_batches:
                                logger.debug('Batches: {}, Expected_Batches: {}, constant.CLIENTS: {}, constant.MAXCLIENTS: {}, Total_Batches: {}'.format(j, expected_batches, constant.CLIENTS, constant.MAXCLIENTS, len(dataloaders['train'])))
                                break
                        if phase == 'val':
                            output, intermediate = splitNN.forward(images, k%constant.CLIENTS)

    ##############################################################
                            intermediate = intermediate.get()
                            images = images.get()
                            if epoch==constant.EPOCHS-1:
                                torch.save(intermediate,   path6+str(epoch)+"_"+str(batch_idx)+'.pt')
                                torch.save(labels,         path8+str(epoch)+"_"+str(batch_idx)+'.pt')
                                
    ##############################################################
                            output = output.get()
                            criterion = nn.CrossEntropyLoss()
                            loss = criterion(output, labels)
                            target_tot, pred_tot =  make_prediction(output, labels.tolist(),  target_tot, pred_tot)
                        
                            del intermediate
                            del images
                            del labels
                        
                            running_loss += float(loss.item())
                            sy.local_worker.clear_objects()
                            k = k+1
                            if k%constant.CLIENTS==0 and k>=expected_batches:
                                logger.debug('Batches: {}, Expected_Batches: {}, constant.CLIENTS: {}, constant.MAXCLIENTS: {}, Total_Batches: {}'.format(j, expected_batches, constant.CLIENTS, constant.MAXCLIENTS, len(dataloaders['val'])))
                                break
                            
            else:
                if phase == 'train':
                    # time_elapsed = time.time() - since
                    # # print('Epoch {} in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))
                    # logger.debug('Epoch {} in {}'.format(epoch, time_elapsed))
                    total_time_train(since, epoch, total_time_client, total_time_server, logger, phase)

                #loss_history[phase].append(running_loss)
                target_tot_ = sum(target_tot, [])
                pred_tot_ = sum(pred_tot, [])
                cm1 = confusion_matrix(target_tot_, pred_tot_)
                if data == 'mnist' or data == 'cifar10':
                    preds = torch.FloatTensor(pred_tot_)
                    targets = torch.FloatTensor(target_tot_)
                    acc = preds.eq(targets).float().mean()
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    # print('Phase: {} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format(phase, epoch, epoch_loss, acc))
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format(phase, epoch, epoch_loss, acc))
                    # file1.write('{} {} {:.4f} {:.4f}\n'.format(phase, epoch, epoch_loss, acc))
                if data =='covid':
                    f1_score_value = f1_score(pred_tot_, target_tot_)
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    # # print('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format(phase, epoch, epoch_loss, f1_score_value))
                    # file1.write('{} {} {:.4f} {:.4f}\n'.format(phase, epoch, epoch_loss, f1_score_value))
                    # print('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format(phase, epoch, epoch_loss, f1_score_value))
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format(phase, epoch, epoch_loss, f1_score_value))
                    # file1.write('{} {} {:.4f} {:.4f}\n'.format(phase, epoch, epoch_loss, acc))
                # print(cm1)
                logger.debug(cm1)
                
                if(torch.cuda.is_available()== True):
                     # print('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                            # sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
                     logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                            sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
                #if data == covid -> early stopping
                if data == 'covid':
                    if(phase == 'val'):   
                        if (counter < constant.MAX_COUNTER):
                            if(epoch == (constant.EPOCHS-1)):
                                constant.EPOCHS= constant.EPOCHS+50
                            if (best_f1_score < f1_score_value):  
                                best_f1_score = f1_score_value
                                counter = 0
                                cmBest = cm1
                            else: 
                                counter = counter+1
                        # else:
                        #     file2.write('{} {}\n {}\n'.format(epoch, best_f1_score, cmBest))
                        #     return 
        
        
        
        epoch=epoch+1
