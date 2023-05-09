#######################################################################################################################
# Parallel Split Learning: vary partition size and fix dataset
# note: change constant.CLIENTS to control partition size, i.e., partition the fixed dataset to how many parts
#######################################################################################################################
import logging
import math
import torch
from time import time
from torch.optim.lr_scheduler import ExponentialLR

from torch import nn, optim
import syft as sy
import time
from sklearn.metrics import confusion_matrix, f1_score
from FSL_algorithm.resources.lenet import get_modelMNIST
from FSL_algorithm.resources.lenet import get_modelCIFAR10
from FSL_algorithm.resources.VPN import get_modelVPN
from FSL_algorithm.resources.vgg import get_modelCIFAR

from FSL_algorithm.resources.setup import setup1
from FSL_algorithm.resources.functions import make_prediction, total_time_train

from FSL_algorithm.resources.classes import SingleSplitNN

from pathlib import Path
import os
from itertools import cycle
from collections import defaultdict

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


def train(x, target, splitNN, batch_size, ready_clis_l):
    splitNN.zero_grads()

    forward_clients = time.time()
    intermediate_list = splitNN.forwardA(x, ready_clis_l)
    forward_clients = time.time()-forward_clients

    loss, inputsB, pred, B_forward_time, B_backward_time = splitNN.for_back_B(target, x, ready_clis_l)

    backward_clients = splitNN.backwardA(batch_size, inputsB, ready_clis_l)

    client_step_time, server_step_time = splitNN.step()

    return loss, pred, forward_clients, backward_clients, B_forward_time, B_backward_time, client_step_time, server_step_time, intermediate_list

def run_model(device, dataloaders, data, constant):
    if(torch.cuda.is_available()== True):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
    wd = os.path.join(constant.PD, 'm1_nop_reconstruction_client_'+str(constant.CLIENTS)+"_vary_partition_size_fix_dataset_base_"+str(constant.MAXCLIENTS)+"_"+str(constant.CUTS[1])+"_"+str(data)+"_"+constant.DATA_DIST)
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
    # path4 = wd+'/intermediate/Train/'
    # Path(path4).mkdir(parents=True, exist_ok=True)
    # path5 = wd+'/source/Train/'
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
        model_all = get_modelMNIST(10)
    if (data == 'VPN'):
        model_all = get_modelVPN(6)

    if (data == 'cifar10'):
        model_all = get_modelCIFAR(10)
    # if (data == 'covid'):
    #     model_all = get_modelCOVID() 
    
    #Split Original Model
    modelsA, modelsB = setup1(model_all, device, constant)

    #Optimizer
    optimizersA = [
        torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        if constant.OPTIMIZER == 'Adam'
        else
        torch.optim.SGD(model.parameters(), lr=0.03,)
        for model in modelsA 
    ]
    
    optimizersB = [
         torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        if constant.OPTIMIZER == 'Adam'
        else
        torch.optim.SGD(model.parameters(), lr=0.03,)
        for model in modelsB 
    ]


    # Workers
    alice_array = []
    client_array = []
    for k in range(constant.CLIENTS):
        for i in range(1):
            remote_client = sy.VirtualWorker(hook, id="client{}{}".format(k+1, i))
            client_array.append(remote_client)
            alice_array.append(remote_client)


    bob_array = []
    remote_client = sy.VirtualWorker(hook, id="bob")
    client_array.append(remote_client)
    bob_array = [remote_client]

    #Create splitNN model
    splitNN = SingleSplitNN(modelsA, modelsB, optimizersA, optimizersB)

    #Send Split Model to location
    for model, location in zip(modelsA, alice_array):
        model.send(location)

    
    for model, location in zip(modelsB, bob_array):
        model.send(location)
    
    counter = 0
    epoch=0
    best_f1_score=0

    while(epoch < constant.EPOCHS):
        logger.debug('Epoch {}/{}'.format(epoch, constant.EPOCHS - 1))
        logger.debug('-' * 10)

        #Training
        images_array=[]
        labels_array=[]
        labels_temp = []
        target_tot = []
        pred_tot = []
        total_time_client = 0
        total_time_client_forw = 0
        total_time_client_back = 0
        total_time_server = 0
        total_time_server_forw = 0
        total_time_server_back = 0
        total_steptime_client = 0
        total_steptime_server = 0
        total_steptime_client_trainA = 0
        total_time_client_trainA = 0
        
        running_loss = 0.0

        splitNN.train()  # Set model to training mode

        batch_count = 0
        # CLIENTS_range = CLIENTS//len(dataloaders['train'])
        loader_cli_map = {}
        for loader_idx in range(len(dataloaders['train'])):
            loader_cli_map[loader_idx] = []
        for cli_idx, loader_idx in zip(range(constant.CLIENTS), cycle(range(len(dataloaders['train'])))):
            loader_cli_map[loader_idx].append(cli_idx)

        cli_pred_map = defaultdict(list)
        cli_target_map = defaultdict(list)

        total_batches = 0
        batches_by_loader = []
        itor_l = []
        for loader in dataloaders['train']:
            total_batches += len(loader) 
            batches_by_loader.append(0)
            itor_l.append(iter(loader))
        
        since = time.time()
        while batch_count < total_batches:
        # for idx, ((images, labels), (images, labels)) in enumerate(zip(*dataloaders['train'])):
        # for dat_idx in range(len(dataloaders['train'])):
        #     for idx, (images, labels) in enumerate(dataloaders['train'][dat_idx]):
            ready_clis_l = []
            for dat_idx, cli_l in loader_cli_map.items():   # pick a dataloader among the clients to send data
                for cli_idx in cli_l:          # pick a client under an assigned dataloader to send data
                    # if len(dataloaders['train'][dat_idx]) > batches_by_loader[dat_idx]:
                    try:
                        # (images, labels) = dataloaders['train'][dat_idx][batches_by_loader[dat_idx]]
                        (images, labels) = next(itor_l[dat_idx])
                        batches_by_loader[dat_idx] += 1
                        batch_count = batch_count+1
                        images = images.to(device)
                        labels = labels.to(device)
                        # images = images.send(modelsA[loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]].location)
                        images = images.send(modelsA[cli_idx].location)

                        images_array.append(images)
                        # labels_temp.append(labels.tolist())
                        labels_array.append(labels.send(modelsB[-1].location))
                        # labels_array.append(labels)
                        ready_clis_l.append(cli_idx)
                        # j = j+1
                    # else:
                    #     break
                    except StopIteration:
                        logger.debug("batches_by_loader:", batches_by_loader)
                        # break
            if len(ready_clis_l) < constant.CLIENTS:
                logger.debug("ready_clis_l:", ready_clis_l)
            if ready_clis_l == []:
                break
            with torch.set_grad_enabled(True):
                # if (idx%CLIENTS == 0):

              # loss, pred,   forward_clients,  backward_clients, B_forward_time,  B_backward_time, client_step_time, server_step_time, intermediate_list
                loss, output, end_clients_forw, end_clients_back, end_server_forw, end_server_back, client_step_time, server_step_time, intermediate_list = train(images_array, labels_array, splitNN, constant.BATCH_SIZE, ready_clis_l)
                temp = loss.get()
                running_loss += float(temp)
                # labels_temp = sum(labels_temp,[])
                # labels = labels.get()
                output = output.get()

                # for idx, (labels) in enumerate(labels_array):
                for idx, cli_idx in enumerate(ready_clis_l):
                    # pred, target_tot, pred_tot =  make_prediction(output, labels.tolist(), target_tot, pred_tot)
                    # target = labels.tolist()
                    target = labels_array[idx].tolist()
                    pred = output[idx*constant.BATCH_SIZE:(idx+1)*constant.BATCH_SIZE].data.max(1, keepdim=True)[1]
                    pred = pred.reshape(len(target)).tolist()
                    target_tot.append(target)
                    pred_tot.append(pred)
                    cli_target_map[cli_idx+1].append(target)
                    cli_pred_map[cli_idx+1].append(pred)

                total_steptime_server += server_step_time
                total_time_server += end_server_forw + end_server_back
                total_time_server_forw += end_server_forw
                total_time_server_back += end_server_back

                total_steptime_client += client_step_time
                total_time_client += end_clients_forw + end_clients_back
                total_time_client_forw += end_clients_forw
                total_time_client_back += end_clients_back
                
                
##############################################################
                # save_time = time.time()
                # intermediate_list = splitNN.forwardA_privacy_test(images_array)
                for idx, (intermediate, images, labels) in enumerate(zip(intermediate_list, images_array, labels_array)):
                    intermediate = intermediate.get()
                    images = images.get()
                    # labels = labels.get()
                    # if epoch==19:
                    if batch_count <= 101:    # consistent to other models to have 101 intermediate data.
                        torch.save(intermediate,   path1+str(epoch)+"_"+str(batch_count)+'_Client'+str(idx)+'.pt')
                        # torch.save(images.copy().get(),         path2+str(epoch)+"_"+str(batch_count)+'_Client'+str(idx)+'.pt')
                        torch.save(labels,         path3+str(epoch)+"_"+str(batch_count)+'_Client'+str(idx)+'.pt')
                    # since += save_time
                    del intermediate
                    del images
                    del labels
                # save_time = time.time() - save_time
                # since += save_time
##############################################################

                sy.local_worker.clear_objects()
                del images_array
                del labels_array
                # del labels_temp
                images_array=[]
                labels_array=[]
                # labels_temp =[]

                        
            # j = j+1
            # batch_count = batch_count+1
                        
        # else:
        # summary for each client
        for cli_dix in range(constant.CLIENTS):
            target_tot_ = sum(cli_pred_map[cli_dix+1], [])
            pred_tot_ = sum(cli_target_map[cli_dix+1], [])
            # pred_tot_ = cli_target_map[cli_dix+1]
            # cm1 = confusion_matrix(target_tot_, pred_tot_)
            if data == 'mnist' or data == 'cifar10' or data == 'VPN':
                preds = torch.FloatTensor(pred_tot_)
                targets = torch.FloatTensor(target_tot_)
                acc = preds.eq(targets).float().mean() 
                # print(cm1)
                # logger.debug(cm1)
                logger.debug('Phase: {} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', cli_dix, epoch, running_loss, acc))
            if data == 'covid':
                preds = torch.FloatTensor(pred_tot_)
                targets = torch.FloatTensor(target_tot_)
                f1_score_value = f1_score(pred_tot_, target_tot_)
                # print(cm1)
                # logger.debug(cm1)
                logger.debug('Phase: {} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', cli_dix, epoch, running_loss, f1_score_value))

        # summary over all clients
        target_tot_ = sum(target_tot, [])
        pred_tot_ = sum(pred_tot, [])
        # cm1 = confusion_matrix(target_tot_, pred_tot_)
        if data == 'mnist' or data == 'cifar10' or data == 'VPN':
            preds = torch.FloatTensor(pred_tot_)
            targets = torch.FloatTensor(target_tot_)
            acc = preds.eq(targets).float().mean() 
            # print(cm1)
            # logger.debug(cm1)
            for i in range(constant.CLIENTS):
                logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss, acc))
        if data == 'covid':
            preds = torch.FloatTensor(pred_tot_)
            targets = torch.FloatTensor(target_tot_)
            f1_score_value = f1_score(pred_tot_, target_tot_)
            # print(cm1)
            # logger.debug(cm1)
            for i in range(constant.CLIENTS):
                logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss, f1_score_value))

        #Total Time for one epoch
        total_time_train(since, epoch, total_time_client, total_time_client_forw, total_time_client_back, total_time_client_trainA, total_time_server, total_time_server_forw, total_time_server_back, 0, 0, total_steptime_client, total_steptime_client_trainA, total_steptime_server, logger, "train"+":"+str(batch_count))
        # total_time_train(since, epoch, total_time_client, total_time_client_trainA, total_time_server, 0, total_steptime_client, total_steptime_client_trainA, total_steptime_server, logger, "train")
        
        #Validation
        splitNN.eval()   # Set model to evaluate mode
            
        
        # j=0
        # k=0
        images_array=[]
        labels_array=[]
        labels_temp = []

        target_tot = []
        pred_tot = []
        total_time_client = 0
        total_time_client_forw = 0
        total_time_client_back = 0
        total_time_server = 0
        total_time_server_forw = 0
        total_time_server_back = 0


        cli_pred_map = defaultdict(list)
        cli_target_map = defaultdict(list)

        running_loss = [0]*len(dataloaders['val'])
        for dat_idx in range(len(dataloaders['val'])):
            since = time.time()
            for idx, (images, labels) in enumerate(dataloaders['val'][dat_idx]):
                images = images.to(device)
                labels = labels.to(device)
                images = images.send(modelsA[idx%constant.CLIENTS].location)
                with torch.set_grad_enabled(False):
                    output, forward_clients, forward_server, intermediate = splitNN.forward(images, idx%constant.CLIENTS)
##############################################################
                    intermediate = intermediate.get()
                    images = images.get()
                    # if epoch==constant.EPOCHS-1:
                    if batch_count <= 101:
                        torch.save(intermediate,   path6+str(epoch)+"_"+str(batch_count)+'.pt')


                        torch.save(labels,         path8+str(epoch)+"_"+str(batch_count)+'.pt')
##############################################################
                    output = output.get()
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output, labels)
                    # pred, target_tot, pred_tot = make_prediction(output, labels.tolist(), target_tot, pred_tot)
                    target = labels.tolist()
                    pred = output.data.max(1, keepdim=True)[1]
                    pred = pred.reshape(len(target)).tolist()
                    total_time_client_forw += forward_clients
                    total_time_server_forw += forward_server
                    target_tot.append(target)
                    pred_tot.append(pred)
                    cli_target_map[(idx%constant.CLIENTS)+1].append(target)
                    cli_pred_map[(idx%constant.CLIENTS)+1].append(pred)


                    temp = loss.item()
                    running_loss[dat_idx] += temp
                    del intermediate
                    del images
                    del labels
                    sy.local_worker.clear_objects()
                    # j = j+1
            else:
                #Total Time for one epoch
                total_time_train(since, epoch, 0, total_time_client_forw, 0, 0, 0, total_time_server_forw, 0, 0, 0, 0, 0, 0, logger, "val"+str(dat_idx)+":"+str(idx))

                # summary for each client
                for cli_dix in range(constant.CLIENTS):
                    target_tot_ = sum(cli_pred_map[cli_dix+1], [])
                    pred_tot_ = sum(cli_target_map[cli_dix+1], [])
                    # cm1 = confusion_matrix(target_tot_, pred_tot_)
                    if data == 'mnist' or data == 'cifar10' or data == 'VPN':
                        preds = torch.FloatTensor(pred_tot_)
                        targets = torch.FloatTensor(target_tot_)
                        acc = preds.eq(targets).float().mean() 
                        # print(cm1)
                        # logger.debug(cm1)
                        logger.debug('Phase: {}{} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, cli_dix, epoch, 0, acc))
                    if data == 'covid':
                        preds = torch.FloatTensor(pred_tot_)
                        targets = torch.FloatTensor(target_tot_)
                        f1_score_value = f1_score(pred_tot_, target_tot_)
                        # print(cm1)
                        # logger.debug(cm1)
                        logger.debug('Phase: {}{} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, cli_dix, epoch, 0, f1_score_value))

                # summary over all clients
                target_tot_ = sum(target_tot, [])
                pred_tot_ = sum(pred_tot, [])
                cm1 = confusion_matrix(target_tot_, pred_tot_)
                if data == 'mnist' or data == 'cifar10' or data == 'VPN':
                    preds = torch.FloatTensor(pred_tot_)
                    targets = torch.FloatTensor(target_tot_)
                    acc = preds.eq(targets).float().mean()
                    # print(cm1)
                    logger.debug(cm1)
                    epoch_loss = running_loss[dat_idx] / len(dataloaders['val'][dat_idx].dataset)
                    # print('Phase: {} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format('val', epoch, epoch_loss, acc))
                    logger.debug('Phase: {}{} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format('val', dat_idx, epoch, epoch_loss, acc))
                    # file1.write('{} {} {:.4f} {:.4f}\n'.format('val', epoch, epoch_loss, acc))
                if data == 'covid':
                    f1_score_value = f1_score(pred_tot_, target_tot_)
                    # print(cm1)
                    logger.debug(cm1)
                    epoch_loss = running_loss[dat_idx] / len(dataloaders['val'][dat_idx].dataset)
                    # print('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', epoch, epoch_loss, f1_score_value))
                    logger.debug('Phase: {}{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, epoch, epoch_loss, f1_score_value))
                    # file1.write('{} {} {:.4f} {:.4f}\n'.format('val', epoch, epoch_loss, f1_score_value))

                if(torch.cuda.is_available()== True):
                    # print('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                        #  sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
                    logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                        sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))

        epoch=epoch+1
