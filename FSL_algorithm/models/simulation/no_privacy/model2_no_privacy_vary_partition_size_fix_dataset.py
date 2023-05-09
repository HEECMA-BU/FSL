#######################################################################################################################
# Federated Split Learning: vary partition size and fix dataset
# note: change constant.CLIENTS to control partition size, i.e., partition the fixed dataset to how many parts
#######################################################################################################################
import logging
import math
import torch
from time import time

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

import syft as sy
import time
from sklearn.metrics import confusion_matrix, f1_score
from FSL_algorithm.resources.lenet import get_modelMNIST
from FSL_algorithm.resources.lenet import get_modelCIFAR10
from FSL_algorithm.resources.VPN import get_modelVPN
from FSL_algorithm.resources.vgg import get_modelCIFAR, remove_dropout, add_batchnorm, add_small_filter

from FSL_algorithm.resources.setup import setup2, average_weights
from FSL_algorithm.resources.functions import make_prediction, total_time_train

from FSL_algorithm.resources.classes import MultiSplitNN

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
    step_times = model.step()    
    return loss, pred, forward_clients, backward_client, forward_server, backward_server, step_times, intermediate


def run_model(device, dataloaders, data, constant):
    if(torch.cuda.is_available()== True):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()
    wd = os.path.join(constant.PD, 'm2_nop_reconstruction_client_'+str(constant.CLIENTS)+"_vary_partition_size_fix_dataset_base_"+str(constant.MAXCLIENTS)+"_"+str(constant.CUTS[1])+"_"+str(data)+"_"+constant.DATA_DIST+"_SerAvg")
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
    
    #Set up Model
    if (data == 'mnist'):
        model_all = get_modelMNIST(10)
    if (data == 'VPN'):
        model_all = get_modelVPN(6)

    if (data == 'cifar10'):
        model_all = get_modelCIFAR(10, device)
        # model_all = add_small_filter(model_all, [2, 4, 7])
        # model_all = add_small_filter(model_all, [2])
        # model_all = add_batchnorm(model_all)
    # if (data == 'covid'):
    #     model_all = get_modelCOVID()

    #Split Original Model
    local_models, models = setup2(model_all, device, constant)

    #Optimizer
    optimizers = {}
    scheduler = {}
    for i in range(constant.CLIENTS):
        optimizers['optimizer{}'.format(i+1)] =  [ 
            torch.optim.AdamW(model.parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
            if constant.OPTIMIZER == 'Adam'
            else
            torch.optim.SGD(model.parameters(), lr=0.03,)
            for model in models['models{}'.format(i+1)]]
        scheduler['optimizer{}'.format(i+1)] = [ExponentialLR(opt_init, gamma=0.9) for opt_init in optimizers['optimizer{}'.format(i+1)]]

    # Workers
    # client_array = []
    client_array = [[] for i in range(constant.CLIENTS)]
    # clientname_array = []
    for k in range(constant.CLIENTS):
        for i in range(constant.THOR):
            remote_client = sy.VirtualWorker(hook, id="client{}{}".format(k+1, i))
            client_array[k].append(remote_client)

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
   
    while(epoch < constant.EPOCHS):
        logger.debug('Epoch {}/{}'.format(epoch, constant.EPOCHS - 1))
        logger.debug('-' * 10)

        #Training
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

        for i in range(constant.CLIENTS):
            running_loss[i]=0
            lun[i]=0

        if (epoch == 0):
            for k in range(constant.CLIENTS):
                splitNNs['splitNN{}'.format(k+1)].train()
                for i in range(constant.THOR):
                    # models['models{}'.format(k+1)][i].send('client{}{}'.format(k+1, i))
                    models['models{}'.format(k+1)][i].send(client_array[k][i])
                    optimizer1 = optim.AdamW(models['models{}'.format(k+1)][i].parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                    scheduler['optimizer{}'.format(k+1)][i] = ExponentialLR(optimizer1, gamma=0.9)
                    optimizers['optimizer{}'.format(k+1)][i] = optimizer1
        for k in range(constant.CLIENTS):
            for i in range(constant.THOR):
                models["models{}".format(k+1)][i].train()
                
        j=0
        # CLIENTS_range = CLIENTS//len(dataloaders['train'])
        loader_cli_map = {}
        for loader_idx in range(len(dataloaders['train'])):
            loader_cli_map[loader_idx] = []
        for cli_idx, loader_idx in zip(range(constant.CLIENTS), cycle(range(len(dataloaders['train'])))):
            loader_cli_map[loader_idx].append(cli_idx)

        cli_pred_map = defaultdict(list)
        cli_target_map = defaultdict(list)
        
        since = time.time()
        for dat_idx in range(len(dataloaders['train'])):
            for idx, (images, labels) in enumerate(dataloaders['train'][dat_idx]):
                images = images.to(device)
                labels = labels.to(device)
                images = images.send(models['models{}'.format(loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]+1)][0].location)
                labels = labels.send(models['models{}'.format(loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]+1)][constant.THOR-1].location)


                loss, output, end_clients_forw, end_clients_back, end_server_forw, end_server_back, step_times, intermediate = train(images, labels, splitNNs['splitNN{}'.format(loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]+1)])
                temp = loss.get()
                running_loss[idx%constant.CLIENTS] += float(temp)
                labels = labels.get()
                output = output.get()

                # pred, target_tot, pred_tot =  make_prediction(output, labels.tolist(), target_tot, pred_tot)
                target = labels.tolist()
                pred = output.data.max(1, keepdim=True)[1]
                pred = pred.reshape(len(target)).tolist()
                target_tot.append(target)
                pred_tot.append(pred)
                cli_target_map[loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]+1].append(target)
                cli_pred_map[loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]+1].append(pred)
                
                total_steptime_server += step_times[1]  # have just trained the corresponding server part in FSL
                total_time_server += end_server_forw + end_server_back
                total_time_server_forw += end_server_forw
                total_time_server_back += end_server_back
                
                total_steptime_client += step_times[0]
                total_time_client += end_clients_forw + end_clients_back
                total_time_client_forw += end_clients_forw
                total_time_client_back += end_clients_back


    ##############################################################
                save_time = time.time()
                intermediate = intermediate.get()
                images = images.get()
                # if idx <= 100:
                #     torch.save(intermediate,   path1+str(epoch)+'_Client_'+str(dat_idx)+"_"+str(idx)+'.pt')
                #     torch.save(labels,         path3+str(epoch)+'_Client_'+str(dat_idx)+"_"+str(idx)+'.pt')
                save_time = time.time() - save_time
                since += save_time
    ##############################################################
                lun[loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]]=lun[loader_cli_map[dat_idx][idx%len(loader_cli_map[dat_idx])]]+len(images)

                del intermediate
                del images
                # labels = labels.get()
                del labels
                sy.local_worker.clear_objects()

                j=j+1
                # break
        else:
            # for k in range(constant.CLIENTS):
            #     for i in range(constant.THOR):
            #         scheduler['optimizer{}'.format(k+1)][i].step()

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
                    logger.debug('Phase: {} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', cli_dix, epoch, running_loss[cli_dix%constant.CLIENTS]/lun[cli_dix%constant.CLIENTS], acc))
                if data == 'covid':
                    preds = torch.FloatTensor(pred_tot_)
                    targets = torch.FloatTensor(target_tot_)
                    f1_score_value = f1_score(pred_tot_, target_tot_)
                    # print(cm1)
                    # logger.debug(cm1)
                    logger.debug('Phase: {} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', cli_dix, epoch, running_loss[cli_dix%constant.CLIENTS]/lun[cli_dix%constant.CLIENTS], f1_score_value))

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
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss[i%constant.CLIENTS]/lun[i%constant.CLIENTS], acc))
            if data == 'covid':
                preds = torch.FloatTensor(pred_tot_)
                targets = torch.FloatTensor(target_tot_)
                f1_score_value = f1_score(pred_tot_, target_tot_)
                # print(cm1)
                # logger.debug(cm1)
                for i in range(constant.CLIENTS):
                    logger.debug('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('train', epoch, running_loss[i%constant.CLIENTS]/lun[i%constant.CLIENTS], f1_score_value))

            with torch.no_grad():
                #Average weights
                # averaging_time_client = time.time()
                # average_weights(models, local_models, False, CLIENTS, 0)
                # averaging_time_client = time.time() - averaging_time_client
                averaging_time_client = 0
                averaging_time_server = time.time()
                average_weights(models, local_models, False, constant, 1)
                averaging_time_server = time.time() - averaging_time_server

                #Total Time for one epoch
                total_time_train(since, epoch, total_time_client, total_time_client_forw, total_time_client_back, total_time_client_trainA, total_time_server, total_time_server_forw, total_time_server_back, averaging_time_client, averaging_time_server, total_steptime_client, total_steptime_client_trainA, total_steptime_server, logger, "train"+":"+str(j))
                
                #Validation
                for k in range(constant.CLIENTS):
                    for i in range(1, constant.THOR):   # only the server parts were averaged at PS.
                        models['models{}'.format(k+1)][i].send(client_array[k][i])
                        optimizer1 = optim.AdamW(models['models{}'.format(k+1)][i].parameters(), lr=constant.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                        scheduler['optimizer{}'.format(k+1)][i] = ExponentialLR(optimizer1, gamma=0.9)
                        optimizers['optimizer{}'.format(k+1)][i] = optimizer1   
            
                for k in range(constant.CLIENTS):
                    for i in range(constant.THOR):
                        models["models{}".format(k+1)][i].eval()
            
                target_tot=[]
                pred_tot=[]
                total_time_client = 0
                total_time_client_forw = 0
                total_time_client_back = 0
                total_time_server = 0
                total_time_server_forw = 0
                total_time_server_back = 0
                cli_pred_map = defaultdict(list)
                cli_target_map = defaultdict(list)
                
                running_loss_val = 0
                with torch.no_grad():
                    for dat_idx in range(len(dataloaders['val'])):
                        since = time.time()
                        for idx, (images, labels) in enumerate(dataloaders['val'][dat_idx]):
                            images = images.to(device)
                            labels = labels.to(device)
                            images = images.send(models['models{}'.format((idx%constant.CLIENTS)+1)][0].location)
                            # labels = labels.send(models['models{}'.format((idx%constant.CLIENTS)+1)][constant.THOR-1].location)
                            output, forward_clients, forward_server, intermediate = splitNNs['splitNN{}'.format((idx%constant.CLIENTS)+1)].forwardVal(images)
        ##############################################################
                            intermediate = intermediate.get()
                            images = images.get()
                            # if epoch==constant.EPOCHS-1:
                            if idx <= 100:
                                torch.save(intermediate,      path6+str(epoch)+'_Client_'+str(dat_idx)+"_"+str(idx)+'.pt')
                                # torch.save(images,            path7+str(epoch)+'_Client_'+str(dat_idx)+"_"+str(idx)+'.pt')
                                # images.get()
                                torch.save(labels,            path8+str(epoch)+'_Client_'+str(dat_idx)+"_"+str(idx)+'.pt')
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
                            running_loss_val += temp
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
                                    logger.debug('Phase: {}{} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, cli_dix, epoch, running_loss[cli_dix%constant.CLIENTS]/lun[cli_dix%constant.CLIENTS], acc))
                                if data == 'covid':
                                    preds = torch.FloatTensor(pred_tot_)
                                    targets = torch.FloatTensor(target_tot_)
                                    f1_score_value = f1_score(pred_tot_, target_tot_)
                                    # print(cm1)
                                    # logger.debug(cm1)
                                    logger.debug('Phase: {}{} Cli:{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, cli_dix, epoch, running_loss[cli_dix%constant.CLIENTS]/lun[cli_dix%constant.CLIENTS], f1_score_value))

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
                                epoch_loss = running_loss_val / len(dataloaders['val'][dat_idx].dataset)
                                # print('Phase: {} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format('val', epoch, epoch_loss, acc))
                                logger.debug('Phase: {}{} Epoch: {} Loss: {:.4f} Accuracy {:.4f}'.format('val', dat_idx, epoch, epoch_loss, acc))
                                # file1.write('{} {} {:.4f} {:.4f}\n'.format('val', epoch, epoch_loss, acc))
                            if data == 'covid':
                                f1_score_value = f1_score(pred_tot_, target_tot_)
                                # print(cm1)
                                logger.debug(cm1)
                                epoch_loss = running_loss_val / len(dataloaders['val'][dat_idx].dataset)
                                # print('Phase: {} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', epoch, epoch_loss, f1_score_value))
                                logger.debug('Phase: {}{} Epoch: {} Loss: {:.4f} F1_Score {:.4f}'.format('val', dat_idx, epoch, epoch_loss, f1_score_value))
                                # file1.write('{} {} {:.4f} {:.4f}\n'.format('val', epoch, epoch_loss, f1_score_value))

                            if(torch.cuda.is_available()== True):
                                # print('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                    #  sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
                                logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                    sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))

        epoch=epoch+1