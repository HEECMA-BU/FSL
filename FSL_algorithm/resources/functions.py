import logging
import torch
import random
import numpy as np
import os
import sys
import time

def set_seed(seed):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

def write_file(vector, name_file):
    f = open("{}.txt".format(name_file), "a")
    for i in range(len(vector)):
      f.write("EPOCH {} {}\n".format(i, vector[i]))
    f.close()

def make_prediction(output, target, target_array, pred_array):
    pred = output.data.max(1, keepdim=True)[1]
    pred = pred.reshape(len(target)).tolist()
    target_array.append(target)
    pred_array.append(pred)
    
    return target_array, pred_array
    
def total_time_train(since, epoch, total_time_client, total_time_client_trainA, total_time_server, averaging_time, total_steptime_client, total_steptime_client_trainA, total_steptime_server, logger, phase):
    time_elapsed = time.time() - since
    print('Epoch {} in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))
    logger.debug(': {},{},{},{},{},{},{},{},{},{}'.format(phase, epoch, time_elapsed, total_time_client, total_time_client_trainA, total_time_server, averaging_time, total_steptime_client, total_steptime_client_trainA, total_steptime_server))
    # file3.write('{} {:.0f}m {:.0f}s {:.0f}m {:.0f}s {:.0f}m {:.0f}s\n'.format(epoch, time_elapsed // 60, time_elapsed % 60, total_time_client // 60, total_time_client % 60, total_time_server // 60, total_time_server % 60))

