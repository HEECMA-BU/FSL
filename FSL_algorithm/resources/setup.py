# from FSL_algorithm.resources.config import Config as constant
from torch import nn

def setup1(model, device, constant):
    modelsA=[]
    modelsB=[]
    for i in range(constant.CLIENTS):
        modules = []
        for module in model[:constant.CUTS[1]]:
            modules.append(module)
        sequential = nn.Sequential(*modules)
        sequential1 = sequential.copy()
        modelsA.append(sequential1.to(device))
    
    modules = []
    for module in model[constant.CUTS[1]:]:
        modules.append(module)
    sequential = nn.Sequential(*modules)
    modelsB.append(sequential.to(device))
    
    return modelsA, modelsB

def setup2(model_all, device, constant):
    local_models={}
    for i in range(constant.THOR):
        modules = []
        sequential = []
        if i == constant.THOR-1:
            for module in model_all[constant.CUTS[i]:]:
                modules.append(module)
            sequential = nn.Sequential(*modules)
            local_models['model{}'.format(i)] = sequential.to(device)
        else:
            for module in model_all[constant.CUTS[i]:constant.CUTS[i+1]]:
                modules.append(module)
            sequential = nn.Sequential(*modules)
            local_models['model{}'.format(i)] = sequential.to(device)
    
    models = {}
    for k in range(constant.CLIENTS):
        sequentials = []
        for i in range(constant.THOR):
            modules = []
            if i == constant.THOR-1:
                for module in model_all[constant.CUTS[i]:]:
                    modules.append(module)
                sequential = nn.Sequential(*modules)
                sequential2 = sequential.copy()
                sequentials.append(sequential2.to(device))
                models['models{}'.format(k+1)] = sequentials
            else:
                for module in model_all[constant.CUTS[i]:constant.CUTS[i+1]]:
                    modules.append(module)
                sequential = nn.Sequential(*modules)
                sequential1 = sequential.copy()
                sequentials.append(sequential1.to(device))
    
    return local_models, models

def setup3_4(model_all, device, constant):
    local_models={}
    for i in range(1):
        modules = []
        sequential = []
        for module in model_all:
            modules.append(module)
        sequential = nn.Sequential(*modules)
        local_models['model{}'.format(i)] = sequential.to(device)
    
    models = {}
    for k in range(constant.CLIENTS):
        sequentials = []
        for i in range(1):  # thor = 1
            modules = []
            for module in model_all:
                modules.append(module)
            sequential = nn.Sequential(*modules)
            sequential1 = sequential.copy()
            sequentials.append(sequential1.to(device))
            models['models{}'.format(k+1)] = sequentials
    
    return local_models, models

def average_weights(models, local_models, nosyft, constant, thor):
    #average all 
    # for k in range(CLIENTS):
    #     for i in range(constant.THOR):
    #         models['models{}'.format(k+1)][i] =models['models{}'.format(k+1)][i].get()
    
    #average only last part
    if nosyft == True:
        for k in range(constant.CLIENTS):
            models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor]
    else:
        for k in range(constant.CLIENTS):
            models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor]

    # for i in range(constant.THOR):
    #     params=[]
    #     params_new = []
    #     params_temp=[]
    #     params = list(local_models['model{}'.format(i)].parameters())

    #     for k in range(CLIENTS):
    #         params_temp.append(list(models["models{}".format(k+1)][i].parameters()))
        
    #     for u in range(len(params)):
    #         avg = 0.0
    #         for j in range(CLIENTS):
    #             avg += params_temp[j][u]
    #         params_new.append(avg/CLIENTS)

    #     for u in range(len(params_temp)):
    #         for param_index in range(len(params)):
    #             params_temp[u][param_index].set_(params_new[param_index])

    params=[]
    params_new = []
    params_temp=[]
    params = list(local_models['model{}'.format(thor)].parameters())

    for k in range(constant.CLIENTS):
        models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor].get()

    for k in range(constant.CLIENTS):
        params_temp.append(list(models["models{}".format(k+1)][thor].parameters()))
    
    for u in range(len(params)):
        avg = 0.0
        for j in range(constant.CLIENTS):
            #if nosyft remove .get()
            # avg += params_temp[j][u].get()
            avg += params_temp[j][u]
        params_new.append(avg/constant.CLIENTS)

    # with torch.no_grad():
    for u in range(constant.CLIENTS):
    # for u in range(len(params_temp)):!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for param_index in range(len(params)):
            params_temp[u][param_index].set_(params_new[param_index])
        # for param_index, params in enumerate(models["models{}".format(k+1)][thor].parameters()):
        #     params.data.copy_(params_new[param_index])
    print("2 Weight Averaging Done!")

    
def average_weights_4(models, local_models, nosyft, constant, thor):
    #average all 
    # for k in range(CLIENTS):
    #     for i in range(constant.THOR):
    #         models['models{}'.format(k+1)][i] =models['models{}'.format(k+1)][i].get()
    
    #average only last part
    if nosyft == True:
        for k in range(constant.CLIENTS):
            models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor]
    else:
        for k in range(constant.CLIENTS):
            models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor]

    params=[]
    params_new = []
    params_temp=[]
    # params = list(local_models['model{}'.format(thor)].parameters())

    for k in range(constant.CLIENTS):
        models['models{}'.format(k+1)][thor] = models['models{}'.format(k+1)][thor].get()

    for k in range(constant.CLIENTS):
        params_temp.append(list(models["models{}".format(k+1)][thor].parameters()))
    
    for idx, (param_name, _) in enumerate(local_models['model{}'.format(thor)].named_parameters()):
        layer_idx = int(param_name.split(".", 1)[0])
        if layer_idx >= constant.CUTS[1]:
            avg = 0.0
            for j in range(constant.CLIENTS):
                #if nosyft remove .get()
                # avg += params_temp[j][idx].get()
                avg += params_temp[j][idx]
            # params_new.append(avg/CLIENTS)
            for u in range(constant.CLIENTS):
                params_temp[u][idx].set_(avg/constant.CLIENTS)

    # # with torch.no_grad():
    # for u in range(CLIENTS):
    # # for u in range(len(params_temp)):
    #     for idx, (param_name, _) in enumerate(local_models['model{}'.format(thor)].named_parameters()):
    #         layer_idx = int(param_name.split(".", 1)[0])
    #         if layer_idx >= CUTS[1]:
    #             params_temp[u][idx].set_(params_new[idx])
    #     # for param_index, params in enumerate(models["models{}".format(k+1)][thor].parameters()):
    #     #     params.data.copy_(params_new[param_index])
    print("4 Weight Averaging Done!")