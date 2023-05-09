import torch
import syft as sy
from torch import nn, optim
import time

from FSL_algorithm.resources.dc_loss import DistanceCorrelationLossWrapper


class MultiSplitNN(torch.nn.Module):
    def __init__(self, models, optimizers, CUTS=[-1,-1]):
        self.models = models
        self.optimizers = optimizers
        self.outputs = [None]*len(self.models)
        self.inputs = [None]*len(self.models)
        self.cuts = CUTS
        super().__init__()

    def forwardVal(self, x):
        begin_clients = time.time()
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        forward_clients = time.time() - begin_clients
        intermediate = self.outputs[0].copy()

        begin_server = time.time()
        if len(self.models) > 1:
            for i in range(1, len(self.models)):
                self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
                if self.outputs[i-1].location != self.models[i].location:
                    self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()               
                self.outputs[i] = self.models[i](self.inputs[i])
        forward_server = time.time() - begin_server
        return self.outputs[-1], forward_clients, forward_server, intermediate #!!
        
    # def forward(self, x, logger):
    #     # torch.cuda.reset_max_memory_allocated()
    #     begin_clients = time.time()
    #     self.inputs[0] = x
    #     self.outputs[0] = self.models[0](self.inputs[0])    # + 602,112 
    #     forward_clients = time.time() - begin_clients
    #     intermediate = self.outputs[0].copy()   # + 602,112 
    #     # logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
    #     #                                     sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
        
    #     # torch.cuda.reset_max_memory_allocated()
    #     begin_server = time.time()
    #     for i in range(1, len(self.models)):
    #         self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
    #         if self.outputs[i-1].location != self.models[i].location:
    #             self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()          # + 602,112        
    #         self.outputs[i] = self.models[i](self.inputs[i])                                            # + 1,268,736
    #     forward_server = time.time() - begin_server
    #     # logger.debug('{} {}'.format(sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
    #     #                                     sum(torch.cuda.max_memory_cached() for i in range(torch.cuda.device_count()))))
    #     return self.outputs[-1], forward_clients, forward_server, intermediate
    
    def forward(self, x):
        begin_clients = time.time()
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        forward_clients = time.time() - begin_clients
        intermediate = self.outputs[0].copy()
        
        begin_server = time.time()
        if len(self.models) > 1:
            for i in range(1, len(self.models)):
                self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
                if self.outputs[i-1].location != self.models[i].location:
                    self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()               
                self.outputs[i] = self.models[i](self.inputs[i])
        forward_server = time.time() - begin_server
        return self.outputs[-1], forward_clients, forward_server, intermediate
        
    def backward(self):
        begin_clients = time.time()
        for i in range(len(self.models)-2, -1, -1):
            grad_in = self.inputs[i+1].grad.copy()
            if self.outputs[i].location != self.inputs[i+1].location:
                grad_in = grad_in.move(self.outputs[i].location)
            self.outputs[i].backward(grad_in)
        backward_client = time.time() - begin_clients
        return backward_client

    def forwardA(self, x):
        begin_clients = time.time()
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        forward_clients = time.time() - begin_clients

        # begin_server = time.time()
        # for i in range(1, len(self.models)):
        #     self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
        #     if self.outputs[i-1].location != self.models[i].location:
        #         self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()               
        #     self.outputs[i] = self.models[i](self.inputs[i])
        # forward_server = time.time() - begin_server
        return self.outputs[0], forward_clients

    def backwardA_NoPeek(self):
        begin_clients = time.time()
        # for i in range(len(self.models)-2, -1, -1):
        #     # grad_in = self.inputs[i+1].grad.copy()
        #     # if self.outputs[i].location != self.inputs[i+1].location:
        #     #     grad_in = grad_in.move(self.outputs[i].location)
        #     criterion = NoPeekLoss(0.3)
        #     loss = criterion(self.inputs[i], self.outputs[i])
        #     loss.backward()
        #     # self.outputs[i].backward()
        criterion = DistanceCorrelationLossWrapper(0.3)
        loss = criterion(self.inputs[0], self.outputs[0])
        loss.backward()
        backward_client = time.time() - begin_clients
        return backward_client

    # def backwardA_NoPeek(self):
    #     for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):
    #         criterion = NoPeekLoss(0.3)
    #         loss = criterion(xA, o)
    #         loss.backward()

    # def backwardA(self, batch_size, inputsB):
    #     grad_in = inputsB.grad.copy()
        
    #     inputsB.get()
    #     del inputsB
    #     #grad_inA = [None]*len(self.modelsA)
    #     for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):
    #         # temp = grad_in.location._objects[grad_in.id_at_location]
    #         temp = grad_in.copy().get()
    #         temp = temp.send(o.location)

    #         # g = make_dot(o)
    #         # g.view()
    #         o.backward(temp[i*batch_size:(i+1)*batch_size])
    #         temp.get()
    #         del temp

    def set_parameter_requires_grad(self, train_shared = True):
        for param in self.models[0].parameters():
            param.requires_grad_(False)
            param.requires_grad = False
            # loc = param.grad.location
            # param.grad = param.grad.get()
            # # param.grad = None   # handle momentum
            # param.grad.set_(None)
            # param.grad = param.grad.send(loc)
            
        for param_name, param in self.models[0].named_parameters():
            layer_idx = int(param_name.split(".", 1)[0])
            # print(layer_idx, type(layer_idx))
            if train_shared:
                if layer_idx >= self.cuts[1]:
                    param.requires_grad_(True)
                    param.requires_grad = True
            else:
                if layer_idx < self.cuts[1]:
                    param.requires_grad_(True)
                    param.requires_grad = True

    def get_parameter_requires_grad(self):
        for k,param in self.models[0].named_parameters():
            print(k, param.copy().get().requires_grad, param.requires_grad)

    def get_parameter_requires_grad_(self):
        for idx, param in enumerate(self.models[0].parameters()):
            print(idx, param.copy().get().requires_grad, param.requires_grad)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()
        
    def stepA(self):
        step_time = []
        t0 = time.time()
        self.optimizers[0].step()
        t1 = time.time()
        step_time.append(t1-t0)
        return step_time

    # def step(self):
    #     client_weight_update = 0
    #     server_weight_update = 0
    #     for idx, opt in enumerate(self.optimizers):
    #         if idx == 0:
    #             client_weight_update = time.time()
    #             opt.step()
    #             client_weight_update = time.time() - client_weight_update
    #         if idx == 1:
    #             server_weight_update = time.time()
    #             opt.step()
    #             server_weight_update = time.time() - server_weight_update
    #     return client_weight_update, server_weight_update
    def step(self):
        step_time = []
        for opt in self.optimizers:
            t0 = time.time()
            opt.step()
            t1 = time.time()
            step_time.append(t1-t0)
        return step_time
    def step_4(self, train_shared):
        step_time = []
        if train_shared:
            t0 = time.time()
            self.optimizers[1].step()
            t1 = time.time()
            step_time.append(t1-t0)
        else:
            t0 = time.time()
            self.optimizers[0].step()
            t1 = time.time()
            step_time.append(t1-t0)
        # for opt in self.optimizers:
        #     t0 = time.time()
        #     opt.step()
        #     t1 = time.time()
        #     step_time.append(t1-t0)
        return step_time
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def send_model(self, images, labels, name):
        self.models[1] = self.models[1].send(name)
        #labels = labels.move(self.models[-1].location)
    
    def copy_model(self, model, name, i):
        self.models[i] = model.copy().send(name)

    def send_model_to_secure_worker(self, secure_worker):
        self.models[1].move(secure_worker)
                   
    @property
    def location(self):
        return self.models[0].location if self.models and len(self.models) else None

class SingleSplitNN(torch.nn.Module):
    def __init__(self, modelsA, modelsB, optimizersA, optimizersB):
        self.modelsA = modelsA
        self.modelsB = modelsB
        self.optimizersA = optimizersA
        self.optimizersB = optimizersB
        # self.num_of_batches = num_of_batches
        self.outputsA = [None]*len(self.modelsA)
        self.inputsA = [None]*len(self.modelsA)
        self.outputsB = [None]*len(self.modelsB)
        self.inputsB = [None]*len(self.modelsB)
        super().__init__()
        
    def forward(self, x, i):
        forward_client = time.time()
        self.inputsA[i] = x
        self.outputsA[i] = self.modelsA[i](self.inputsA[i])
        forward_client = time.time() - forward_client
        intermediate = self.outputsA[i].copy()

        begin_server = time.time()
        self.inputsB[0] = self.outputsA[i].detach().requires_grad_()
        if self.outputsA[i].location != self.modelsB[0].location:
            self.inputsB[0] = self.inputsB[0].move(self.modelsB[0].location).requires_grad_()               
        self.outputsB[0] = self.modelsB[0](self.inputsB[0])
        begin_server = time.time() - begin_server
        return self.outputsB[-1], forward_client, begin_server, intermediate# !!
       
    # def forwardA(self, xA):
    #     for i, x in enumerate(xA):
    #         self.inputsA[i] = x
    #         self.outputsA[i] = self.modelsA[i](self.inputsA[i])

    def forwardA(self, xA, ready_clis_l):
        ret = []
        for i, x in zip(ready_clis_l, xA):
            self.inputsA[i] = x
            self.outputsA[i] = self.modelsA[i](self.inputsA[i])   
            ret.append(self.outputsA[i])
        return ret
           
    def for_back_B(self, target, xA, ready_clis_l):       

        B_forward_time = time.time()
        loss=0
        target_remote_arr=[]
        x_remote_arr=[]
        # target_remote = target[0].location._objects[target[0].id_at_location]
        target_remote = target[0].get()
        target[0] = target_remote
        # x_remote = self.outputsA[0].location._objects[self.outputsA[0].id_at_location]
        x_remote = self.outputsA[ready_clis_l[0]].copy().get()
        all_target = target_remote
        all_x = x_remote
        for i in range(1,len(target)):
            # target_remote = target[i].location._objects[target[i].id_at_location]
            target_remote = target[i].get()
            target[i] = target_remote
            # x_remote = self.outputsA[i].location._objects[self.outputsA[i].id_at_location]
            x_remote = self.outputsA[ready_clis_l[i]].copy().get()
            all_target = torch.cat((all_target, target_remote),0)
            all_x = torch.cat((all_x, x_remote),0)

        third_target = all_target.send(self.modelsB[0].location)
        inputsB  = all_x.send(self.modelsB[0].location).requires_grad_()
        # !!!!!!!!!!!!!!!!!!!!!!!!!! timeit !!!!!!!!!!!!!!
        self.outputsB[0] = self.modelsB[0](inputsB)
        pred = self.outputsB[-1]

        # criterion = NoPeekLoss(0.3)
        # intermediate = torch.cat((self.outputsA[0].copy().get(), self.outputsA[1].copy().get()), 0)
        # xA = torch.cat((xA[0].copy().get(), xA[1].copy().get()), 0)
        # intermediate = intermediate.send(self.modelsB[0].location)
        # xA = xA.send(self.modelsB[0].location).requires_grad_()
        # loss = criterion(xA, intermediate, pred, third_target)


        # criterion = NoPeekLoss(0.3)
        # intermediate =  torch.cat((self.outputsA[0], self.outputsA[1]),0)
        # xA =  torch.cat((xA[0].send(), xA[1]),0)
        # loss = criterion(xA, intermediate, pred, third_target)



        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, third_target)
        B_forward_time = time.time()-B_forward_time
        
        third_target.get()
        del third_target

        # g = make_dot(loss)
        # g.view()
        B_backward_time = time.time()
        loss.backward()
        B_backward_time = time.time() - B_backward_time
        return loss, inputsB, pred, B_forward_time, B_backward_time
     
    def backwardA(self, batch_size, inputsB, ready_clis_l):
        backward_clients = time.time()
        grad_in = inputsB.grad.copy()
    
        for idx, i in enumerate(ready_clis_l):
            temp = grad_in[idx*batch_size:(idx+1)*batch_size]   # TODO: try to combine forward interm in the same way.
            temp = temp.move(self.outputsA[i].location)
            self.outputsA[i].backward(temp)

        backward_clients = time.time() - backward_clients
        return backward_clients

    def backwardA_NoPeek(self, ready_clis_l):
        for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):
            if i in ready_clis_l:
                criterion = DistanceCorrelationLossWrapper(0.3)
                # xA = torch.cat((self.inputsA[0].copy().get(), self.inputsA[1].copy().get()), 0)
                # xA = xA.send(o.location)
                # intermediate = torch.cat((self.outputsA[0].copy().get(), self.outputsA[1].copy().get()), 0)
                # intermediate = intermediate.send(o.location)
                loss = criterion(xA, o)
                loss.backward()


    def zero_grads(self):
        for opt in self.optimizersA:
            opt.zero_grad()
        
        for opt in self.optimizersB:
            opt.zero_grad()

    # def step(self, batch_idx):
    #     for opt in self.optimizersA:
    #         # opt.step()
    #         if batch_idx >= self.num_of_batches:
    #             opt.step()    # this will call privacy engine's step()
    #             opt.zero_grad()
    #         else:
    #             opt.virtual_step()   # this will call privacy engine's virtual_step()

    #     for opt in self.optimizersB:
    #         opt.step()
    #         # if batch_idx >= self.num_of_batches:
    #         #     opt.step()    # this will call privacy engine's step()
    #         #     opt.zero_grad()
    #         # else:
    #         #     opt.virtual_step()   # this will call privacy engine's virtual_step()
    #         #   

    def step(self, ready_clis_l):
        client_weight_update = time.time()
        for cli_idx in ready_clis_l:
        # for opt_idx, opt in enumerate(self.optimizersA):
            # if opt_idx in ready_clis_l:
            self.optimizersA[cli_idx].step()
        client_weight_update = time.time()-client_weight_update

        server_weight_update = time.time()
        for opt in self.optimizersB:
            opt.step()
        server_weight_update = time.time()-server_weight_update
        return client_weight_update, server_weight_update

    def stepA(self, ready_clis_l):
        client_weight_update = time.time()
        for cli_idx in ready_clis_l:
        # for opt_idx, opt in enumerate(self.optimizersA):
            # if opt_idx in ready_clis_l:
            self.optimizersA[cli_idx].step()
        client_weight_update = time.time()-client_weight_update

        # for opt in self.optimizersB:
        #     opt.step()
        return client_weight_update
            
    
    def train(self):
        for model in self.modelsA:
            model.train()
        for model in self.modelsB:
            model.train()
    
    def eval(self):
        for model in self.modelsA:
            model.eval()
        for model in self.modelsB:
            model.eval()
                       
    @property
    def location(self):
        return self.modelsA[0].location if self.modelsA and len(self.modelsA) else None
