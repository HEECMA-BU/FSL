import torch
import syft as sy
from torch import nn, optim
import time

from FSL_algorithm.resources.dc_loss import DistanceCorrelationLossWrapper


class MultiSplitNN(torch.nn.Module):
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        self.outputs = [None]*len(self.models)
        self.inputs = [None]*len(self.models)
        super().__init__()

    def forwardVal(self, x):
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        intermediate = self.outputs[0].copy()

        for i in range(1, len(self.models)):
            self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
            if self.outputs[i-1].location != self.models[i].location:
                self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()               
            self.outputs[i] = self.models[i](self.inputs[i])
        return self.outputs[-1], intermediate #!!
        
    def forward(self, x):
        begin_clients = time.time()
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        forward_clients = time.time() - begin_clients
        intermediate = self.outputs[0].copy()
        
        begin_server = time.time()
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
        # begin_clients = time.time()
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        # forward_clients = time.time() - begin_clients

        # begin_server = time.time()
        # for i in range(1, len(self.models)):
        #     self.inputs[i] = self.outputs[i-1].detach().requires_grad_()
        #     if self.outputs[i-1].location != self.models[i].location:
        #         self.inputs[i] = self.inputs[i].move(self.models[i].location).requires_grad_()               
        #     self.outputs[i] = self.models[i](self.inputs[i])
        # forward_server = time.time() - begin_server
        return self.outputs[0]

    def backwardA_NoPeek(self):
        begin_clients = time.time()
        # for i in range(len(self.models)-2, -1, -1):
        #     # grad_in = self.inputs[i+1].grad.copy()
        #     # if self.outputs[i].location != self.inputs[i+1].location:
        #     #     grad_in = grad_in.move(self.outputs[i].location)
        #     criterion = NoPeekLoss(0.1)
        #     loss = criterion(self.inputs[i], self.outputs[i])
        #     loss.backward()
        #     # self.outputs[i].backward()
        criterion = DistanceCorrelationLossWrapper(0.1)
        loss = criterion(self.inputs[0], self.outputs[0])
        loss.backward()
        backward_client = time.time() - begin_clients
        return backward_client

    # def backwardA_NoPeek(self):
    #     for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):
    #         criterion = NoPeekLoss(0.1)
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

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()
        
    def stepA(self):
        self.optimizers[0].step()

    def step(self):
        for opt in self.optimizers:
            opt.step()
    
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
    def __init__(self, modelsA, modelsB, optimizersA, optimizersB, num_of_batches):
        self.modelsA = modelsA
        self.modelsB = modelsB
        self.optimizersA = optimizersA
        self.optimizersB = optimizersB
        self.num_of_batches = num_of_batches
        self.outputsA = [None]*len(self.modelsA)
        self.inputsA = [None]*len(self.modelsA)
        self.outputsB = [None]*len(self.modelsB)
        self.inputsB = [None]*len(self.modelsB)
        super().__init__()
        
    def forward(self, x, i):
        self.inputsA[i] = x
        self.outputsA[i] = self.modelsA[i](self.inputsA[i])
        intermediate = self.outputsA[i].copy()
        self.inputsB[0] = self.outputsA[i].detach().requires_grad_()
        if self.outputsA[i].location != self.modelsB[0].location:
            self.inputsB[0] = self.inputsB[0].move(self.modelsB[0].location).requires_grad_()               
        self.outputsB[0] = self.modelsB[0](self.inputsB[0])
        return self.outputsB[-1], intermediate# !!
       
    # def forwardA(self, xA):
    #     for i, x in enumerate(xA):
    #         self.inputsA[i] = x
    #         self.outputsA[i] = self.modelsA[i](self.inputsA[i])

    def forwardA(self, xA):
        ret = []
        for i, x in enumerate(xA):
            self.inputsA[i] = x
            self.outputsA[i] = self.modelsA[i](self.inputsA[i])   
            ret.append(self.outputsA[i])
        return ret
           
    def for_back_B(self, target, xA):       
        loss=0
        target_remote_arr=[]
        x_remote_arr=[]
        # target_remote = target[0].location._objects[target[0].id_at_location]
        target_remote = target[0].get()
        target[0] = target_remote
        # x_remote = self.outputsA[0].location._objects[self.outputsA[0].id_at_location]
        x_remote = self.outputsA[0].copy().get()
        all_target = target_remote
        all_x = x_remote
        for i in range(1,len(target)):
            # target_remote = target[i].location._objects[target[i].id_at_location]
            target_remote = target[i].get()
            target[i] = target_remote
            # x_remote = self.outputsA[i].location._objects[self.outputsA[i].id_at_location]
            x_remote = self.outputsA[i].copy().get()
            all_target = torch.cat((all_target, target_remote),0)
            all_x = torch.cat((all_x, x_remote),0)

        third_target = all_target.send(self.modelsB[0].location)
        inputsB  = all_x.send(self.modelsB[0].location).requires_grad_()
        
        self.outputsB[0] = self.modelsB[0](inputsB)
        pred = self.outputsB[-1]

        # criterion = NoPeekLoss(0.1)
        # intermediate = torch.cat((self.outputsA[0].copy().get(), self.outputsA[1].copy().get()), 0)
        # xA = torch.cat((xA[0].copy().get(), xA[1].copy().get()), 0)
        # intermediate = intermediate.send(self.modelsB[0].location)
        # xA = xA.send(self.modelsB[0].location).requires_grad_()
        # loss = criterion(xA, intermediate, pred, third_target)


        # criterion = NoPeekLoss(0.1)
        # intermediate =  torch.cat((self.outputsA[0], self.outputsA[1]),0)
        # xA =  torch.cat((xA[0].send(), xA[1]),0)
        # loss = criterion(xA, intermediate, pred, third_target)



        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, third_target)


        third_target.get()
        del third_target
        # g = make_dot(loss)
        # g.view()
        loss.backward()
        return loss, inputsB, pred
     
    def backwardA(self, batch_size, inputsB):
        grad_in = inputsB.grad.copy()
        
        inputsB = inputsB.get()
        del inputsB
        #grad_inA = [None]*len(self.modelsA)
        for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):
            # temp = grad_in.location._objects[grad_in.id_at_location]
            temp = grad_in[i*batch_size:(i+1)*batch_size].copy()
            # temp = temp.send(o.location)
            temp = temp.move(o.location)
            # temp = grad_in.copy().move(o.location)

            # g = make_dot(o)
            # g.view()
            o.backward(temp)
            temp = temp.get()
            del temp
            del o
            del xA

            # criterion = NoPeekLoss(0.1)
            # # xA = torch.cat((self.inputsA[0].copy().get(), self.inputsA[1].copy().get()), 0)
            # # xA = xA.send(o.location)
            # # intermediate = torch.cat((self.outputsA[0].copy().get(), self.outputsA[1].copy().get()), 0)
            # # intermediate = intermediate.send(o.location)
            # loss = criterion(xA, o)
            # loss.backward(retain_graph=False)
        
        #     grad_inA[i] = temp[i*batch_size:(i+1)*batch_size].move(self.modelsB[0].location)

        # grad_avg = (grad_inA[0].clone()+grad_inA[1].clone())/2
        # grad_avg = grad_avg.move(self.inputsA[0].location)
        # self.inputsA[0].grad = grad_avg
        # grad_avg = grad_avg.move(self.inputsA[1].location)
        # self.inputsA[1].grad = grad_avg

    def backwardA_NoPeek(self):
        for i, (o, xA) in enumerate(zip(self.outputsA, self.inputsA)):

            criterion = DistanceCorrelationLossWrapper(0.1)
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

    def step(self, batch_idx):
        for opt in self.optimizersA:
            opt.step()

        for opt in self.optimizersB:
            opt.step()

    def stepA(self):
        for opt in self.optimizersA:
            opt.step()

        # for opt in self.optimizersB:
        #     opt.step()
            
    
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
