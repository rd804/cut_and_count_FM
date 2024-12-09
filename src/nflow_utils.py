import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.utils import *
from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from torch import nn
import nflows
import src.flows as fnn
from functools import partial
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import *
from nflows.transforms.coupling import *
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
# import bactchnorm
from nflows.distributions.uniform import BoxUniform as Uniform
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet
from nflows.transforms.made import MADE
#from nflows.distributions.uniform import Uniform
#from torch import distributions



def r_anode(model_S,model_B,w,optimizer, data_loader, params,scheduler=False ,device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
    #    model_B.eval()

    else:
        model_S.eval()
     #   model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        #data_CR = data[1].to(device)
        model_B_log_prob = data[1].to(device).flatten()


        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,1:-1],context=data_SR[:,0].reshape(-1,1))
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
           # model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
            #                                     transform=False)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            
            
            
            data_p = w * torch.exp(model_S_log_prob) + (1-w) * torch.exp(model_B_log_prob)
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
    

    total_loss /= len(data_loader)


   # if mode == 'train':
    # set batch norm layers to eval mode
    # what dafaq is this doing?
  #      print('setting batch norm layers to eval mode')
   #     has_batch_norm = False
   #     for module in model_S.modules():
    #        if isinstance(module, fnn.BatchNormFlow):
     #           has_batch_norm = True
       #         module.momentum = 0
        # forward pass to update batch norm statistics
      #  if has_batch_norm:
       #     with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
        #        model_S(data_loader.dataset.tensors[1][:,1:-1].to(data[0].device),
         #           data_loader.dataset.tensors[1][:,0].to(data[0].device).reshape(-1,1).float())

          #  for module in model_S.modules():
           #     if isinstance(module, fnn.BatchNormFlow):
            #        module.momentum = 1
    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss

def r_anode_uncond(model_S,model_B,w,optimizer, data_loader,
             params,scheduler=False ,device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
    #    model_B.eval()

    else:
        model_S.eval()
     #   model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        #data_CR = data[1].to(device)
        model_B_log_prob = data[1].to(device).flatten()


        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
           # model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
            #                                     transform=False)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            
            
            
            data_p = w * torch.exp(model_S_log_prob) + (1-w) * torch.exp(model_B_log_prob)
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
    

    total_loss /= len(data_loader)


   # if mode == 'train':
    # set batch norm layers to eval mode
    # what dafaq is this doing?
  #      print('setting batch norm layers to eval mode')
   #     has_batch_norm = False
   #     for module in model_S.modules():
    #        if isinstance(module, fnn.BatchNormFlow):
     #           has_batch_norm = True
       #         module.momentum = 0
        # forward pass to update batch norm statistics
      #  if has_batch_norm:
       #     with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
        #        model_S(data_loader.dataset.tensors[1][:,1:-1].to(data[0].device),
         #           data_loader.dataset.tensors[1][:,0].to(data[0].device).reshape(-1,1).float())

          #  for module in model_S.modules():
           #     if isinstance(module, fnn.BatchNormFlow):
            #        module.momentum = 1
    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss



def r_anode_mass(model_S,model_B,w, \
                 optimizer, data_loader, params,scheduler=False ,device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
    #    model_B.eval()

    else:
        model_S.eval()
     #   model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        #data_CR = data[1].to(device)
        model_B_log_prob = data[1].to(device).flatten()
        mass_density_sig = data[2].to(device).flatten()
        mass_density_bkg = data[3].to(device).flatten()



        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,1:-1],context=data_SR[:,0].reshape(-1,1))
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
           # model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
            #                                     transform=False)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            
            
            
            data_p = w*torch.exp(model_S_log_prob)*mass_density_sig + \
            (1-w)*torch.exp(model_B_log_prob)*mass_density_bkg
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
    

    total_loss /= len(data_loader)


   # if mode == 'train':
    # set batch norm layers to eval mode
    # what dafaq is this doing?
  #      print('setting batch norm layers to eval mode')
   #     has_batch_norm = False
   #     for module in model_S.modules():
    #        if isinstance(module, fnn.BatchNormFlow):
     #           has_batch_norm = True
       #         module.momentum = 0
        # forward pass to update batch norm statistics
      #  if has_batch_norm:
       #     with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
        #        model_S(data_loader.dataset.tensors[1][:,1:-1].to(data[0].device),
         #           data_loader.dataset.tensors[1][:,0].to(data[0].device).reshape(-1,1).float())

          #  for module in model_S.modules():
           #     if isinstance(module, fnn.BatchNormFlow):
            #        module.momentum = 1
    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss

def r_anode_mass_joint_untransformed(model_S,model_B,w, \
                 optimizer, data_loader, params,scheduler=False ,device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood',w_train = False):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
    #    model_B.eval()

    else:
        model_S.eval()
     #   model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        #data_CR = data[1].to(device)
        model_B_log_prob = data[1].to(device).flatten()
      #  mass_density_sig = data[2].to(device).flatten()
        mass_density_bkg = data[2].to(device).flatten()



        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,:-1])
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
           # model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
            #                                     transform=False)
           # model_S_log_prob = evaluate_log_prob_mass(model_S, data_SR, params_SR,
                                                #     transform=True)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            

            if mode == 'train':

                if not w_train:
                    data_p = w*torch.exp(model_S_log_prob) + \
                    (1-w)*torch.exp(model_B_log_prob)*mass_density_bkg
                else:
                    data_p = torch.sigmoid(w)*torch.exp(model_S_log_prob) + \
                    (1-torch.sigmoid(w))*torch.exp(model_B_log_prob)*mass_density_bkg
            else:
                with torch.no_grad():
                    if not w_train:
                        data_p = w*torch.exp(model_S_log_prob) + \
                        (1-w)*torch.exp(model_B_log_prob)*mass_density_bkg
                    else:
                        data_p = torch.sigmoid(w)*torch.exp(model_S_log_prob) + \
                        (1-torch.sigmoid(w))*torch.exp(model_B_log_prob)*mass_density_bkg
            
            data_loss = torch.log(data_p + 1e-32)


        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_S.parameters(),1)
            optimizer.step()
            if scheduler:
                scheduler.step()
    

    total_loss /= len(data_loader)

    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss


def r_anode_mass_joint(model_S,model_B,w, \
                 optimizer, data_loader, params,scheduler=False ,device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
    #    model_B.eval()

    else:
        model_S.eval()
     #   model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        #data_CR = data[1].to(device)
        model_B_log_prob = data[1].to(device).flatten()
      #  mass_density_sig = data[2].to(device).flatten()
        mass_density_bkg = data[2].to(device).flatten()



        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,:-1])
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
           # model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
            #                                     transform=False)
          #  model_S_log_prob = evaluate_log_prob_mass(model_S, data_SR, params_SR,
                                                  #   transform=True)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            
            
            
            data_p = w*torch.exp(model_S_log_prob) + \
            (1-w)*torch.exp(model_B_log_prob)*mass_density_bkg
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
    

    total_loss /= len(data_loader)

    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss


def r_anode_mass_joint_multigpu(model_S,w, epoch, \
                 optimizer, data_loader, sampler, params,scheduler=False, 
                 mode='train', data_loss_expr = 'true_likelihood',
                 local_rank=0,rank=0):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
        sampler.set_epoch(epoch)

    else:
        model_S.eval()

    total_loss = 0
    params_CR = params['CR']
    params_SR = params['SR']

   # average_grad_norm = 0

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(local_rank)
        model_B_log_prob = data[1].to(local_rank).flatten()
        mass_density_bkg = data[2].to(local_rank).flatten()

        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
            model_S_log_prob = model_S.module.log_prob(data_SR[:,:-1])

           # if batch_idx==0:
            #    assert model_S_log_prob.shape == model_B_log_prob.shape
             #   print(f'value of w: {w}')    
            
            
            
            data_p = w*torch.exp(model_S_log_prob) + \
            (1-w)*torch.exp(model_B_log_prob)*mass_density_bkg
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        if rank==0:
            total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            # print the gradient norm
           # print('gradient norm')
           # norm_sum = 0
            #for p in model_S.module.parameters():
             #   norm_sum += p.grad.norm().item()
            #print(norm_sum)
            #average_grad_norm += norm_sum
            #print(f'gradient norm: {torch.norm(model_S.module.transform_net[0].weight.grad)}')
            #from torch.nn.utils import clip_grad_norm_
            #clip_grad_norm_(model_S.parameters(),0.01)
            optimizer.step()
            if scheduler:
                scheduler.step()
    
    if rank==0:
        total_loss /= len(data_loader)
       # print(f'average grad norm: {average_grad_norm/len(data_loader)}')
        #print(len(data_loader))


    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')

    if rank==0:
        return total_loss
    else:
        return 0.0


def anode_uncond(model,train_loader, optimizer, params, device='cpu', mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0


    for batch_idx, data in enumerate(train_loader):


        data = data[0].to(device)
        #params = params.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        
        loss = - evaluate_log_prob_uncond(model, data, params).mean()
        total_loss += loss.item()

        if mode == 'train':
            loss.backward()        
            optimizer.step()

    total_loss /= len(train_loader)

    if mode == 'train':
        # set batch norm layers to eval mode
        # what dafaq is this doing?
        print('setting batch norm layers to eval mode')
        has_batch_norm = False
        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                has_batch_norm = True
                module.momentum = 0
        # forward pass to update batch norm statistics
        if has_batch_norm:
            with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
                model(train_loader.dataset.tensors[0][:,1:-1].to(data[0].device))

            for module in model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1

    return total_loss


def anode(model,train_loader, optimizer, params, device='cpu', mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0


    for batch_idx, data in enumerate(train_loader):


        data = data[0].to(device)
        #params = params.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        
        loss = - evaluate_log_prob(model, data, params).mean()
        total_loss += loss.item()

        if mode == 'train':
            loss.backward()        
            optimizer.step()

    total_loss /= len(train_loader)

    if mode == 'train':
        # set batch norm layers to eval mode
        # what dafaq is this doing?
        print('setting batch norm layers to eval mode')
        has_batch_norm = False
        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                has_batch_norm = True
                module.momentum = 0
        # forward pass to update batch norm statistics
        if has_batch_norm:
            with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
                model(train_loader.dataset.tensors[0][:,1:-1].to(data[0].device),
                    train_loader.dataset.tensors[0][:,0].to(data[0].device).reshape(-1,1).float())

            for module in model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1

    return total_loss
    
def train_model(model,train_loader, optimizer, params, device='cpu', mode='train',
                clip_grad_norm = False, grad_norm_clip = 1,sample = False):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    if sample:
        samples=[]

    for batch_idx, data in enumerate(train_loader):

       # print(data)
        data = data[0].to(device)
        #print(data.shape)
        #params = params.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        
        if not sample:
            loss = -model.log_prob(data[:,1:-1],context=data[:,0].reshape(-1,1)).mean()
        else:
            samples,loss = model.sample_and_log_prob(data[:,1:-1],context=data[:,0].reshape(-1,1))
            samples.append(samples)
            loss = -loss.mean()


        total_loss += loss.item()

        if mode == 'train':
            loss.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),grad_norm_clip)
            optimizer.step()
            #if scheduler:
                #scheduler.step()
           # scheduler.step()

    total_loss /= len(train_loader)

 
    if sample:
        return total_loss, samples
    else:
        return total_loss


# def evaluate_log_prob_mass(model, data, preprocessing_params, transform=False, mode='train'):
#     if mode == 'train':
#         logit_prob = model.log_prob(data[:, :-1])
#     else:
#         model.eval()
#         with torch.no_grad():
#             logit_prob = model.log_prob(data[:, :-1])
#        # logit_prob = model.log_prob(data[:, 1:-1])
    
#     if transform:
#         log_prob = logit_prob.flatten() + torch.log(
#             2 * (1 + torch.cosh(data[:, 0] * preprocessing_params["std"][0] + preprocessing_params["mean"][0]))
#             / (preprocessing_params["std"][0] * (preprocessing_params["max"][0] - preprocessing_params["min"][0]))
#         +1e-32)
#     else:
#         log_prob = logit_prob.flatten()
#     return log_prob

def evaluate_log_prob(model, data, preprocessing_params, transform=False):
    logit_prob = model.log_probs(data[:, 1:-1], data[:,0].reshape(-1,1))
    
    if transform:
        log_prob = logit_prob.flatten() + torch.sum(
        torch.log(
            2 * (1 + torch.cosh(data[:, 1:-1] * preprocessing_params["std"] + preprocessing_params["mean"]))
            / (preprocessing_params["std"] * (preprocessing_params["max"] - preprocessing_params["min"]))
        +1e-32), axis=1
    ) # type: ignore
    else:
        log_prob = logit_prob.flatten()
    return log_prob

# def evaluate_log_prob_uncond(model, data, preprocessing_params, transform=False):
#     logit_prob = model.log_probs(data[:, 1:-1])
    
#     if transform:
#         log_prob = logit_prob.flatten() + torch.sum(
#         torch.log(
#             2 * (1 + torch.cosh(data[:, 1:-1] * preprocessing_params["std"] + preprocessing_params["mean"]))
#             / (preprocessing_params["std"] * (preprocessing_params["max"] - preprocessing_params["min"]))
#         +1e-32), axis=1
#     ) # type: ignore
#     else:
#         log_prob = logit_prob.flatten()
#     return log_prob

# def flows_for_gaussian(gaussian_dim = 2, num_transforms = 2, num_blocks = 3, 
#                        hidden_features = 32, device = 'cpu'):

#     base_dist = nflows.distributions.normal.StandardNormal(shape=[gaussian_dim])

#     list_transforms = []
#     for _ in range(num_transforms):
#         list_transforms.append(
#             nflows.transforms.permutations.RandomPermutation(gaussian_dim)
#         )
#         list_transforms.append(
#             nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
#                 features=gaussian_dim, 
#                 hidden_features=hidden_features,
#                 num_blocks=num_blocks,
#                 activation=torch.nn.functional.relu
#             )
#         )

#     transform = nflows.transforms.base.CompositeTransform(list_transforms)

#     flow = nflows.flows.base.Flow(transform, base_dist).to(device)

#     return flow

def flows_model_affine_AR(num_layers = 8, num_features=4, num_blocks = 2, 
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):


    num_layers = num_layers
    base_dist = StandardNormal(shape=[num_features])

    transforms = []
    for _ in range(num_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(features=num_features, 
                                                hidden_features=hidden_features, 
                                                use_residual_blocks=False,
                                                context_features=context_features, random_mask=random_mask,
                                                num_blocks=num_blocks, use_batch_norm=use_batch_norm,
                                                activation=F.leaky_relu, dropout_probability=dropout_probability))
        transforms.append(BatchNorm(features=num_features))
        transforms.append(RandomPermutation(features=num_features))

        # transform.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,))

    transform = CompositeTransform(transforms)

    model = Flow(transform, base_dist).to(device)

    return model

def flows_model_additive_coupling(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask}
            rec_flow_blocks.append(AdditiveCouplingTransform(**flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model

def flows_model_linear_AR(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):


    num_layers = num_layers
    base_dist = StandardNormal(shape=[num_features])

    transforms = []
    for _ in range(num_layers):
        transforms.append(
            MaskedPiecewiseLinearAutoregressiveTransform(features=num_features, 
                                                hidden_features=hidden_features, 
                                                context_features=context_features,
                                                num_bins=8,
                                                num_blocks=num_blocks, use_batch_norm=use_batch_norm,
                                                  dropout_probability=dropout_probability))
        transforms.append(BatchNorm(features=num_features))
        transforms.append(RandomPermutation(features=num_features))

    transform = CompositeTransform(transforms)

    model = Flow(transform, base_dist).to(device)

    return model


def flows_model_linear_coupling(num_layers = 8, num_features=4, 
                hidden_features = 64, device = 'cpu', num_blocks = 2,
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask,
                                        'num_bins': 8,
                                        'tails':'linear',
                                        'tail_bound': 4}
            rec_flow_blocks.append(PiecewiseLinearCouplingTransform(**flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model



def flows_model_cubic_coupling(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask,
                                        'num_bins': 8,
                                        'tails':'linear',
                                        'tail_bound': 4}
            rec_flow_blocks.append(PiecewiseCubicCouplingTransform(**flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model


def flows_model_quadratic_coupling(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask,
                                        'num_bins': 8,
                                        'tails':'linear',
                                        'tail_bound': 4}
            rec_flow_blocks.append(PiecewiseQuadraticCouplingTransform(**flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model




def flows_model_affine_coupling(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask}
            rec_flow_blocks.append(AffineCouplingTransform(**flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model


def flows_model_RQS_coupling(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    net = partial(ResidualNet, hidden_features=hidden_features, 
                  context_features=context_features,
                    num_blocks=num_blocks, dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm)

    rec_flow_blocks = []
    for _ in range(num_layers):
            mask = torch.from_numpy(np.random.choice([True, False], 
                                                     num_features, 
                                                     p=[0.5, 0.5]))*1
            
            flow_params_rec_energy = {'transform_net_create_fn': net,
                                        'mask': mask,
                                        'num_bins': 8,
                                        'tails':'linear',
                                        'tail_bound': 4}
            rec_flow_blocks.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    **flow_params_rec_energy))
            rec_flow_blocks.append(BatchNorm(num_features))
            rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model
    
def flows_model_quadratic_AR(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    flow_params_rec_energy = {'num_blocks': num_blocks, #num of layers per block
                                'features': num_features,
                                'context_features': context_features,
                                'hidden_features': hidden_features,
                                'use_residual_blocks': False,
                                'use_batch_norm': use_batch_norm,
                                'dropout_probability': dropout_probability,
                                'activation':getattr(F, 'leaky_relu'),
                                'random_mask': random_mask,
                                'num_bins': 3,
                                'tails':'linear',
                                'tail_bound': 4,
                                'min_bin_width': 1e-6,
                                'min_bin_height': 1e-6,
                                'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(num_layers):
        rec_flow_blocks.append(
            MaskedPiecewiseQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
       # rec_flow_blocks.append(BatchNorm(num_features))
        rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model

def flows_model_cubic_AR(num_layers = 8, num_features=4, num_blocks = 2,
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1):
    
    flow_params_rec_energy = {'num_blocks': num_blocks, #num of layers per block
                                'features': num_features,
                                'context_features': context_features,
                                'hidden_features': hidden_features,
                                'use_residual_blocks': False,
                                'use_batch_norm': use_batch_norm,
                                'dropout_probability': dropout_probability,
                                'activation':getattr(F, 'leaky_relu'),
                                'random_mask': random_mask,
                                'num_bins': 5}
    rec_flow_blocks = []
    for _ in range(num_layers):
        rec_flow_blocks.append(
            MaskedPiecewiseCubicAutoregressiveTransform(
                **flow_params_rec_energy))
       # rec_flow_blocks.append(BatchNorm(num_features))
        rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model

def flows_model_RQS_AR(num_layers = 8, num_features=4, num_blocks = 2, 
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.1,
                use_residual_blocks = False,
                num_bins = 6,
                tail_bound = 9):
    
    flow_params_rec_energy = {'num_blocks': num_blocks, #num of layers per block
                                'features': num_features,
                                'context_features': context_features,
                                'hidden_features': hidden_features,
                                'use_residual_blocks': use_residual_blocks,
                                'use_batch_norm': use_batch_norm,
                                'dropout_probability': dropout_probability,
                                'activation':getattr(F, 'leaky_relu'),
                                'random_mask': random_mask,
                                'num_bins': num_bins,
                                'tails':'linear',
                                'tail_bound': tail_bound,
                                'min_bin_width': 1e-6,
                                'min_bin_height': 1e-6,
                                'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(num_layers):
        rec_flow_blocks.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
        rec_flow_blocks.append(RandomPermutation(num_features))

    #    rec_flow_blocks.append(BatchNorm(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    
    #base_distribution = Uniform(low=torch.zeros(num_features), high=torch.ones(num_features))
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])
   # rec_flow_base_distribution = NflowsUniform(shape=[num_features],low=-8.0,high=8.0, device=device)
    model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    #model = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model

def flows_model_RQS_old(num_layers = 6, num_features=4, num_blocks = 2, 
                hidden_features = 64, device = 'cpu',
                context_features = 1, random_mask = True, 
                use_batch_norm = True, dropout_probability = 0.2):
    
    flow_params_rec_energy = {'num_blocks': num_blocks, #num of layers per block
                                'features': num_features,
                                'context_features': context_features,
                                'hidden_features': hidden_features,
                                'use_residual_blocks': False,
                                'use_batch_norm': use_batch_norm,
                                'dropout_probability': dropout_probability,
                                'activation':getattr(F, 'leaky_relu'),
                                'random_mask': random_mask,
                                'num_bins': 8,
                                'tails':'linear',
                                'tail_bound': 9,
                                'min_bin_width': 1e-6,
                                'min_bin_height': 1e-6,
                                'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(num_layers):
        rec_flow_blocks.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
     #   rec_flow_blocks.append(BatchNorm(num_features))
        rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model_S = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution).to(device)

    return model_S



from nflows.distributions.base import Distribution
class NflowsUniform(Distribution):
    """A multivariate uniform"""
    def __init__(self, shape,low,high, device='cpu'):
        super().__init__()
        self._shape = torch.Size(shape)
        self._low=low
        self._high=high
        self.device=device

       # print(f'low: {low}')
       # print(f'high: {high}')
       # print(f'shape: {shape}')
       # print(f'self._shape: {self._shape}')
#        self.register_buffer("_log_z",
#                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
#                                          dtype=torch.float64),
#                             persistent=False)
    def _log_prob(self, inputs, context=None):
        # Note: the context is ignored.
      #  print(inputs.shape)
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
       # print(self._shape[0])
       # print
       # print(torch.ones(inputs.shape[0],device=inputs.device))
        neg_energy = 1/(self._high-self._low)**(self._shape[0])*torch.ones(inputs.shape[0],device=inputs.device)
        return neg_energy

    def _sample(self, num_samples, context=None):
        if context is None:
            return torch.rand(num_samples, *self._shape).to(self.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = self._low+(self._high-self._low)*torch.rand(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])
    def _mean(self, context=None):
#        if context is None:
#            return self._log_z.new_zeros(self._shape)
#        else:
            # The value of the context is ignored, only its size is taken into account.
        return context.new_zeros(context.shape[0], *self._shape)
