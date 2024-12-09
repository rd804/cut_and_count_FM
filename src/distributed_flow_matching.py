import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
import time
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from zuko.utils import odeint
import torch.nn as nn
# from nflows.nn.nets import ResidualNet


from torch.distributions import Normal
from sklearn.metrics import roc_curve, roc_auc_score
import os
import numpy as np
import torch


def train_flow(data, model, optimizer=None, num_epochs=2, batch_size=5,
               device=torch.device('cuda:0'),sigma_fm=0.001, 
               save_model = False, model_path=None, mode='distributed',
    ot=False):

    if ot:
        ot_sampler = OTPlanSampler(method="exact")
    start = time.time()
    losses = []
    for epoch in range(num_epochs):
        data_ = create_loader(data, shuffle=True)
        running_loss = 0
        if epoch % 1 == 0:
            print('epoch', epoch)

        for i in range(len(data_)//batch_size+1):
            optimizer.zero_grad()
            
            x1 = data_[i*batch_size:(i+1)*batch_size].to(device)

            context = x1[:,0].reshape(-1,1)
            x1 = x1[:,1:-1]

            x0 = torch.randn_like(x1).to(device)
            t = torch.rand_like(x0[:, 0].reshape(-1,1)).to(device)

            xt = sample_conditional_pt(x0, x1, t, sigma_fm)
            if ot:
                pi = ot_sampler.get_map(x0, x1)
                i, j = ot_sampler.sample_map(pi, x0.shape[0], replace=False)
                xt = x1[j]
                t = t[j]
                x0 = x0[i]
            

            ut = compute_conditional_vector_field(x0, x1, sigma_fm)

            vt = model(torch.cat([xt, t], dim=-1),context=context)
            loss = torch.mean((vt - ut) ** 2)

            running_loss += loss.item()
           
            loss.backward()
            optimizer.step()

        
        total_loss = running_loss/(len(data_)//batch_size+1)
        losses.append(total_loss)
    
        # if total_loss is lower than all losses in best_loss_array, save model
        if save_model:
            if num_epochs-epoch < 10: 
                print('saving model at epoch', epoch)
                torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')

       # torch.save(model.state_dict(), f'{model_path}_epoch_{epoch}.pth')
       # if epoch % INTERVAL == 0:
    end = time.time()
    print('Time taken: ', end - start)

    return losses
