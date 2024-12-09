# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
#print(sys.path)
sys.path.append('.')
#from nflows.nn.nets import ResidualNet
#from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
#from src.flows import *
from src.flow_matching import *
import os
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
# import train_test_split
from sklearn.model_selection import train_test_split, ShuffleSplit
import argparse
import wandb
import pickle
import sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--frequencies', type=int, default=3)

parser.add_argument('--ensemble_size', type=int, default=50)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--data_dir', type=str, default='data/extended1', help='data directory')
parser.add_argument('--time_embedding', action='store_true', help='if time embedding is used')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_flow_matching')
parser.add_argument('--wandb_job_type', type=str, default='test_time_embedding')
parser.add_argument('--wandb_run_name', type=str, default='extended3')


args = parser.parse_args()
save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'

if os.path.exists(f'{save_path}samples.npy'):
    print(f'already done {args.wandb_run_name}')
    sys.exit()


if not os.path.exists(save_path):
    os.makedirs(save_path)

CUDA = True
device = torch.device(args.device if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="cathode_ext", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)
CR_data = np.load(f'{args.data_dir}/outerdata.npy')
SR_data = np.load(f'{args.data_dir}/innerdata.npy')

print('x_train shape', CR_data.shape)
n_features = int(CR_data.shape[1]-2)
print('n_features', n_features)

#if args.wandb:
 #   wandb.config.update({'true_w': true_w, 'sigma': sigma, 'n_features':n_features})


pre_parameters_cpu = preprocess_params_fit(CR_data)
x_train = preprocess_params_transform(CR_data, pre_parameters_cpu)

# save pre_parameters
with open(save_path+'pre_parameters.pkl', 'wb') as f:
    pickle.dump(pre_parameters_cpu, f)


if not args.shuffle_split:    
    data_train = x_train
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.5, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break

traintensor = torch.from_numpy(data_train.astype('float32')).to(device)

print('X_train shape', traintensor.shape)

pre_parameters = {}
for key in pre_parameters_cpu.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters_cpu[key].astype('float32')).to(device)


for i in range(1):
    print(f'Ensemble {i}')

    if args.time_embedding:
        print('time embedding')
        model = Conditional_ResNet(context_features=1, 
                            input_dim=n_features, 
                            device=device, 
                            hidden_dim=256, 
                            num_blocks=3, 
                            dropout_probability=0.2, 
                            time_embed=SinusoidalPosEmb(256))
        print(model)
    else:
        print('no time embedding')
        model = Conditional_ResNet(frequencies=args.frequencies, 
                                   context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                use_batch_norm=True, 
                                dropout_probability=0.2)

                            
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainloss = train_flow(traintensor, model, optimizer=optimizer,
            num_epochs=args.epochs, batch_size=args.batch_size,
            device=device, sigma_fm=0.001,
            save_model=True, model_path=f'{save_path}')





mass = torch.from_numpy(SR_data[:,0].reshape(-1,1)).to(device).float()
data = torch.from_numpy(SR_data[:,1:-1]).to(device).float()
noise1 = torch.randn_like(data).to(device).float()
noise2 = torch.randn_like(data).to(device).float()
noise = torch.cat([noise1, noise2], dim=0)
mass = torch.cat([mass, mass], dim=0)
samples = sample_minibatches(model, noise, mass, start=0.0, end=1.0)
samples = torch.concat([mass, samples], axis=1)
samples = torch.concat([samples, torch.ones(samples.shape[0],1).to(device)], axis=1)
samples_inverse = inverse_transform(samples, pre_parameters)


np.save(f'{save_path}samples.npy', samples_inverse.cpu().detach().numpy())
np.save(f'{save_path}samples_preprocessed.npy', samples.cpu().detach().numpy())

## %%
for i in range(1,n_features+1,1):
    plt.hist(SR_data[:,i],bins=100,density=True,histtype='step')
    plt.hist(samples_inverse[:,i].cpu().detach().numpy(),bins=100,density=True,
            histtype='step')
    plt.savefig(f'{save_path}feature_{i}.png')
    plt.close()

sample_data = np.vstack((samples.detach().cpu().numpy(),SR_data))
sample_labels = np.concatenate([np.ones(samples.shape[0]), np.zeros(SR_data.shape[0])]).reshape(-1,1)
sample_data = np.concatenate([sample_data[:,:-1], sample_labels], axis=1)

sample_data_train, sample_data_val = train_test_split(sample_data, test_size=0.5, random_state=args.seed)
clf = HistGradientBoostingClassifier(max_iter=1000,verbose=0)
clf.fit(sample_data_train[:,1:-1], sample_data_train[:,-1])
predict = clf.predict_proba(sample_data_val[:,1:-1])[:,1]
auc = roc_auc_score(sample_data_val[:,-1], predict)
print('AUC sample quality: ', auc)

if args.wandb:
    wandb.log({'AUC_sample_quality': auc})
