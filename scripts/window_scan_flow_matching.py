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
from scipy.stats import rv_histogram
#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--frequencies', type=int, default=3)
parser.add_argument('--x_train', type=str, default='CR')
parser.add_argument('--ensemble_size', type=int, default=50)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--data_dir', type=str, default='data/extended1', help='data directory')
parser.add_argument('--context_embedding', action='store_true', help='if time embedding is used')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
parser.add_argument('--baseline', action='store_true', help='if baseline is used')
parser.add_argument('--non_linear_context', action='store_true', help='if non linear context is used')
#parser.add_argument('--try', type=int, default=0)
parser.add_argument('--window_index', type=int, default=5)
parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_flow_matching')
parser.add_argument('--wandb_job_type', type=str, default='test_time_embedding')
parser.add_argument('--wandb_run_name', type=str, default='extended3')


# rank          = int(os.environ["SLURM_PROCID"])
# gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
# local_rank = rank - gpus_per_node * (rank // gpus_per_node)



args = parser.parse_args()
#args.wandb_run_name = f'{args.wandb_run_name}_{local_rank}'

#args.wandb_run_name = f'{args.wandb_run_name}_{args.window_index}'

save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'


if not os.path.exists(save_path):
    os.makedirs(save_path)

if os.path.exists(save_path+'samples_SR.npy'):
    print('already done')
    sys.exit()

CUDA = True

#device = torch.device(f'cuda:{local_rank}' if CUDA else "cpu")   
device = torch.device(args.device if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="cut_and_count", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)

SR_center = 3.5 + (args.window_index - 5)*0.1
min_SR_mass = SR_center - 0.2
max_SR_mass = SR_center + 0.2



SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, 
                                                  n_sig = 0, 
                                                  resample_seed = args.seed,
                                                  resample = args.resample,
                                                  minmass = min_SR_mass,
                                                  maxmass = max_SR_mass)

# if args.baseline:
#     SR_data = np.concatenate([SR_data[:,:5], SR_data[:,-1].reshape(-1,1)], axis=1)
#     CR_data = np.concatenate([CR_data[:,:5], CR_data[:,-1].reshape(-1,1)], axis=1)

#n_features = 4

print('x_train shape', CR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

#baseline = CR_data[:,:5]
#CR_data = np.concatenate([baseline, CR_data[:,-1].reshape(-1,1)], axis=1)
n_features = int(CR_data.shape[1]-2)
print('n_features', n_features)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma, 'n_features':n_features})

if args.x_train == 'CR':
    pre_parameters_cpu = preprocess_params_fit(CR_data)
    x_train = preprocess_params_transform(CR_data, pre_parameters_cpu)
    print('training on CR, training data shape', x_train.shape)
elif args.x_train == 'SR':
    pre_parameters_cpu = preprocess_params_fit(SR_data)
    x_train = preprocess_params_transform(SR_data, pre_parameters_cpu)
    print('training on SR, training data shape', x_train.shape)
elif args.x_train == 'data':
    data_ = np.vstack((CR_data, SR_data))
    data_ = shuffle(data_)
    print('training on data, training data shape', data_.shape)
    pre_parameters_cpu = preprocess_params_fit(data_)
    x_train = preprocess_params_transform(data_, pre_parameters_cpu)
elif args.x_train == 'no_signal':
    data_ = np.vstack((CR_data, SR_data))
    data_ = shuffle(data_)
    data_ = data_[data_[:,-1]==0]
    print('training on no signal, training data shape', data_.shape)
    pre_parameters_cpu = preprocess_params_fit(data_)
    x_train = preprocess_params_transform(data_, pre_parameters_cpu)

# save pre_parameters
with open(save_path+'pre_parameters.pkl', 'wb') as f:
    pickle.dump(pre_parameters_cpu, f)


if args.context_embedding:
    max_mass = np.max(x_train[:,0])
    min_mass = np.min(x_train[:,0])
    context_bins = np.arange(min_mass, max_mass, 0.1)

    np.save(f'{save_path}context_bins.npy', context_bins)

    x_train[:,0] = np.digitize(x_train[:,0], context_bins)



if not args.shuffle_split: 
    data_train, data_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)

    #data_train = x_train   
    #data_train, data_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.5, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break

x_SR = preprocess_params_transform(SR_data, pre_parameters_cpu)

if args.context_embedding:
    x_train[:,0] = np.digitize(x_train[:,0], context_bins)
    #x_test[:,0] = np.digitize(x_test[:,0], context_bins)
    x_SR[:,0] = np.digitize(x_SR[:,0], context_bins)

traintensor = torch.from_numpy(data_train.astype('float32')).to(device)
valtensor = torch.from_numpy(data_val.astype('float32')).to(device)
#testtensor = torch.from_numpy(x_test.astype('float32')).to(device)

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
#print('X_test shape', testtensor.shape)

pre_parameters = {}
for key in pre_parameters_cpu.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters_cpu[key].astype('float32')).to(device)


for i in range(1):
    print(f'Ensemble {i}')

    if args.context_embedding:
        print('context embedding sinusoidal')
        model = Discrete_Conditional_ResNet(context_features=1, 
                            input_dim=n_features, 
                            device=device, 
                            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                            use_batch_norm=True, 
                            dropout_probability=0.2, 
                            context_embed=SinusoidalPosEmb(dim=32,theta=100),
                            non_linear_context=args.non_linear_context)
       # print(model)
    else:
        print('no context embedding')
        model = Conditional_ResNet_time_embed(frequencies=args.frequencies, 
                                context_features=1, 
                                input_dim=n_features, device=device,
                                hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                use_batch_norm=True, 
                                dropout_probability=0.2,
                                non_linear_context=args.non_linear_context)

                            
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
    trainloss, logprob_list, logprob_epoch = train_flow(traintensor, 
            model, valdata=valtensor ,optimizer=optimizer,
            num_epochs=args.epochs, batch_size=args.batch_size,
            device=device, sigma_fm=0.001,
            save_model=True, model_path=f'{save_path}',
            compute_log_likelihood=True,
            likelihood_interval=5, likelihood_start=100,
            early_stop_patience=20,
            wandb_log=args.wandb,
            scheduler=scheduler)
    
log_prob_mean = np.array(logprob_list)

figure = plt.figure()
plt.plot(logprob_epoch,log_prob_mean)
plt.savefig(f'{save_path}val_log_prob.png')
plt.close()

log_prob_mean_sorted = np.argsort(log_prob_mean)
logprob_epoch = np.array(logprob_epoch)
lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]

np.save(f'{save_path}val_logprob.npy', log_prob_mean)
np.save(f'{save_path}val_logprob_epoch.npy', logprob_epoch)


log_prob_mean_sorted = np.argsort(log_prob_mean)
logprob_epoch = np.array(logprob_epoch)
lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]

print('lowest epochs', lowest_epochs)
# import scipy histogram
# import scipy.stats as stats

SR_mass = SR_data[:,0]
SR_hist = np.histogram(SR_mass, bins=60, density=True)
SR_density = rv_histogram(SR_hist)

CR_mass = CR_data[:,0]
CR_hist = np.histogram(CR_mass, bins=100, density=True)
CR_density = rv_histogram(CR_hist)



#mass = torch.from_numpy(SR_mass).to(device).float()
# data = torch.from_numpy(SR_data[:,1:-1]).to(device).float()
# noise1 = torch.randn_like(data).to(device).float()
# noise2 = torch.randn_like(data).to(device).float()
# noise3 = torch.randn_like(data).to(device).float()
# noise = torch.cat([noise1, noise2], dim=0)
# noise = torch.cat([noise, noise3], dim=0)

noise_SR = torch.randn(2000000, n_features).to(device).float()
mass_samples_SR = SR_density.rvs(size=len(noise_SR))

noise_CR = torch.randn(2000000, n_features).to(device).float()
mass_samples_CR = CR_density.rvs(size=len(noise_CR))

mass_samples_CR = mass_samples_CR.reshape(-1,1)
mass_samples_SR = mass_samples_SR.reshape(-1,1)
mass_samples_CR = torch.from_numpy(mass_samples_CR).to(device).float()
mass_samples_SR = torch.from_numpy(mass_samples_SR).to(device).float()

mini_batch_length = len(noise_SR)//10

ensembled_samples_SR = []
ensembled_mass_SR = []

ensembled_samples_CR = []
ensembled_mass_CR = []

#lowest_epochs = np.arange(0,10,1)
for i,epoch in enumerate(lowest_epochs):
    model.load_state_dict(torch.load(f'{save_path}model_epoch_{epoch}.pth'))
    noise_batch = noise_SR[i*mini_batch_length:(i+1)*mini_batch_length]
    mass_batch = mass_samples_SR[i*mini_batch_length:(i+1)*mini_batch_length] 
    samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
    ensembled_samples_SR.append(samples)
    ensembled_mass_SR.append(mass_batch)

    noise_batch = noise_CR[i*mini_batch_length:(i+1)*mini_batch_length]
    mass_batch = mass_samples_CR[i*mini_batch_length:(i+1)*mini_batch_length]
    samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
    ensembled_samples_CR.append(samples)
    ensembled_mass_CR.append(mass_batch)



print('deleting models')

delete_paths = [f'{save_path}model_epoch_{epoch}.pth' for epoch in logprob_epoch if epoch not in lowest_epochs]

for path in delete_paths:
    os.remove(path)


mass_SR = torch.cat(ensembled_mass_SR, dim=0)
samples_SR = torch.cat(ensembled_samples_SR, dim=0)
samples_SR = torch.concat([mass_SR, samples_SR], axis=1)
samples_SR = torch.concat([samples_SR, torch.zeros(samples_SR.shape[0],1).to(device)], axis=1)
samples_inverse_SR = inverse_transform(samples_SR, pre_parameters).cpu().detach().numpy()

mass_CR = torch.cat(ensembled_mass_CR, dim=0)
samples_CR = torch.cat(ensembled_samples_CR, dim=0)
samples_CR = torch.concat([mass_CR, samples_CR], axis=1)
samples_CR = torch.concat([samples_CR, torch.zeros(samples_CR.shape[0],1).to(device)], axis=1)
samples_inverse_CR = inverse_transform(samples_CR, pre_parameters).cpu().detach().numpy()


np.save(f'{save_path}samples_SR.npy', samples_inverse_SR)
np.save(f'{save_path}samples_CR.npy', samples_inverse_CR)
np.save(f'{save_path}samples_preprocessed_SR.npy', samples_SR.cpu().detach().numpy())
np.save(f'{save_path}samples_preprocessed_CR.npy', samples_CR.cpu().detach().numpy())

## %%
for i in range(0,n_features+1,1):
    figure = plt.figure()
    max_val = np.max(SR_data[:,i])
    min_val = np.min(SR_data[:,i])
    bins = np.arange(min_val, max_val, 0.03)
    
    plt.hist(SR_data[:,i],bins=bins,density=True, histtype='stepfilled', label='data', color='gray', alpha=0.5)
    plt.hist(samples_inverse_SR[:,i],bins=bins,density=True,
            histtype='step', label='samples')
   # plt.hist(_x_test[:,i][_x_test[:,-1]==0], bins=bins, density=True, histtype='step', label='true background', color='black')
    plt.legend()
    plt.savefig(f'{save_path}feature_SR_{i}.png')
    if args.wandb:
        #wandb.log({'sic_curve': wandb.Image(figure)})
        wandb.log({f'feature_SR_{i}': wandb.Image(figure)})

    plt.close()

for i in range(0,n_features+1,1):
    figure = plt.figure()
    max_val = np.max(CR_data[:,i])
    min_val = np.min(CR_data[:,i])
    bins = np.arange(min_val, max_val, 0.03)
    
    plt.hist(CR_data[:,i],bins=bins,density=True, histtype='stepfilled', label='data', color='gray', alpha=0.5)
    plt.hist(samples_inverse_CR[:,i],bins=bins,density=True,
            histtype='step', label='samples')
    plt.legend()
    plt.savefig(f'{save_path}feature_CR_{i}.png')
    if args.wandb:
        #wandb.log({'sic_curve': wandb.Image(figure)})
        wandb.log({f'feature_CR_{i}': wandb.Image(figure)})

    plt.close()

if args.wandb:
    wandb.finish()

