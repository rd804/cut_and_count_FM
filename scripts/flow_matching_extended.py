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

parser.add_argument('--non_linear_context', action='store_true', help='if non linear context is used')
#parser.add_argument('--try', type=int, default=0)

parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_flow_matching')
parser.add_argument('--wandb_job_type', type=str, default='test_time_embedding')
parser.add_argument('--wandb_run_name', type=str, default='extended3')


# rank          = int(os.environ["SLURM_PROCID"])
# gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
# local_rank = rank - gpus_per_node * (rank // gpus_per_node)



args = parser.parse_args()
#args.wandb_run_name = f'{args.wandb_run_name}_{local_rank}'

save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'

#save_path = f'{save_path}_'

# if os.path.exists(f'{save_path}best_val_loss_scores.npy'):
#     print(f'already done {args.wandb_run_name}')
#     sys.exit()


if not os.path.exists(save_path):
    os.makedirs(save_path)

CUDA = True

#device = torch.device(f'cuda:{local_rank}' if CUDA else "cpu")   
device = torch.device(args.device if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="fast_interpolation", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)

SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

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
    data_train, data_val = train_test_split(x_train, test_size=0.15, random_state=args.seed)

    #data_train = x_train   
    #data_train, data_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.5, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break



_x_test = np.load(f'{args.data_dir}/x_test.npy')
x_test = preprocess_params_transform(_x_test, pre_parameters_cpu)
x_SR = preprocess_params_transform(SR_data, pre_parameters_cpu)

if args.context_embedding:
    x_train[:,0] = np.digitize(x_train[:,0], context_bins)
    x_test[:,0] = np.digitize(x_test[:,0], context_bins)
    x_SR[:,0] = np.digitize(x_SR[:,0], context_bins)




traintensor = torch.from_numpy(data_train.astype('float32')).to(device)
valtensor = torch.from_numpy(data_val.astype('float32')).to(device)
testtensor = torch.from_numpy(x_test.astype('float32')).to(device)

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
print('X_test shape', testtensor.shape)

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
        model = Conditional_ResNet(frequencies=args.frequencies, 
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
mass = torch.from_numpy(SR_data[:,0].reshape(-1,1)).to(device).float()
data = torch.from_numpy(SR_data[:,1:-1]).to(device).float()
noise1 = torch.randn_like(data).to(device).float()
noise2 = torch.randn_like(data).to(device).float()
noise3 = torch.randn_like(data).to(device).float()

noise = torch.cat([noise1, noise2], dim=0)
noise = torch.cat([noise, noise3], dim=0)
mass = torch.cat([mass, mass, mass], dim=0)

if args.context_embedding:
    mass_numpy = mass.cpu().detach().numpy()
    digitized_mass = np.digitize(mass_numpy, context_bins)
    digitized_mass = torch.from_numpy(digitized_mass).to(device).float()

mini_batch_length = len(noise)//10

ensembled_samples = []
ensembled_mass = []

#lowest_epochs = np.arange(0,10,1)
for i,epoch in enumerate(lowest_epochs):
    model.load_state_dict(torch.load(f'{save_path}model_epoch_{epoch}.pth'))
    noise_batch = noise[i*mini_batch_length:(i+1)*mini_batch_length]
    if args.context_embedding:
        digitized_mass_batch = digitized_mass[i*mini_batch_length:(i+1)*mini_batch_length]
        mass_batch = mass[i*mini_batch_length:(i+1)*mini_batch_length]
        samples = sample(model, noise_batch, digitized_mass_batch, start=0.0, end=1.0)
    else:
        mass_batch = mass[i*mini_batch_length:(i+1)*mini_batch_length] 
        samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
    ensembled_samples.append(samples)
    ensembled_mass.append(mass_batch)



print('deleting models')

delete_paths = [f'{save_path}model_epoch_{epoch}.pth' for epoch in logprob_epoch if epoch not in lowest_epochs]

for path in delete_paths:
    os.remove(path)


mass = torch.cat(ensembled_mass, dim=0)
samples = torch.cat(ensembled_samples, dim=0)
samples = torch.concat([mass, samples], axis=1)
samples = torch.concat([samples, torch.ones(samples.shape[0],1).to(device)], axis=1)


samples_inverse = inverse_transform(samples, pre_parameters)

np.save(f'{save_path}samples.npy', samples_inverse.cpu().detach().numpy())
np.save(f'{save_path}samples_preprocessed.npy', samples.cpu().detach().numpy())

## %%
for i in range(1,n_features+1,1):
    figure = plt.figure()
    #bins = np.arange(0, 1, 0.03)
    max_val = np.max(SR_data[:,i])
    min_val = np.min(SR_data[:,i])
    bins = np.arange(min_val, max_val, 0.03)
    plt.hist(SR_data[:,i],bins=bins,density=True, histtype='stepfilled', label='data', color='gray', alpha=0.5)
    plt.hist(samples_inverse[:,i].cpu().detach().numpy(),bins=bins,density=True,
            histtype='step', label='samples')
    plt.hist(_x_test[:,i][_x_test[:,-1]==0], bins=bins, density=True, histtype='step', label='true background', color='black')
    plt.legend()
    plt.savefig(f'{save_path}feature_{i}.png')
    if args.wandb:
        #wandb.log({'sic_curve': wandb.Image(figure)})
        wandb.log({f'feature_{i}': wandb.Image(figure)})

    plt.close()


extrabkg = np.load(f'{args.data_dir}/extrabkg.npy')
extra_bkg = preprocess_params_transform(extrabkg, pre_parameters_cpu)[:266666]

sample_data = np.vstack((samples.detach().cpu().numpy(),x_test[x_test[:,-1]==0]))
sample_data_train, sample_data_val = train_test_split(sample_data, test_size=0.5, random_state=args.seed)

clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0)
clf.fit(sample_data[:,1:-1], sample_data[:,-1])
predict = clf.predict_proba(sample_data_val[:,1:-1])[:,1]

auc = roc_auc_score(sample_data_val[:,-1], predict)
print('AUC_sample_quality: ', auc)

if args.wandb:
    wandb.log({'AUC_sample_quality': auc})

from sklearn.utils.class_weight import compute_sample_weight

iad_data = np.vstack((x_SR[:,1:-1], extra_bkg[:,1:-1]))
iad_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(extra_bkg.shape[0])])
cathode_data = np.vstack((x_SR[:,1:-1], samples.detach().cpu().numpy()[:,1:-1]))
cathode_labels = np.concatenate([np.ones(x_SR.shape[0]), np.zeros(samples.shape[0])])
sample_weights_cathode = compute_sample_weight(class_weight='balanced', y=cathode_labels)
sample_weights_iad = compute_sample_weight(class_weight='balanced', y=iad_labels)

predict_iad = []

for seed in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
    clf.fit(iad_data, iad_labels.reshape(-1,1), sample_weight=sample_weights_iad)
    predict_iad.append(clf.predict_proba(x_test[:,1:-1])[:,1])
# clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0)
# clf.fit(iad_data, iad_labels.reshape(-1,1), sample_weight=sample_weights_iad)
predict_iad = np.mean(predict_iad, axis=0)

sic_score_iad , tpr_score_iad , _ = SIC_cut(x_test[:,-1], predict_iad)

predict_cathode = []

for seed in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000,verbose=0, random_state=seed)
    clf.fit(cathode_data, cathode_labels.reshape(-1,1), sample_weight=sample_weights_cathode)
    predict_cathode.append(clf.predict_proba(x_test[:,1:-1])[:,1])

predict_cathode = np.mean(predict_cathode, axis=0)
sic_score_cathode , tpr_score_cathode , _ = SIC_cut(x_test[:,-1], predict_cathode)

figure = plt.figure()
plt.plot(tpr_score_iad, sic_score_iad, label='IAD')
plt.plot(tpr_score_cathode, sic_score_cathode, label=f'Cathode {args.x_train}')
plt.legend()
if args.wandb:
    wandb.log({'sic_curve': wandb.Image(figure)})
plt.savefig(f'{save_path}sic_curve.png')

plt.close()



np.save(f'{save_path}sic_cathode.npy', sic_score_cathode)
np.save(f'{save_path}tpr_cathode.npy', tpr_score_cathode)
np.save(f'{save_path}sic_iad.npy', sic_score_iad)
np.save(f'{save_path}tpr_iad.npy', tpr_score_iad)

