# %%
import numpy as np
import matplotlib.pyplot as plt
from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
from src.flows import *
import os
from sklearn.metrics import roc_curve, roc_auc_score
from nflows import transforms, distributions, flows
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
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=0, help='split number')
parser.add_argument('--data_dir', type=str, default='data/extended1', help='data directory')
parser.add_argument('--model_type', type=str, default='affine', help='affine or RQS')
parser.add_argument('--device', type=str, default='cuda:2', help='gpu to train on')

parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_anode_CR')
parser.add_argument('--wandb_job_type', type=str, default='affine_4_features')
parser.add_argument('--wandb_run_name', type=str, default='try_')


args = parser.parse_args()
save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'

if os.path.exists(f'{save_path}best_val_loss_scores.npy'):
    print(f'already done {args.wandb_run_name}')
    sys.exit()


if not os.path.exists(save_path):
    os.makedirs(save_path)

CUDA = True
device = torch.device(args.device if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="r_anode", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)

SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)


#CR_data = np.concatenate((CR_data,CR_data[:,-1].reshape(-1,1)),axis=1)
n_features = int(CR_data.shape[1]-2)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma, 'n_features':n_features})



print('x_train shape', CR_data.shape)
print('true_w', true_w)
print('sigma', sigma)
print('n_features', n_features)

pre_parameters_cpu = preprocess_params_fit(CR_data)
x_train = preprocess_params_transform(CR_data, pre_parameters_cpu)

# save pre_parameters
with open(save_path+'pre_parameters.pkl', 'wb') as f:
    pickle.dump(pre_parameters_cpu, f)


if not args.shuffle_split:    
    data_train, data_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.5, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break



_x_test = np.load(f'{args.data_dir}/x_test.npy')
#_x_test = np.concatenate((_x_test[:,:5],_x_test[:,-1].reshape(-1,1)),axis=1)
x_test = preprocess_params_transform(_x_test, pre_parameters_cpu)


# %%
for i in range(n_features):
    plt.hist(data_train[:,i+1], bins=100, alpha=0.5, label='train')
    plt.show()

# %%
len(x_train)//2, len(x_test[x_test[:,-1]==0]), 

# %%
traintensor = torch.from_numpy(data_train.astype('float32')).to(device)
valtensor = torch.from_numpy(data_val.astype('float32')).to(device)
testtensor = torch.from_numpy(x_test.astype('float32')).to(device)

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
print('X_test shape', testtensor.shape)

pre_parameters = {}
for key in pre_parameters_cpu.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters_cpu[key].astype('float32')).to(device)

train_tensor = torch.utils.data.TensorDataset(traintensor)
val_tensor = torch.utils.data.TensorDataset(valtensor)
test_tensor = torch.utils.data.TensorDataset(testtensor)


# Use the standard pytorch DataLoader
batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(test_tensor, batch_size=test_batch_size, shuffle=False)



# # %%
# define savepath
if args.model_type == 'affine':
            model = flows_model_affine(device=device, num_features=n_features, context_features=1,
                                       num_layers=15, hidden_features=128, num_blocks=1)
if args.model_type == 'RQS':
            model = flows_model_RQS(device=device, num_features=n_features, context_features=1,
                                       num_layers=15, hidden_features=64, num_blocks=2)
    


trainloss_list=[]
valloss_list=[]

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

##############
# train model
for epoch in range(args.epochs):
    trainloss=train_model(model,trainloader, optimizer, pre_parameters, device=device, mode='train')
    valloss=train_model(model,trainloader, optimizer, pre_parameters, device=device, mode='val')

    torch.save(model.state_dict(), save_path+'model_CR_'+str(epoch)+'.pt')

    valloss_list.append(valloss)
    trainloss_list.append(trainloss)

    print('epoch: ', epoch, 'trainloss: ', trainloss, 'valloss: ', valloss)

    if np.isnan(trainloss) or np.isnan(valloss):
        print(' nan loss ')
        if args.wandb:
            wandb.finish()
        sys.exit()

    if args.wandb:
        wandb.log({'train_loss': trainloss, 'val_loss': valloss, 'epoch': epoch})

    if epoch % 50 == 0:
        if ~(np.isnan(trainloss) or np.isnan(valloss)):
            model.eval()
            with torch.no_grad():
            # TODO : add context feature to model.sample
                 x_samples = model.sample(1, testtensor[:,0][testtensor[:,-1]==0].reshape(-1,1)).detach().cpu()
            x_samples = x_samples.reshape(-1,n_features)
            x_samples = torch.hstack((x_samples, torch.ones((len(x_samples),1))))
            x_samples_untransformed = torch.hstack((testtensor[:,0][testtensor[:,-1]==0].reshape(-1,1).detach().cpu(), x_samples))
            x_samples = inverse_transform(x_samples_untransformed, pre_parameters_cpu).numpy()
            x_samples = x_samples[~np.isnan(x_samples).any(axis=1)]
            print('x_samples shape', x_samples.shape)

            figure = plt.figure(figsize=(5,5))
            for i in range(1,n_features+1):
                      #  plt.subplot(n_features//2,2,i)
                        #if dims > 1:
                       # plt.hist(_x_test[:,i][_x_test[:,-1]==1],bins=50, density=True, label=f'sig', histtype='step')
                        plt.hist(_x_test[:,i][_x_test[:,-1]==0],bins=50, density=True, label=f'true', histtype='step')
                        plt.hist(x_samples[:,i],bins=50, density=True, label=f'sample', histtype='step')
                        plt.legend()
                       # plt.show()

            #if args.wandb:
           #     wandb.log({f'nflow_B': wandb.Image(figure)})
            
                        plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/nflow_B_{epoch}.png')
                        plt.close()

            # classifier test
            sample_data = np.vstack((x_samples_untransformed,x_test[x_test[:,-1]==0]))
            sample_data_train, sample_data_val = train_test_split(sample_data, test_size=0.5, random_state=args.seed)
            clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000, random_state=args.seed,verbose=0)
    
            clf.fit(sample_data[:,1:-1], sample_data[:,-1])
            predict = clf.predict_proba(sample_data_val[:,1:-1])[:,1]

            auc = roc_auc_score(sample_data_val[:,-1], predict)
            print('AUC_sample_quality: ', auc)

            if args.wandb:
                  wandb.log({'AUC_sample_quality': auc})      
            


trainloss_list=np.array(trainloss_list)
valloss_list=np.array(valloss_list)



# load best model
valloss_list=np.array(valloss_list)
min_epoch=np.argmin(valloss_list)
print('min epoch SR: ',min_epoch)

#min_epoch = args.epochs-1
model.load_state_dict(torch.load(save_path+'model_CR_'+str(min_epoch)+'.pt'))
torch.save(model.state_dict(), save_path+'model_CR_best.pt')


# check density estimation
not_lowest_10 = np.argsort(valloss_list)[10:]
file_list = ['model_CR_{}.pt'.format(i) for i in not_lowest_10]
print(f'Deleting models not in lowest 10 epochs')
for file_ in file_list:
    os.remove(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/{file_}')


np.save(save_path+'trainloss_list.npy', trainloss_list)
np.save(save_path+'valloss_list.npy', valloss_list)

if args.wandb:
    wandb.finish()
    


plt.plot(trainloss_list, label='train')
plt.plot(valloss_list, label='val')
plt.legend()
plt.savefig(save_path+'loss.png')
plt.show()





# check density estimation
lowest_10 = np.argsort(valloss_list)[:10]
file_list = ['model_CR_{}.pt'.format(i) for i in lowest_10]
total_samples = 280_000
sample_size = total_samples//10
samples = []
mass_SR = SR_data[:,0].astype('float32')

for i,file_ in enumerate(file_list):
        print(f'loading {file_}')
        # choose sample_size mass points from SR
        mass_SR_sample = np.random.choice(mass_SR, size=sample_size, replace=True)

        model.load_state_dict(torch.load(save_path+file_))
        model.eval()
       # mask = int(i*sample_size):int((i+1)*sample_size)
        with torch.no_grad():
        # TODO : add context feature to model.sample
                x_samples = model.sample(1, torch.from_numpy(mass_SR_sample).reshape(-1,1).to(device)).detach().cpu()
        x_samples = x_samples.reshape(-1,n_features)
        x_samples = torch.hstack((x_samples, torch.ones((len(x_samples),1))))
        x_samples_untransformed = torch.hstack((torch.from_numpy(mass_SR_sample).reshape(-1,1), x_samples))
        x_samples = inverse_transform(x_samples_untransformed, pre_parameters_cpu).numpy()
        samples.append(x_samples)

samples = np.array(np.vstack(samples))
print('samples shape', samples.shape)
print(total_samples)


# Classifier test
_extra_bkg = np.load(f'{args.data_dir}/extrabkg.npy')
#_extra_bkg = np.concatenate((_extra_bkg[:,:5],_extra_bkg[:,-1].reshape(-1,1)),axis=1)
extra_bkg = preprocess_params_transform(_extra_bkg, pre_parameters_cpu)
samples_processed = preprocess_params_transform(samples, pre_parameters_cpu)

sample_data = np.vstack((samples_processed,extra_bkg))
sample_data_train, sample_data_val = train_test_split(sample_data, test_size=0.5, random_state=args.seed)
clf = HistGradientBoostingClassifier(validation_fraction=0.5,max_iter=1000, random_state=args.seed,verbose=0)

clf.fit(sample_data[:,1:-1], sample_data[:,-1])
predict = clf.predict_proba(sample_data_val[:,1:-1])[:,1]

auc = roc_auc_score(sample_data_val[:,-1], predict)
print('AUC_sample_quality: ', auc)



np.save(f'{save_path}/samples.npy', samples)


