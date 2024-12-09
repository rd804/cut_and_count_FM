import numpy as np
import argparse
import os
from matplotlib import pyplot as plt

#datasets = ['extended1_norm', 'extended1_without']
#datasets = ['extended3_norm', 'extended3_without']
#datasets = ['extended1']
#dataset = 'extended1'
dataset = 'extended2'
#
#windows = [f'window{i}' for i in range(1, 10, 1)]
trails = [i for i in range(5)]
n_samples = np.array([250000, 400000, 500000])
print(n_samples)
samples_path = f'results/fm_{dataset}'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

save_path = f'results/fm_{dataset}'

#data_path = 'data/data_ranit'

path = f'{samples_path}/fm_hidd_256_blocks_4_f_20/'

for n_sample in n_samples/5:
    #print(f'Generating samples: {n_sample * 5}')
   # for dataset in datasets:
        #for window in windows:
            print(f'Ensembling {dataset}: {int(n_sample)*5}')
            #new_sample_path = f'{samples_path}'
            #if not os.path.exists(new_sample_path):
             #   os.makedirs(new_sample_path)
            samples_ensembled = []

            for tries in trails:
                sample_path = f'{path}/{dataset}_{tries}'
                samples = np.load(f'{sample_path}/samples.npy')
                random_indices = np.random.choice(samples.shape[0], int(n_sample), replace=False)
                samples_ensembled.append(samples[random_indices])
            
            samples_ensembled = np.array(samples_ensembled)
            samples_ensembled = np.concatenate(samples_ensembled, axis=0)

            #data_sample = np.load(f'{data_path}/{dataset}/{window}/data/innerdata.npy')

            # print(f'max of mass samples {np.max(samples_ensembled[:,0])}')
            # print(f'min of mass samples {np.min(samples_ensembled[:,0])}')
            # print(f'max of mass data {np.max(data_sample[:,0])}')
            # print(f'min of mass data {np.min(data_sample[:,0])}')
            
            if int(n_sample)==50_000:
                for feature in range(samples_ensembled.shape[1]):
                    plt.hist(samples_ensembled[:,feature], bins=100, histtype='step', label='Samples', density=True)
                   # plt.hist(data_sample[:,feature], bins=100, histtype='step', label='Data', density=True)
                    plt.legend()
                    plt.savefig(f'{save_path}/hist_feature_{feature}.png')
                    plt.close()

            
            np.save(f'{save_path}/samples_{5*int(n_sample)}.npy', samples_ensembled)

