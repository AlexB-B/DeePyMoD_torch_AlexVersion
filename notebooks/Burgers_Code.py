#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:56:40 2019

@author:
"""

# This is the code from the burger's equation example except I have pulled out the code as the formatting of ip-nbs on spyder is rubbish

import numpy as np
import torch
from deepymod_torch.library_function import library_1D_in
from deepymod_torch.DeepMod import DeepMoD

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

np.random.seed(40)

##############

data = np.load('../data/processed/burgers.npy', allow_pickle=True).item()
print('Shape of grid:', data['x'].shape)

##############
'''
fig, ax = plt.subplots()
im = ax.contourf(data['x'], data['t'], np.real(data['u']))
ax.set_xlabel('x')
ax.set_ylabel('t')
fig.colorbar(mappable=im)

plt.show()
'''
#############

X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))

print(X.shape, y.shape)

############

noise_level = 0.05
y_noisy = y + noise_level * np.std(y) * np.random.randn(y.size, 1)

############

number_of_samples = 1000

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_noisy[idx, :][:number_of_samples], dtype=torch.float32)

#############
'''
fig, axes = plt.subplots(ncols=3, figsize=(15, 4))

im0 = axes[0].contourf(data['x'], data['t'], np.real(data['u']), cmap='coolwarm')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_title('Ground truth')

im1 = axes[1].contourf(data['x'], data['t'], y_noisy.reshape(data['x'].shape), cmap='coolwarm')
axes[1].set_xlabel('x')
axes[1].set_title('Noisy')

sampled = np.array([y_noisy[index, 0] if index in idx[:number_of_samples] else np.nan for index in np.arange(data['x'].size)])
sampled = np.rot90(sampled.reshape(data['x'].shape)) #array needs to be rotated because of imshow

im2 = axes[2].imshow(sampled, aspect='auto', cmap='coolwarm')
axes[2].set_xlabel('x')
axes[2].set_title('Sampled')

fig.colorbar(im1, ax=axes.ravel().tolist())

plt.show()
'''
#############

optim_config = {'lambda': 10**-5, 'max_iterations': 20000}

#############

network_config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1}

############

lib_config = {'type': library_1D_in, 'poly_order': 2, 'diff_order': 3}

############

sparse_coeff_vector, sparsity_mask, network = DeepMoD(X_train, y_train, network_config, lib_config, optim_config)

############

print('Final result:')
print(sparse_coeff_vector, sparsity_mask)