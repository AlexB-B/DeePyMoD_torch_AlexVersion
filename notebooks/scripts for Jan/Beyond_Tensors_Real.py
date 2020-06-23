# import os
# import sys
# from datetime import datetime
# import pickle
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import torch

# sys.path.append('../src')
# import deepymod_torch.VE_params as VE_params
# import deepymod_torch.VE_datagen as VE_datagen
# from deepymod_torch.DeepMod import run_deepmod

from deepymod_torch.DeepMod import Configuration, DeepMod
import deepymod_torch.training as training
import torch.nn as nn

# random seeding
np_seed = 4
torch_seed = 0
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

##

# general_path = 'Oscilloscope data CRI electronics analogy/'
# specific_path = 'AWG 7V half sinc KELVIN cap 1000/' # It is precisely here that changes the data we are grabbing to test.
# path = general_path + specific_path
path = ''

# Some of these factors are just for saving at the end but...
# ... input_type is used in recalc after DM
# ... omega is used in time scaling.
# ... mech_model is used to predict coeffs and recover mech params
# input_type = 'Strain'
# mech_model = 'GKM'
# func_desc = 'Half Sinc'
omega = 2*np.pi * 5 * 0.1
# Amp = 7

channel_1_data = np.loadtxt(path+'Channel 1 total voltage.csv', delimiter=',', skiprows=3)
channel_2_data = np.loadtxt(path+'Channel 2 voltage shunt resistor.csv', delimiter=',', skiprows=3)

##

lower = 806
upper = -759

voltage_array = channel_1_data[lower:upper, 1:]
voltage_shunt_array = channel_2_data[lower:upper, 1:]
time_array = channel_1_data[lower:upper, :1]

##

# Maxwell shunt
r_shunt = 10.2 # measured using multimeter
# Kelvin shunt
# r_shunt = 10.2 # measured using multimeter

current_array = voltage_shunt_array/r_shunt

##

# 'normalising'
t_sf = omega/1.2 # Aim for this to be such that the T of the scaled data is a bit less than 2pi
V_sf = 1/np.max(abs(voltage_array))
I_sf = 1/np.max(abs(current_array))
scaled_time_array = time_array*t_sf
scaled_voltage = voltage_array*V_sf
scaled_current = current_array*I_sf

# structuring
target_array = np.concatenate((scaled_voltage, scaled_current), axis=1)

##

# random sampling
number_of_samples = scaled_time_array.size

reordered_row_indices = np.random.permutation(scaled_time_array.size)
reduced_time_array = scaled_time_array[reordered_row_indices, :][:number_of_samples]
reduced_target_array = target_array[reordered_row_indices, :][:number_of_samples]

##

time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor(reduced_target_array, dtype=torch.float32)





##

import torch.autograd as auto

def mech_library_real(inputs, **library_config):    
    
    prediction, data = inputs
    
    # The first column of prediction is always strain
    strain_derivs = auto_deriv(data, prediction[:, :1], library_config['diff_order'])
    strain_theta = torch.cat((prediction[:, :1], strain_derivs), dim=1)
    
    # The second column is always stress
    stress_derivs = auto_deriv(data, prediction[:, 1:], library_config['diff_order'])
    stress_theta = torch.cat((prediction[:, 1:], stress_derivs), dim=1)
    
    strain_t = strain_theta[:, 1:2] # Extract the first time derivative of strain
    strain_theta = torch.cat((strain_theta[:, 0:1], strain_theta[:, 2:]), dim=1) # remove this before it gets put into theta
    strain_theta *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
    theta = torch.cat((strain_theta, stress_theta), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
    return [strain_t], theta


def auto_deriv(data, prediction, max_order):
    '''
    data and prediction must be single columned tensors.
    If it is desired to calculate the derivatives of different predictions wrt different data, this function must be called multiple times.
    This function does not return a column with the zeroth derivative (the prediction).
    '''
    
    # First derivative builds off prediction.
    derivs = auto.grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    for _ in range(max_order-1):
        # Higher derivatives chain derivatives from first derivative.
        derivs = torch.cat((derivs, auto.grad(derivs[:, -1:], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]), dim=1)
            
    return derivs

##

percent = 0.05
def thresh_pc(*args): # Keep as full function so that it can be pickled
    return percent

##

library_config = {'library_func': mech_library_real,
                  'diff_order': 3,
                  'coeff_sign': 'positive'}

network_config = {'hidden_dim': 30}

optim_config = {'l1': 10**-5,
                'thresh_func': thresh_pc,
                'lr_coeffs': 0.002}

report_config = {'plot': True}

##

def run_deepmod(data, target, library_config, network_config={}, optim_config={}, report_config={}):
        
    configs = Configuration(library_config, network_config, optim_config, report_config)
        
    ind_vars = data.shape[1]
    tar_vars = target.shape[1]
    hidden_list = configs.network['layers']*[configs.network['hidden_dim']]
    
    model = DeepMod(ind_vars, hidden_list, tar_vars, configs.library['library_func'], configs)
    
    pre_trained_network = configs.network['pre_trained_network']
    if pre_trained_network: # Overides network for pretrained network
        model.network = pre_trained_network
    
    if configs.optim['mse_only_iterations']:
        optimizer = torch.optim.Adam(model.network.parameters(), lr=configs.optim['lr_nn'], betas=configs.optim['betas'], amsgrad=configs.optim['amsgrad'])
        training.train_mse(model, data, target, optimizer)
        prediction, time_deriv_list, sparse_theta_list = model.forward(data)[:3]
    
    model.fit.initial_guess = None
    if configs.optim['use_lstsq_approx']:
        lstsq_guess_list = [np.linalg.lstsq(sparse_theta.detach(), time_deriv.detach(), rcond=None)[0] for sparse_theta, time_deriv in zip(sparse_theta_list, time_deriv_list)]
        model.fit.initial_guess = lstsq_guess_list
        model.fit.coeff_vector = nn.ParameterList([nn.Parameter(torch.tensor(lstsq_guess, dtype=torch.float32)) for lstsq_guess in lstsq_guess_list])
    
    optimizer = torch.optim.Adam(({'params': model.network.parameters(), 'lr': configs.optim['lr_nn']}, {'params': model.fit.coeff_vector.parameters(), 'lr': configs.optim['lr_coeffs']}), betas=configs.optim['betas'], amsgrad=configs.optim['amsgrad'])
    
    training.train_deepmod(model, data, target, optimizer)
    
    model.optimizer = optimizer
    
    return model

##

model = run_deepmod(time_tensor, target_tensor, library_config, network_config, optim_config, report_config)