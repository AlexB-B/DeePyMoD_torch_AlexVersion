# import os
import sys
# from datetime import datetime
# import pickle
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#import sympy as sym
import torch

sys.path.append('../src')
import deepymod_torch.VE_datagen as VE_datagen
# import deepymod_torch.VE_params as VE_params
# from deepymod_torch.DeepMod import run_deepmod

np_seed = 2
torch_seed = 0
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

##

input_type = 'Strain'

# For Boltzmann DG, specific model required for calculation of response given manipulation type. Strain -> GMM, Stress -> GKM.
# For odeint method, no need to consider.
# mech_model = 'GMM' 

E = [1, 1, 1]
eta = [0.5, 2.5]

##

omega = 1
Amp = 7
input_expr = lambda t: Amp*np.sin(omega*t)/(omega*t)
d_input_expr = lambda t: (Amp/t)*(np.cos(omega*t) - np.sin(omega*t)/(omega*t))
input_torch_lambda = lambda t: Amp*torch.sin(omega*t)/(omega*t)

##

time_array = np.linspace(10**-10, 10*np.pi/omega, 5000).reshape(-1, 1)

strain_array, stress_array = VE_datagen.calculate_strain_stress(input_type, time_array, input_expr, E, eta, D_input_lambda=d_input_expr)

##

# 'normalising'
time_sf = omega/1.2
strain_sf = 1/np.max(abs(strain_array))
stress_sf = 1/np.max(abs(stress_array))
# print(time_sf, strain_sf, stress_sf)

scaled_time_array = time_array*time_sf
scaled_strain_array = strain_array*strain_sf
scaled_stress_array = stress_array*stress_sf
if input_type == 'Strain':
    scaled_input_expr = lambda t: strain_sf*input_expr(t/time_sf)
    scaled_input_torch_lambda = lambda t: strain_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_stress_array
elif input_type == 'Stress':
    scaled_input_expr = lambda t: stress_sf*input_expr(t/time_sf)
    scaled_input_torch_lambda = lambda t: stress_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_strain_array

##

number_of_samples = 1000

reordered_row_indices = np.random.permutation(time_array.size)

reduced_time_array = scaled_time_array[reordered_row_indices, :][:number_of_samples]
reduced_target_array = noisy_target_array[reordered_row_indices, :][:number_of_samples]

##

time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor(reduced_target_array, dtype=torch.float32)






##

import torch.autograd as auto
    
def mech_library(inputs, **library_config):    
    
    prediction, data = inputs
    
    # Load already calculated derivatives of manipulation variable
    input_theta = library_config['input_theta']
    if data.shape[0] == 1: # Swaps real input_theta out for dummy in initialisation pass.
        input_theta = torch.ones((1, input_theta.shape[1]))
    
    # Automatic derivatives of response variable 
    output_derivs = auto_deriv(data, prediction, library_config['diff_order'])
    output_theta = torch.cat((prediction, output_derivs), dim=1)
    
    # Identify the manipulation/response as Stress/Strain and organise into returned variables
    if library_config['input_type'] == 'Strain':
        strain = input_theta
        stress = output_theta
    else: # 'Stress'
        strain = output_theta
        stress = input_theta
        
    strain_t = strain[:, 1:2] # Extract the first time derivative of strain
    strain = torch.cat((strain[:, 0:1], strain[:, 2:]), dim=1) # remove this before it gets put into theta
    strain *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
    theta = torch.cat((strain, stress), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
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

library_diff_order = 3

input_data = scaled_input_torch_lambda(time_tensor)
input_derivs = auto_deriv(time_tensor, input_data, library_diff_order)
input_theta = torch.cat((input_data.detach(), input_derivs.detach()), dim=1)

##

percent = 0.05
def thresh_pc(*args): # Keep as full function so that it can be pickled
    return percent

##

library_config = {'library_func': mech_library,
                  'diff_order': library_diff_order,
                  'coeff_sign': 'positive',
                  'input_type': input_type,
                  'input_theta': input_theta}

network_config = {'hidden_dim': 30}

optim_config = {'lr_coeffs': 0.002,
                'thresh_func': thresh_pc,
                'l1': 10**-6}

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