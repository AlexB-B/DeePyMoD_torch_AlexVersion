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

# scaled_time_array = time_array*time_sf
scaled_strain_array = strain_array*strain_sf
scaled_stress_array = stress_array*stress_sf
if input_type == 'Strain':
#     scaled_input_expr = lambda t: strain_sf*input_expr(t/time_sf)
#     scaled_input_torch_lambda = lambda t: strain_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_stress_array
elif input_type == 'Stress':
#     scaled_input_expr = lambda t: stress_sf*input_expr(t/time_sf)
#     scaled_input_torch_lambda = lambda t: stress_sf*input_torch_lambda(t/time_sf)
    scaled_target_array = scaled_strain_array
    
##

number_of_samples = 1000

reordered_row_indices = np.random.permutation(time_array.size)

reduced_time_array = scaled_time_array[reordered_row_indices, :][:number_of_samples]
reduced_target_array = noisy_target_array[reordered_row_indices, :][:number_of_samples]

##

time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor(reduced_target_array, dtype=torch.float32)