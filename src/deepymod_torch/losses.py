import torch
from deepymod_torch.sparsity import scaling # Why here?

import torch.nn as nn


def reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list):
    '''Loss function for the regularisation loss. Calculates loss for each term in list.'''
    loss = torch.stack([torch.mean((time_deriv - theta @ coeff_vector)**2) for time_deriv, theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)])
    return loss

def mse_loss(prediction, target):
    '''Loss functions for the MSE loss. Calculates loss for each term in list.'''
    loss = torch.mean((prediction - target)**2, dim=0)
    return loss

def l1_loss(scaled_coeff_vector_list, l1):
    '''Loss functions for the L1 loss on the coefficients. Calculates loss for each term in list.'''
    loss = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in scaled_coeff_vector_list])
    return l1 * loss
    
def sign_loss(scaled_coeff_vector_list, configs):
    kappa = configs.optim['kappa']
    sign = configs.library.get('coeff_sign', 1)
    '''Loss functions for the sign loss on the coefficients. Calculates loss for each term in list.'''
    loss = torch.stack([torch.sum(nn.functional.relu(-sign*coeff_vector))**2 for coeff_vector in scaled_coeff_vector_list])        
    return kappa * loss


 
  
