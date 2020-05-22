import torch
import torch.nn as nn
from deepymod_torch.network import Fitting, Library

import numpy as np
import deepymod_torch.training as training

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
        training.train_mse(model, data, target, optimizer, configs)
        prediction, time_deriv_list, sparse_theta_list = model.forward(data)[:3]
        lstsq_guess_list = [np.linalg.lstsq(sparse_theta.detach(), time_deriv.detach(), rcond=None)[0] for sparse_theta, time_deriv in zip(sparse_theta_list, time_deriv_list)]
    
    model.fit.initial_guess = None
    if configs.optim['use_lstsq_approx']:
        model.fit.initial_guess = lstsq_guess_list
        model.fit.coeff_vector = nn.ParameterList([nn.Parameter(torch.tensor(lstsq_guess, dtype=torch.float32)) for lstsq_guess in lstsq_guess_list])
    
    optimizer = torch.optim.Adam(({'params': model.network.parameters(), 'lr': configs.optim['lr_nn']}, {'params': model.fit.coeff_vector.parameters(), 'lr': configs.optim['lr_coeffs']}), betas=configs.optim['betas'], amsgrad=configs.optim['amsgrad'])
    
    training.train_deepmod(model, data, target, optimizer, configs)
        
    return model
    
    
class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, n_in, hidden_dims, n_out, library_function, configs):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, configs.library)
        self.fit = self.build_fit_layer(n_in, configs.library)
        self.configs = configs
        
    def forward(self, input):
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.fit(theta)
        return prediction, time_deriv, sparse_theta, coeff_vector

    def build_network(self, n_in, hidden_dims, n_out):
        # NN
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        return network

    def build_fit_layer(self, n_in, library_config):
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        time_deriv_list, theta = self.library((self.network(sample_input), sample_input))
        n_equations = len(time_deriv_list)
        n_terms = theta.shape[1] # do sample pass to infer shapes
        fit_layer = Fitting(n_equations, n_terms, library_config)

        return fit_layer

    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()
        
        
class Configuration():
    def __init__(self, library, network={}, optim={}, report={}):
        self.library = library
        self.network = network
        self.optim = optim
        self.report = report
        self.add_defaults()
        
    def add_defaults(self):
        if 'pre_trained_network' not in self.network:
            self.network['pre_trained_network'] = None

        if 'hidden_dim' not in self.network:
            self.network['hidden_dim'] = 50

        if 'layers' not in self.network:
            self.network['layers'] = 4

        if 'PINN' not in self.optim:
            self.optim['PINN'] = False

        if 'l1' not in self.optim:
            self.optim['l1'] = 10**-5

        if 'kappa' not in self.optim:
            self.optim['kappa'] = 0 # Not used by default
            if 'coeff_sign' in self.library:
                self.optim['kappa'] = 1
        
        if 'lr_nn' not in self.optim:
            self.optim['lr_nn'] = 0.001 # default is default for optimizer
        
        if 'lr_coeffs' not in self.optim:
            self.optim['lr_coeffs'] = 0.001 # default is default for optimizer

        if 'betas' not in self.optim:
            self.optim['betas'] = (0.9, 0.999) # default is default for optimizer

        if 'amsgrad' not in self.optim:
            self.optim['amsgrad'] = False # default is default for optimizer
            
        if 'mse_only_iterations' not in self.optim:
            self.optim['mse_only_iterations'] = None

        if 'max_iterations' not in self.optim:
            self.optim['max_iterations'] = 100001

        if 'final_run_iterations' not in self.optim:
            self.optim['final_run_iterations'] = 10001

        if 'use_lstsq_approx' not in self.optim:
            self.optim['use_lstsq_approx'] = False

        if 'thresh_func' not in self.optim:
            self.optim['thresh_func'] = lambda coeff_vector_scaled, *args: torch.std(coeff_vector_scaled, dim=0)
            
        if 'print_interval' not in self.report:
            self.report['print_interval'] = 1000
            
        if 'plot' not in self.report:
            self.report['plot'] = False
            
        if 'coeff_sign' in self.library:
            convert_dict = {'positive': 1, 1: 1, 'negative': -1, -1: -1}
            self.library['coeff_sign'] = convert_dict[self.library['coeff_sign']]