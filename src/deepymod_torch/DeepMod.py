from deepymod_torch.neural_net import deepmod_init, train, train_mse
from deepymod_torch.sparsity import scaling, threshold

import numpy as np
import torch

def DeepMoD(data, target, library_config, network_config={}, optim_config={}, print_interval=1000, plot=False):
    '''
    Runs the deepmod algorithm on the supplied dataset. Mostly a convenience function and can be used as
    a basis for more complex training means. First column of data must correspond to time coordinates, spatial coordinates
    are second column and up. Diagnostics are written to runs/ directory, can be analyzed with tensorboard.

    Parameters
    ----------
    data : Tensor of size (N x M)
        Coordinates corresponding to target data. First column must be time.
    target : Tensor of size (N x L)
        Data the NN is supposed to learn.
    network_config : dict
        Dict containing parameters for the network: {'input_dim': , 'hidden_dim': , 'layers': ,'output_dim':}
            input_dim : number of input neurons, should be same as second dimension of data.
            hiddem_dim : number of neurons in each hidden layer.
            layers : number of hidden layers.
            output_dim : number of output neurons, should be same as second dimension of data.
    library_config : dict
        Dict containing parameters for the library function: {'type': , **kwargs}
            type : library function to be used.
            kwargs : arguments to be used in the selected library function.
    optim_config : dict
        Dict containing parameters for training: {'lambda': , 'max_iterations':}
            lambda : value of l1 constant.
            max_iterations : maximum number of iterations used for training.
    Returns
    -------
    coeff_vector_list
        List of tensors containing remaining values of weight vector.
    sparsity_mask_list
        List of tensors corresponding to the maintained components of the coefficient vectors. Each list entry is one equation.
    network : pytorch sequential model
        The trained neural network.
    '''
    
    complete_configs(network_config, optim_config)
    
    # Pull config params, using defaults where necessary.
    pre_trained_network = network_config['pre_trained_network']
    mse_only_iterations = optim_config['mse_only_iterations']
    max_iterations = optim_config['max_iterations']
    final_run_iterations = optim_config['final_run_iterations']
    use_lstsq_approx = optim_config['use_lstsq_approx']
    
    optim_config_internal = optim_config.copy()
    
    # Initiating
    network, coeff_vector_list, sparsity_mask_list = deepmod_init(data, target, network_config, library_config)
    if pre_trained_network: # Overides network for pretrained network
        network = pre_trained_network
    
    original_coeff_vector_list = coeff_vector_list.copy()
    coeff_vector_list_each_iteration = []
    sparsity_mask_list_each_iteration = []
    scaled_coeff_vector_list_each_iteration = []
    
    lstsq_guess_list = []
    # Initial training to just minimise MSE. coeff_vector_list only necessary for housekeeping.
    if mse_only_iterations:
        print('Training MSE only')
        train_mse(data, target, network, coeff_vector_list, optim_config_internal, print_interval=print_interval, plot=plot)
        # Make initial guess at coeffs using least squares.
        # Nothing is done with this, save for returning the result, unless the next if statement is satisfied.
        # Note, this will definitely not produce much sense if fitting was not complete after train_mse finishes.
        prediction = network(data)
        time_deriv_list, theta = library_config['type'](data, prediction, library_config)
        lstsq_guess_list = [np.linalg.lstsq(theta.detach(), time_deriv.detach(), rcond=None)[0] for time_deriv in time_deriv_list]
    
    if use_lstsq_approx:
        coeff_vector_list = [torch.tensor(lstsq_guess, dtype=torch.float32, requires_grad=True) for lstsq_guess in lstsq_guess_list]
    
    Final = False
    while True:
        
        if Final:
            # Final Training without L1 and with the accepted sparsity pattern
            print(original_coeff_vector_list, list(coeff_vector_list), list(sparsity_mask_list))
            print('Now running final cycle.')
            
            optim_config_internal['lambda'] = 0
            optim_config_internal['max_iterations'] = final_run_iterations
        else:
            print('Running full training')
        
        # Training of the network
        time_deriv_list, sparse_theta_list, coeff_vector_list = train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal, print_interval=print_interval, plot=plot)
        
        # Thresholding
        scaled_coeff_vector_list = [scaling(coeff_vector, theta, time_deriv) for coeff_vector, theta, time_deriv in zip(coeff_vector_list, sparse_theta_list, time_deriv_list)]
        
        coeff_vector_list_each_iteration += [list(coeff_vector_list)]
        scaled_coeff_vector_list_each_iteration += [scaled_coeff_vector_list]
        sparsity_mask_list_each_iteration += [list(sparsity_mask_list)]
        
        if Final:
            break
        
        #Because of zip, the output variables are all tuples, not lists, but they function the same way
        coeff_vector_list, sparsity_mask_list = zip(*[threshold(scaled_coeff_vector, coeff_vector, sparsity_mask, optim_config_internal, library_config) for scaled_coeff_vector, coeff_vector, sparsity_mask in zip(scaled_coeff_vector_list, coeff_vector_list, sparsity_mask_list)])
        
        Final = True
    
    coeff_info_tuple = (coeff_vector_list_each_iteration, scaled_coeff_vector_list_each_iteration, sparsity_mask_list_each_iteration)
    
    return coeff_info_tuple, lstsq_guess_list, network


def complete_configs(network_config, optim_config):
    
    if 'pre_trained_network' not in network_config:
        network_config['pre_trained_network'] = None
    
    if 'hidden_dim' not in network_config:
        network_config['hidden_dim'] = 50
        
    if 'layers' not in network_config:
        network_config['layers'] = 4
    
    if 'lambda' not in optim_config:
        optim_config['lambda'] = 10**-5
    
    if 'kappa' not in optim_config:
        optim_config['kappa'] = 1 # Even if 1, may not be used depending on library_config
    
    if 'lr_coeffs' not in optim_config:
        optim_config['lr_coeffs'] = 0.001 # default is default for optimizer
    
    if 'betas_coeffs' not in optim_config:
        optim_config['betas_coeffs'] = (0.9, 0.999) # default is default for optimizer
    
    if 'mse_only_iterations' not in optim_config:
        optim_config['mse_only_iterations'] = None
    
    if 'max_iterations' not in optim_config:
        optim_config['max_iterations'] = 100001
    
    if 'final_run_iterations' not in optim_config:
        optim_config['final_run_iterations'] = 10001
        
    if 'use_lstsq_approx' not in optim_config:
        optim_config['use_lstsq_approx'] = False
        
    if 'thresh_func' not in optim_config:
        optim_config['thresh_func'] = lambda scaled_coeff_vector, *args: torch.std(scaled_coeff_vector, dim=0)