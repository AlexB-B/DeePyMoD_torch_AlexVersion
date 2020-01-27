import torch

from deepymod_torch.neural_net import deepmod_init, train, train_group_mse, train_no_equation, train_optim_NN_only
from deepymod_torch.sparsity import scaling, threshold


def DeepMoD(data, target, network_config, library_config, optim_config, NN=False, coeffs=False):
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

    optim_config_internal = optim_config.copy()
    
    # Initiating
    network, coeff_vector_list, sparsity_mask_list = deepmod_init(network_config, library_config)
    if NN: #Overides network for pretrained network
        network = NN
    if coeffs: #Overides coeffs for specified coeffs
        coeff_vector_list, sparsity_mask_list = coeffs
    
    original_coeff_vector_list = coeff_vector_list.copy()
        
    Final = False
    while True:
        
        if Final:
            # Final Training without L1 and with the accepted sparsity pattern
            print(original_coeff_vector_list, list(coeff_vector_list), list(sparsity_mask_list))
            print('Now running final cycle.')
            
            optim_config_internal['lambda'] = 0
        
        # Training of the network
        time_deriv_list, sparse_theta_list, coeff_vector_list = train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)
        
        if Final:
            break
        
        # Thresholding
        scaled_coeff_vector_list = [scaling(coeff_vector, theta, time_deriv) for coeff_vector, theta, time_deriv in zip(coeff_vector_list, sparse_theta_list, time_deriv_list)]
        #Because of zip, the output variables are all tuples, not lists, but they function the same way
        coeff_vector_list, sparsity_mask_list, Overode_list = zip(*[threshold(scaled_coeff_vector, coeff_vector, sparsity_mask, library_config) for scaled_coeff_vector, coeff_vector, sparsity_mask in zip(scaled_coeff_vector_list, coeff_vector_list, sparsity_mask_list)])
        
        Final = True
        for Overode_Response in Overode_list:
            if Overode_Response:
                Final = False
                print('Overode, reduced library size')
                break

    return list(coeff_vector_list), list(sparsity_mask_list), network


def Not_DeepMoD(data, target, network_config, library_config, optim_config):
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
    network : pytorch sequential model
        The trained neural network.
    '''

    # Initiating
    network, coeff_vector_list, _ = deepmod_init(network_config, library_config)
        
    # Training of the network
    train_group_mse(data, target, network, coeff_vector_list, optim_config)
        
    return network


def DeepMoD_no_equation(data, target, network_config, library_config, optim_config):
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
    network : pytorch sequential model
        The trained neural network.
    '''

    # Initiating
    network, coeff_vector_list, _ = deepmod_init(network_config, library_config)
        
    # Training of the network
    time_deriv, theta = train_no_equation(data, target, network, coeff_vector_list, library_config, optim_config)
    
    index = theta.shape[1] // 2
    full_library = torch.cat((-1*theta[:, 0:1], time_deriv, -1*theta[:, 1:index], theta[:, index:]), axis=1)
    
    return network, full_library.detach()


def DeepMoD_PINN(data, target, network_config, library_config, optim_config, NN=False, coeffs=False):
    '''
    Run DeepMoD as a PINN. The difference is there is no thresholding and selection of terms and no sparsity bias. It's just a straight up simultaneous fit to the data and fit to a given equation. (FYI, this last idea is misleading. I am able to provide the whole equation only in VE case because it is so simple to construct, with an easy to replicate pattern.)
    '''

    optim_config_internal = optim_config.copy()
    
    # Initiating
    network, coeff_vector_list, sparsity_mask_list = deepmod_init(network_config, library_config)
    if NN:
        network = NN
    if coeffs: #Overides coeffs for specified coeffs
        coeff_vector_list, sparsity_mask_list = coeffs
            
    optim_config_internal['lambda'] = 0
        
    # Training of the network
    time_deriv_list, sparse_theta_list, coeff_vector_list = train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)

    return list(coeff_vector_list), list(sparsity_mask_list), network


def DeepMoD_known_eq(data, target, network_config, library_config, optim_config, coeffs, NN=False):
    '''
    coeffs actually not optional for this
    '''

    optim_config_internal = optim_config.copy()
    
    # Initiating
    network, coeff_vector_list, sparsity_mask_list = deepmod_init(network_config, library_config)
    if NN: #Overides network for pretrained network
        network = NN
    coeff_vector_list, sparsity_mask_list = coeffs
    
    original_coeff_vector_list = coeff_vector_list.copy()
            
    optim_config_internal['lambda'] = 0
        
    # Training of the network
    time_deriv_list, sparse_theta_list, coeff_vector_list = train_optim_NN_only(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)
        

    return list(coeff_vector_list), list(sparsity_mask_list), network