import torch
import os
import numpy as np
from datetime import datetime

from deepymod_torch.neural_net import deepmod_init, train, train_group_mse, train_no_equation, train_optim_NN_only, train_least_squares
from deepymod_torch.sparsity import scaling, threshold
from deepymod_torch.library_function import stress_input_library
import deepymod_torch.VE_datagen as VE_datagen
import deepymod_torch.VE_params as VE_params


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
    coeff_vector_list_each_iteration = []
    sparsity_mask_list_each_iteration = []
    scaled_coeff_vector_list_each_iteration = []
        
    Final = False
    while True:
        
        if Final:
            # Final Training without L1 and with the accepted sparsity pattern
            print(original_coeff_vector_list, list(coeff_vector_list), list(sparsity_mask_list))
            print('Now running final cycle.')
            
            optim_config_internal['lambda'] = 0
            optim_config_internal['max_iterations'] = optim_config['final_run_iterations']
        
        # Training of the network
        time_deriv_list, sparse_theta_list, coeff_vector_list = train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)
        
        # Thresholding
        scaled_coeff_vector_list = [scaling(coeff_vector, theta, time_deriv) for coeff_vector, theta, time_deriv in zip(coeff_vector_list, sparse_theta_list, time_deriv_list)]
        
        coeff_vector_list_each_iteration += [list(coeff_vector_list)]
        scaled_coeff_vector_list_each_iteration += [scaled_coeff_vector_list]
        sparsity_mask_list_each_iteration += [list(sparsity_mask_list)]
        
        if Final:
            break
        
        #Because of zip, the output variables are all tuples, not lists, but they function the same way
        coeff_vector_list, sparsity_mask_list, Overode_list = zip(*[threshold(scaled_coeff_vector, coeff_vector, sparsity_mask, library_config) for scaled_coeff_vector, coeff_vector, sparsity_mask in zip(scaled_coeff_vector_list, coeff_vector_list, sparsity_mask_list)])
        
        Final = True
        for Overode_Response in Overode_list:
            if Overode_Response:
                Final = False
                print('Overode, reduced library size')
                break

    return coeff_vector_list_each_iteration, scaled_coeff_vector_list_each_iteration, sparsity_mask_list_each_iteration, network


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


def DeepMoD_least_squares(data, target, library_config, optim_config):
    '''
    Runs a version of the deepmod algorithm that has no NN! Just a training optimisation process for the coeffs given an equation with teh correct terms.
    '''
    
    total_terms = library_config['diff_order']*2 + 1
    
    # Initiating
    coeff_vector_list = [torch.randn((total_terms, 1), dtype=torch.float32, requires_grad=True)]
    if library_config.get('coeff_sign', None) == 'positive':
        coeff_vector_list = [abs(coeff_vector_list[0]).detach().requires_grad_()]
    
    sparsity_mask_list = [torch.arange(total_terms)]
        
    # Training of the network
    time_deriv_list, sparse_theta_list, coeff_vector_list = train_least_squares(data, target, coeff_vector_list, sparsity_mask_list, library_config, optim_config)
        

    return list(coeff_vector_list), list(sparsity_mask_list)


def iterate_2nd_decay_constant(decay_constant_values):
    
    omega = 1
    E = [1, 1, 1]
    eta = [2.5, 1]
    input_expr = lambda t: np.sin(t)/(t)
    dsigma = lambda t: (1/t)*(np.cos(t) - np.sin(t)/(t))
    input_torch_lambda = lambda t: torch.sin(t)/(t)
    input_type = 'Stress'
    func_desc = 'Sinc'
    
    time_array = np.linspace(0.00001, 30, 5000)
    
    for tau_2 in decay_constant_values:
        
        np.random.seed(0)
        torch.manual_seed(0)
        
        # Data generation and DeepMoD prep
        eta[1] = tau_2*E[2]
        
        strain_array, stress_array = VE_datagen.calculate_strain_stress(input_type, time_array, input_expr, E, eta, D_input_lambda=dsigma)
    
        time_array = time_array.reshape(-1, 1)
        strain_array = strain_array.reshape(-1, 1)

        # Kept just for future, could remove entirely
        noise_level = 0
        noisy_strain_array = strain_array + noise_level * np.std(strain_array) * np.random.standard_normal(strain_array.shape)
        
        number_of_samples = 1000
        reordered_row_indices = np.random.permutation(time_array.size)
        reduced_time_array = time_array[reordered_row_indices, :][:number_of_samples]
        reduced_strain_array = noisy_strain_array[reordered_row_indices, :][:number_of_samples]
        
        time_tensor = torch.tensor(reduced_time_array, dtype=torch.float32, requires_grad=True)
        strain_tensor = torch.tensor(reduced_strain_array, dtype=torch.float32)
        
        optim_config = {'lambda': 10**-6, 'max_iterations': 40001, 'final_run_iterations': 5001}
        network_config = {'input_dim': 1, 'hidden_dim': 40, 'layers': 5, 'output_dim': 1}
        lib_config = {'type': stress_input_library, 'diff_order': 3, 'coeff_sign': 'positive', 'input_type': input_type, 'input_expr': input_torch_lambda}
        
        now = datetime.now()
        dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
        
        # DeepMoD
        sparse_coeff_vector_list_list, scaled_coeff_vector_list_list, sparsity_mask_list_list, network = DeepMoD(time_tensor, strain_tensor, network_config, lib_config, optim_config)
        
        
        # Organising results and calculating errors
        expected_coeffs = VE_params.coeffs_from_model_params(E, eta)
        
        investigated_param = 'Decay Constant 2'
        param_value = tau_2
        #repeat_instance = 99
        
        stress_array = stress_array.reshape(-1,1)
        reduced_stress_array = stress_array[reordered_row_indices, :][:number_of_samples]
        prediction_array = np.array(network(time_tensor).detach())
        
        target_coeffs_array = np.array(expected_coeffs).reshape(-1,1)
        pre_thresh_coeffs_array = np.array(sparse_coeff_vector_list_list[0][0].detach())
        pre_thresh_scaled_coeffs_array = np.array(scaled_coeff_vector_list_list[0][0].detach())
        final_coeffs_array = np.array(sparse_coeff_vector_list_list[-1][0].detach())
        
        good_dims = False
        if target_coeffs_array.shape == final_coeffs_array.shape:
            #coeffs_ME = np.sum(abs(target_coeffs_array - final_coeffs_array))/len(not_floats)
            #coeffs_ME = np.array(coeffs_ME).reshape(1)
            sparsity_mask_array = np.array(sparsity_mask_list_list[-1][0]).reshape(-1,1)
            final_scaled_coeffs_array = np.array(scaled_coeff_vector_list_list[-1][0].detach())
            good_dims = True
            
        series_data = np.concatenate((reduced_time_array, reduced_strain_array, reduced_stress_array, prediction_array), axis=1)
        pre_thresh_coeffs_data = np.concatenate((pre_thresh_coeffs_array, pre_thresh_scaled_coeffs_array), axis=1)
        if good_dims:
            coeffs_data = np.concatenate((target_coeffs_array, final_coeffs_array, final_scaled_coeffs_array, sparsity_mask_array), axis=1)
            
        DG_info_list = [str(omega), str(E), str(eta), input_type, func_desc]
        misc_list = [dt_string, investigated_param, str(param_value)]#, str(repeat_instance), success_state]
        
        
        # Saving results
        first_subfolder = investigated_param
        second_subfolder = 'param_' + str(param_value).replace('.', '-')
        #third_subfolder = 'repeat_' + str(repeat_instance)
        parent_folder = '../data/Results_tau2_testing'
        foldername = parent_folder + '/' + first_subfolder + '/' + second_subfolder# + '/' + third_subfolder
        
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        
        np.savetxt(foldername+'/series_data.csv', series_data, delimiter=',', header='Time, Target_Strain, Stress, Prediction_Strain')
        np.savetxt(foldername+'/pre_thresh_coeffs_data.csv', pre_thresh_coeffs_data, delimiter=',', header='Prediction_Coeffs, Scaled_Prediction_Coeffs')
        
        if good_dims:
            np.savetxt(foldername+'/coeffs_data.csv', coeffs_data, delimiter=',', header='Target_Coeffs, Prediction_Coeffs, Scaled_Prediction_Coeffs, Sparsity_Mask')
            #np.savetxt(foldername+'/error.csv', coeffs_ME, delimiter=',')
            
        with open(foldername+'/misc_list.txt', 'w') as file:
            file.writelines("%s\n" % line for line in misc_list)
            
        with open(foldername+'/DG_info_list.txt', 'w') as file:
            file.writelines("%s\n" % line for line in DG_info_list)