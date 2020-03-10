import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from IPython import display

from deepymod_torch.sparsity import scaling
from torch.utils.tensorboard import SummaryWriter
from deepymod_torch.tensorboard import custom_board


def deepmod_init(network_config, library_config):
    '''
    Constructs the neural network, trainable coefficient vectors and sparsity mask.

    Parameters
    ----------
    network_config : dict
        dict containing parameters for network construction. See DeepMoD docstring.
    library_config : dict
        dict containing parameters for library function. See DeepMoD docstring.

    Returns
    -------
    torch_network: pytorch NN sequential module
        The to-be-trained neural network.
    coeff_vector_list: tensor list
        list of coefficient vectors to be optimized, one for each equation.
    sparsity_mask_list: tensor list
        list of sparsity masks, one for each equation.
    '''

    # Building network
    input_dim = network_config['input_dim']
    hidden_dim = network_config['hidden_dim']
    layers = network_config['layers']
    output_dim = network_config['output_dim']

    network = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]  # Input layer

    for hidden_layer in np.arange(layers):  # Hidden layers
        network.append(nn.Linear(hidden_dim, hidden_dim))
        network.append(nn.Tanh())

    network.append(nn.Linear(hidden_dim, output_dim))  # Output layer
    torch_network = nn.Sequential(*network)

    # Building coefficient vectors and sparsity_masks
    library_function = library_config['type']

    sample_data = torch.ones(1, input_dim, requires_grad=True)  # we run a single forward pass on fake data to infer shapes
    sample_prediction = torch_network(sample_data)
    time_deriv_list, theta = library_function(sample_data, sample_prediction, library_config)
    total_terms = theta.shape[1]
    
    coeff_vector_list = [torch.randn((total_terms, 1), dtype=torch.float32, requires_grad=True) for _ in time_deriv_list]
    if library_config.get('coeff_sign', None) == 'positive':
        coeff_vector_list = [abs(tensor_for_output).detach().requires_grad_() for tensor_for_output in coeff_vector_list]
    
    sparsity_mask_list = [torch.arange(total_terms) for _ in time_deriv_list]
    
    return torch_network, coeff_vector_list, sparsity_mask_list


def train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config, plot=False):
    '''
    Trains the deepmod neural network and its coefficient vectors until maximum amount of iterations. Writes diagnostics to
    runs/ directory which can be analyzed with tensorboard.
    
    Parameters
    ----------
    data : Tensor of size (N x M)
        Coordinates corresponding to target data. First column must be time.
    target : Tensor of size (N x L)
        Data the NN is supposed to learn.
    network : pytorch NN sequential module
        Network to be trained.
    coeff_vector_list : tensor list
        List of coefficient vectors to be optimized
    sparsity_mask_list : tensor list
        List of sparsity masks applied to the library function.
    library_config : dict
        Dict containing parameters for the library function. See DeepMoD docstring.
    optim_config : dict
        Dict containing parameters for training. See DeepMoD docstring.
    
    Returns
    -------
    time_deriv_list : tensor list
        list of the time derivatives after training.
    theta : tensor
        library matrix after training.
    coeff_vector_list : tensor list
        list of the trained coefficient vectors.
    '''

    max_iterations = optim_config['max_iterations']
    l1 = optim_config['lambda']
    library_function = library_config['type']
    
    optimizer = torch.optim.Adam(({'params': network.parameters(), 'lr': 0.001}, {'params': coeff_vector_list, 'lr': 0.002}))
 
    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(target, coeff_vector_list))
    
    if plot: # only works for ODEs (one independant variable)
        axes1, axes2 = prep_plot(data, target)

    start_time = time.time()
    
    # Training
    for iteration in np.arange(max_iterations):
        # Calculating prediction and library
        prediction = network(data)
        time_deriv_list, theta = library_function(data, prediction, library_config)
        sparse_theta_list = [theta[:, sparsity_mask] for sparsity_mask in sparsity_mask_list]
        
        # Scaling
        coeff_vector_scaled_list = [scaling(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]
        
        # Calculating PI
        reg_cost_list = torch.stack([torch.mean((time_deriv - sparse_theta @ coeff_vector)**2) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)])
        loss_reg = torch.sum(reg_cost_list)
        
        # Calculating MSE
        MSE_cost_list = torch.mean((prediction - target)**2, dim=0)
        loss_MSE = torch.sum(MSE_cost_list)
        
        # Calculating L1
        l1_cost_list = l1 * torch.stack([torch.sum(torch.abs(coeff_vector_scaled)) for coeff_vector_scaled in coeff_vector_scaled_list])
        loss_l1 = torch.sum(l1_cost_list)

        # Calculating total loss
        loss = loss_MSE + loss_reg + loss_l1

        # Tensorboard stuff
        if iteration % 50 == 0:
            writer.add_scalar('Total loss', loss, iteration)
            for idx in np.arange(len(MSE_cost_list)):
                # Costs
                writer.add_scalar('MSE '+str(idx), MSE_cost_list[idx], iteration)
            
            for idx in np.arange(len(reg_cost_list)):
                writer.add_scalar('Regression '+str(idx), reg_cost_list[idx], iteration)
                writer.add_scalar('L1 '+str(idx), l1_cost_list[idx], iteration)

                # Coefficients
                for element_idx, element in enumerate(torch.unbind(coeff_vector_list[idx])):
                    writer.add_scalar('coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

                # Scaled coefficients
                for element_idx, element in enumerate(torch.unbind(coeff_vector_scaled_list[idx])):
                    writer.add_scalar('scaled_coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

        # Printing
        if iteration % 200 == 0:
            display.clear_output(wait=True)
            
            if plot:
                update_plot(axes1, axes2, data, prediction)
            
            print('Epoch | Total loss | MSE | PI | L1 ')
            print(iteration, "%.1E" % loss.item(), "%.1E" % loss_MSE.item(), "%.1E" % loss_reg.item(), "%.1E" % loss_l1.item())
            for coeff_vector in zip(coeff_vector_list, coeff_vector_scaled_list):
                print(coeff_vector[0])
            
            print('lrs are', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            seconds = time.time() - start_time
            print('Time elapsed:', seconds//60, 'minutes', seconds%60, 'seconds')
            
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.close()
    
    return time_deriv_list, sparse_theta_list, coeff_vector_list


def train_mse(data, target, network, coeff_vector_list, optim_config, plot=False):
    '''
    Trains the deepmod neural network and its coefficient vectors until maximum amount of iterations. Writes diagnostics to
    runs/ directory which can be analyzed with tensorboard.
    
    Parameters
    ----------
    data : Tensor of size (N x M)
        Coordinates corresponding to target data. First column must be time.
    target : Tensor of size (N x L)
        Data the NN is supposed to learn.
    network : pytorch NN sequential module
        Network to be trained.
    optim_config : dict
        Dict containing parameters for training. See DeepMoD docstring.
    '''

    max_iterations = optim_config['mse_only_iterations']

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001) 
    
    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(target, coeff_vector_list))
    
    if plot:
        axes1, axes2 = prep_plot(data, target)
    
    start_time = time.time()
    
    # Training
    for iteration in np.arange(max_iterations):
        # Calculating prediction and library
        prediction = network(data)

        # Calculating MSE
        MSE_cost_list = torch.mean((prediction - target)**2, dim=0)
        loss_MSE = torch.sum(MSE_cost_list)

        # Calculating total loss
        loss = loss_MSE 
        
        # Tensorboard stuff
        if iteration % 50 == 0:
            writer.add_scalar('Total loss', loss, iteration)
            for idx in np.arange(len(MSE_cost_list)):
                # Costs
                writer.add_scalar('MSE '+str(idx), MSE_cost_list[idx], iteration)
                #writer.add_scalar('L1 '+str(idx), l1_cost_list[idx], iteration)

        if iteration % 500 == 0:
            display.clear_output(wait=True)
            
            if plot:
                update_plot(axes1, axes2, data, prediction)
            
            print('Epoch | MSE loss ')
            print(iteration, "%.1E" % loss.item())
            print('lr is', optimizer.param_groups[0]['lr'])
            seconds = time.time() - start_time
            print('Time elapsed:', seconds//60, 'minutes', seconds%60, 'seconds')

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    writer.close()

    return


def prep_plot(data, target):
    
    number_graphs = target.shape[1]
    fig, axes1 = plt.subplots(ncols=number_graphs, squeeze=False, figsize=(6.4*number_graphs, 4.8)) # 6.4 and 4.8 are the default graph plot dimensions
    axes1 = axes1.flatten()
    plt.title('Current prediction ability of network')
    colour = 'blue'
    axes2 = np.array([])
    for col, ax1 in enumerate(axes1):
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Target', color=colour)
        ax1.plot(data.detach(), target[:, col], color=colour, linestyle='None', marker='.', markersize=1)
        ax1.tick_params(axis='y', labelcolor=colour)
        axes2 = np.append(axes2, ax1.twinx())
        axes2[col].tick_params(axis='y', labelcolor='red')
    
    return axes1, axes2


def update_plot(axes1, axes2, data, prediction):
    
    for col, ax2 in enumerate(axes2):
        ax2.clear()
        ax2.set_ylabel('Prediction', color='red')
        ax2.plot(data.detach(), prediction[:, col].detach(), color='red', linestyle='None', marker='.', markersize=1)
        ax2.set_ylim(axes1[col].get_ylim())
        
    plt.tight_layout()
    display.display(plt.gcf())