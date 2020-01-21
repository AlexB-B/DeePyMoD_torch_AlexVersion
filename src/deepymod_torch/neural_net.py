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
    _, theta = library_function(sample_data, sample_prediction, library_config)
    total_terms = theta.shape[1]
    
    coeff_vector_list = [torch.randn((total_terms, 1), dtype=torch.float32, requires_grad=True) for _ in torch.arange(output_dim)]
    if library_config.get('coeff_sign', None) == 'positive':
        coeff_vector_list = [abs(tensor_for_output).detach().requires_grad_() for tensor_for_output in coeff_vector_list]
    
    sparsity_mask_list = [torch.arange(total_terms) for _ in torch.arange(output_dim)]

    return torch_network, coeff_vector_list, sparsity_mask_list


def train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config):
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
    
    optimizer_NN = torch.optim.Adam(network.parameters(), lr=0.01)
    optimizer_coeffs = torch.optim.Adam(coeff_vector_list, lr=0.001)
    scheduler_NN = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NN, factor=0.5, patience=100, cooldown=50)
    # I am worried that the lr for the second parameter group (coeff_vector_list) starts off too high. But i do not want the scheduler to make it small while the MSE has not yet been reduced. It may be necessary to make 2 seperate optimisers if I want 2 seperate schedulers....
    
    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(coeff_vector_list))

    # preparing plot
    fig, ax1 = plt.subplots()
    plt.title('Current prediction ability of network')
    ax1.set_xlabel('Time (s)')
    colour = 'blue'
    ax1.set_ylabel('Target', color=colour)
    ax1.plot(data.detach(), target, color=colour, linestyle='None', marker='.', markersize=1)
    ax1.tick_params(axis='y', labelcolor=colour)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
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
        l1_cost_list = torch.stack([torch.sum(torch.abs(coeff_vector_scaled)) for coeff_vector_scaled in coeff_vector_scaled_list])
        loss_l1 = l1 * torch.sum(l1_cost_list)

        # Calculating total loss
        loss = loss_MSE + loss_reg + loss_l1

        # Optimizer step
        optimizer_NN.zero_grad()
        optimizer_coeffs.zero_grad()
        loss.backward()
        optimizer_NN.step()
        optimizer_coeffs.step()
        scheduler_NN.step(loss)

        # Tensorboard stuff
        if iteration % 50 == 0:
            writer.add_scalar('Total loss', loss, iteration)
            for idx in np.arange(len(MSE_cost_list)):
                # Costs
                writer.add_scalar('MSE '+str(idx), MSE_cost_list[idx], iteration)
                writer.add_scalar('Regression '+str(idx), reg_cost_list[idx], iteration)
                writer.add_scalar('L1 '+str(idx), l1_cost_list[idx], iteration)

                # Coefficients
                for element_idx, element in enumerate(torch.unbind(coeff_vector_list[idx])):
                    writer.add_scalar('coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

                # Scaled coefficients
                for element_idx, element in enumerate(torch.unbind(coeff_vector_scaled_list[idx])):
                    writer.add_scalar('scaled_coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

        # Printing
        if iteration % 50 == 0:
            display.clear_output(wait=True)
            
            #Update plot
            '''
            if iteration == 0:
                ax3 = ax1.twinx()
                ax3.plot(data.detach(), -theta[:, 0].detach(), color='green', linestyle='None', marker='.', markersize=1)
            '''
            
            ax2.clear()
            ax2.set_ylabel('Prediction', color='red')
            ax2.plot(data.detach(), prediction.detach(), color='red', linestyle='None', marker='.', markersize=1)
            ax2.set_ylim(ax1.get_ylim())
            display.display(plt.gcf())
            
            print('Epoch | Total loss | MSE | PI | L1 ')
            print(iteration, "%.1E" % loss.item(), "%.1E" % loss_MSE.item(), "%.1E" % loss_reg.item(), "%.1E" % loss_l1.item())
            for coeff_vector in zip(coeff_vector_list, coeff_vector_scaled_list):
                print(coeff_vector[0])
            
            print('lrs are', optimizer_NN.param_groups[0]['lr'], optimizer_coeffs.param_groups[0]['lr'])
            seconds = time.time() - start_time
            print('Time elapsed:', seconds//60, 'minutes', seconds%60, 'seconds')

    writer.close()
    
    display.clear_output(wait=True)
    print('Epoch | Total loss | MSE | PI | L1 ')
    print(iteration, "%.1E" % loss.item(), "%.1E" % loss_MSE.item(), "%.1E" % loss_reg.item(), "%.1E" % loss_l1.item())
    for coeff_vector in zip(coeff_vector_list, coeff_vector_scaled_list):
        print(coeff_vector[0])
            
    print('lrs are', optimizer_NN.param_groups[0]['lr'], optimizer_coeffs.param_groups[0]['lr'])
    print('Total time elapsed:', seconds//60, 'minutes', seconds%60, 'seconds')
            
    return time_deriv_list, sparse_theta_list, coeff_vector_list


def train_group_mse(data, target, network, coeff_vector_list, optim_config):
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

    max_iterations = optim_config['max_iterations']

    optimizer = torch.optim.Adam(network.parameters(), lr=0.01) # Do not start with a lr as large as 0.1. 
    #loss_threshold = 10**-3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=0.0001, cooldown=50, eps=1e-08)
    '''
    an explanation of this scheduler
    
    What this is doing is it will reduce the learning rate if the improvement to the loss appears to be stagnating. This is to get the best of both worlds between a high learning rate (fast to remove early high loss and able to transition between different minima) and a low learning rate (able to find the optimum position with a minimum well)
    
    I have left lots of parameters explicitely stated, even if some are default, for transparency.
    factor=0.5 (not default) means when the lr is adjusted, it is halved
    patience=50 (not default) and threshold=0.0001 (default) must be discussed together. As the loss decreases the best (probably most recent at first) loss is remembered. As the improvement to loss slows, it may get to a point where, after a certain loss, none of the next patience=50 epochs achieve a loss better than this loss*(1-threshold=0.0001). Once this improvement has slowed down to this extent, a lr adjustment occurs. Essentially, the network as 50 epochs to reduce loss by 0.01% at some point, or else. This is only considered after the cooldown phase. Both increasing patience or decreasing threshold mean we are asking the loss improvment to have slowed down to a greater extent before adjustment.
    cooldown=50 (not default) means after a lr adjustment, the network is allowed to settle for 50 epochs before another adjustment is considered. This is to allow the network to have a small lag phase before it improves without adjustments being too aggressive, and also just reduces the frequency of adjustment.
    eps=1e-08 (default) means that very fine adjustments to lr are not bothered with. Considering the initial lr and factor, this amounts to a minimum of deterministic magnitude.
    '''
    
    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(coeff_vector_list))
    
    #preparing plot
    fig, ax1 = plt.subplots()

    plt.title('Current prediction ability of network')
    
    ax1.set_xlabel('Time (s)')

    colour = 'tab:blue'
    ax1.set_ylabel('target', color=colour)
    ax1.plot(data.detach(), target, color=colour)
    ax1.tick_params(axis='y', labelcolor=colour)
    
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Training
    for iteration in np.arange(max_iterations):
        # Calculating prediction and library
        prediction = network(data)

        # Calculating MSE
        MSE_cost_list = torch.mean((prediction - target)**2, dim=0)
        loss_MSE = torch.sum(MSE_cost_list)

        # Calculating total loss
        loss = loss_MSE 

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        '''
        if loss < loss_threshold:
            optimizer.param_groups[0]['lr'] *= 0.1
            loss_threshold *= 0.1
            #Notice the [0]. This is because if the optimiser has many param groups (in DeepMoD it has 2, 1 for network params and 1 for coeff_vector) so the
            #attribute called optimiser.param_groups is first a list, which contains dictionaries.
        '''
        
        # Tensorboard stuff
        if iteration % 50 == 0:
            writer.add_scalar('Total loss', loss, iteration)
            for idx in np.arange(len(MSE_cost_list)):
                # Costs
                writer.add_scalar('MSE '+str(idx), MSE_cost_list[idx], iteration)
                #writer.add_scalar('L1 '+str(idx), l1_cost_list[idx], iteration)

        if iteration % 200 == 0:
            display.clear_output(wait=True)
            
            #Update plot
            ax2.clear()
            ax2.set_ylabel('prediction', color='tab:red')
            ax2.plot(data.detach(), prediction.detach(), color='tab:red')
            display.display(plt.gcf())
            
            print('Epoch | MSE loss ')
            print(iteration, "%.1E" % loss.item())
            print('lr is ', optimizer.param_groups[0]['lr'])

    writer.close()
    display.clear_output(wait=False)
    print('Epoch | MSE loss ')
    print(iteration, "%.1E" % loss.item())
    print('lr is ', optimizer.param_groups[0]['lr'])
    return

