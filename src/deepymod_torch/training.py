import torch
import time

from deepymod_torch.output import Tensorboard, progress
from deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from deepymod_torch.sparsity import scaling, threshold

import IPython.display as dis
from deepymod_torch.output import prep_plot, update_plot
from deepymod_torch.losses import na_loss


def train(model, data, target, optimizer, configs):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_tar_vars = target.shape[1]
    number_of_terms_list = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_tar_vars, number_of_terms_list)
    
    max_iterations = configs.optim['max_iterations']
    
    plot = configs.report['plot']
    print_interval = configs.report['print_interval']
    if plot: # only works for ODEs (one independant variable)
        axes = prep_plot(data, target)
    
    # Training
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction, target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, configs.optim['l1'])
        loss_na = na_loss(coeff_vector_scaled_list, configs.optim['kappa'])
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1) + torch.sum(loss_na)
        
        # Writing, first live progress, then tensorboard.
        if iteration % print_interval == 0:
            dis.clear_output(wait=True)
            if plot:
                update_plot(axes, data, prediction)
            
            print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |      NA |')
            progress(iteration.item(), start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item(), torch.sum(loss_na).item())
            print()
            print(list(coeff_vector_list))
#             print(sparsity_mask_list)
        
        if iteration % 100 == 0:
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, loss_na, coeff_vector_list, coeff_vector_scaled_list)
            
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_mse(model, data, target, optimizer, configs):
    '''Trains the deepmod model only on the MSE. Updates model in-place.'''
    start_time = time.time()
    number_of_tar_vars = target.shape[1]
    number_of_terms_list = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_tar_vars, number_of_terms_list)
    
    max_iterations = configs.optim['mse_only_iterations']
    
    plot = configs.report['plot']
    print_interval = configs.report['print_interval']
    if plot: # only works for ODEs (one independant variable)
        axes = prep_plot(data, target)
    
    # Training
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 

        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss = torch.sum(loss_mse)

        # Writing, first live progress, then tensorboard.
        if iteration % print_interval == 0:
            dis.clear_output(wait=True)
            if plot:
                update_plot(axes, data, prediction)
                
            print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |      NA |')
            progress(iteration.item(), start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), 0, 0, 0)
        
        if iteration % 100 == 0:
            board.write(iteration, loss, loss_mse, [0], [0], [0], coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_deepmod(model, data, target, optimizer, configs):
    '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
    
    if not configs.optim['PINN']:
        # Train first cycle and get prediction
        train(model, data, target, optimizer, configs)
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)

        model.fit.coeff_vector_history += [model.fit.coeff_vector]
        model.fit.sparsity_mask_history += [model.fit.sparsity_mask]

        # Threshold, set sparsity mask and coeff vector
        sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list, configs.optim)
        model.fit.sparsity_mask = sparsity_mask_list
        model.fit.coeff_vector = torch.nn.ParameterList(sparse_coeff_vector_list)

        #Resetting optimizer for different shapes, train without l1 
    #     optimizer.param_groups[0]['params'] = model.parameters() # packs nn and coeff_vector into same optimiser group ...
        # but doesn't remove now duplicate coeff_vector params? I would propose to replace with the below...
        optimizer.param_groups[1]['params'] = model.fit.coeff_vector.parameters()
        # or add line ...
        # del optimizer.param_groups[1]
        
        configs.optim['max_iterations'] = configs.optim['final_run_iterations']
    
    #print() #empty line for correct printing
    external_values = configs.optim['l1'], configs.optim['max_iterations']
    configs.optim['l1'] = 0.0
    train(model, data, target, optimizer, configs)
    configs.optim['l1'], configs.optim['max_iterations'] = external_values
    
    model.fit.coeff_vector_history += [model.fit.coeff_vector]
    model.fit.sparsity_mask_history += [model.fit.sparsity_mask]