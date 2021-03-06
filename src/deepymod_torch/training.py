import torch
import time

from deepymod_torch.output import Tensorboard, progress
from deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from deepymod_torch.sparsity import scaling, threshold

import IPython.display as dis
from deepymod_torch.output import prep_plot, update_plot
from deepymod_torch.losses import sign_loss


def train(model, data, target, optimizer, *args):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    configs = model.configs
    if len(args) > 0: # backwards compatibility
        configs.optim['max_iterations'], configs.optim['l1'] = args[0], args[1]['l1']
    start_time = time.time()
    number_of_tar_vars = target.shape[1]
    number_of_terms_list = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_tar_vars, number_of_terms_list)
    
    max_iterations = configs.optim['max_iterations']
    
    plot = configs.report['plot']
    print_interval = configs.report['print_interval']
    if plot: # only works for ODEs (one independant variable)
        axes = prep_plot(data, target)
    
    print(optimizer.param_groups)
    
    # Training
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction, target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, configs.optim['l1'])
        loss_sign = sign_loss(coeff_vector_scaled_list, configs)
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1) + torch.sum(loss_sign)
        
        # Writing, first live progress, then tensorboard.
        if iteration % print_interval == 0:
            dis.clear_output(wait=True)
            if plot:
                update_plot(axes, data, prediction)
            
            print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |     Sign |')
            progress(iteration.item(), start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item(), torch.sum(loss_sign).item())
            print()
            print(list(coeff_vector_list))
#             print(sparsity_mask_list)
        
        if iteration % 100 == 0:
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, loss_sign, coeff_vector_list, coeff_vector_scaled_list)
            
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_mse(model, data, target, optimizer, *args):
    '''Trains the deepmod model only on the MSE. Updates model in-place.'''
    configs = model.configs
    if len(args) > 0: # backwards compatibility
        configs.optim['max_iterations'] = args[0]
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
                
            print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |     Sign |')
            progress(iteration.item(), start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), 0, 0, 0)
        
        if iteration % 100 == 0:
            board.write(iteration, loss, loss_mse, [0], [0], [0], coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()
    
def train_deepmod(model, data, target, optimizer, *args):
    '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
    
    configs = model.configs
    if len(args) > 0: # backwards compatibility
        configs.optim['max_iterations'], configs.optim['final_run_iterations'] = args[0], args[0]
        configs.optim['l1'] = args[1]['l1']
    external_values = configs.optim['l1'], configs.optim['max_iterations']
    
    if not configs.optim['PINN']:
        # Train first cycle and get prediction
        train(model, data, target, optimizer)
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)

        model.fit.coeff_vector_history += [model.fit.coeff_vector]
        model.fit.sparsity_mask_history += [model.fit.sparsity_mask]

        # Threshold, set sparsity mask and coeff vector
        sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list, configs.optim)
        model.fit.sparsity_mask = sparsity_mask_list
        model.fit.coeff_vector = torch.nn.ParameterList(sparse_coeff_vector_list)

        # Alter optimizer for sparse coeff vectors
        optimizer.param_groups[1]['params'] = list(model.fit.coeff_vector.parameters())
        
        configs.optim['max_iterations'] = configs.optim['final_run_iterations']
    
    # Train without l1
    print() #empty line for correct printing
    configs.optim['l1'] = 0.0
    train(model, data, target, optimizer)
    configs.optim['l1'], configs.optim['max_iterations'] = external_values
    
    model.fit.coeff_vector_history += [model.fit.coeff_vector]
    model.fit.sparsity_mask_history += [model.fit.sparsity_mask]