import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product
from functools import reduce


def library_poly(prediction, library_config):
    '''
    Calculates polynomials of function u up to order M of given input, including M=0. Each column corresponds to power, i.e.
    the columns correspond to [1, u, u^2... , u^M].

    Parameters
    ----------
    prediction : tensor of size (N x 1)
        dataset whose polynomials are to be calculated.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    u : tensor of (N X (M+1))
        Tensor containing polynomials.
    '''
    max_order = library_config['poly_order']

    # Calculate the polynomes of u
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def mech_library(data, prediction, library_config):    
    
    # Begin by computing the values of the terms corresponding to the input, for which an analytical expression is given.
    # This only needs to be done for the very first epoch, after which the values are known and stored in the library_config dictionary.
    # The first condition checks if input_theta was previously already calculated and saved.
    # It also checks that the previous saved input_theta was not one that corresponded to a different data tensor.
    if ('input_theta' in library_config) and (library_config['input_theta'].shape[0] == data.shape[0]):
        input_theta = library_config['input_theta']
    else:
        input_data = library_config['input_expr'](data)
        input_derivs = library_deriv(data, input_data, library_config)
        
        input_data, input_derivs = input_data.detach(), input_derivs.detach()
        
        input_theta = torch.cat((input_data, input_derivs), dim=1)
        library_config['input_theta'] = input_theta
    
    # Next use the result of the feedforward pass of the NN to calculate derivatives of your prediction with respect to time. 
    output_derivs = library_deriv(data, prediction, library_config)
    output_theta = torch.cat((prediction, output_derivs), dim=1)
    
    # Next identify the input/output as Stress/Strain and organise into returned variables
    input_type = library_config['input_type']
    if input_type == 'Strain':
        strain = input_theta
        stress = output_theta
    elif input_type == 'Stress':
        strain = output_theta
        stress = input_theta
    else:
        print('Improper description of input choice. Was: '+input_type+'. Should be either \'Strain\' or \'Stress\'')
        
    strain_t = strain[:, 1:2] # Extract the first time derivative of strain
    strain = torch.cat((strain[:, 0:1], strain[:, 2:]), dim=1) # remove this before it gets put into theta
    strain *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
    theta = torch.cat((strain, stress), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
    return [strain_t], theta


def mech_library_real(data, prediction, library_config):    
    
    # The first column of prediction is always strain
    strain_derivs = library_deriv(data, prediction[:, :1], library_config)
    strain_theta = torch.cat((prediction[:, :1], strain_derivs), dim=1)
    
    # The second column is always stress
    stress_derivs = library_deriv(data, prediction[:, 1:], library_config)
    stress_theta = torch.cat((prediction[:, 1:], stress_derivs), dim=1)
    
    strain_t = strain_theta[:, 1:2] # Extract the first time derivative of strain
    strain_theta = torch.cat((strain_theta[:, 0:1], strain_theta[:, 2:]), dim=1) # remove this before it gets put into theta
    strain_theta *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
    theta = torch.cat((strain_theta, stress_theta), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
    return [strain_t], theta


def strain_input_library(data, prediction, library_config):    
    
    # Begin by computing the values of the terms corresponding to the input, for which an analytical expression is given.
    # This only needs to be done for the very first epoch, after which the values are known and stored in the library_config dictionary.
    # The first condition checks if input_theta was previously already calculated and saved.
    # It also checks that the previous saved input_theta was not one that corresponded to a different data tensor.
    if ('input_theta' in library_config) and (library_config['input_theta'].shape[0] == data.shape[0]):
        input_theta = library_config['input_theta']
    else:
        input_data = library_config['input_expr'](data)
        input_derivs = library_deriv(data, input_data, library_config)
        input_data, input_derivs = input_data.detach(), input_derivs.detach()
        input_theta = torch.cat((input_data, input_derivs), dim=1)
        library_config['input_theta'] = input_theta
    
    #Next use the result of the feedforward pass of the NN to calculate derivatives of your prediction with respect to time. 
    output_derivs = library_deriv(data, prediction, library_config)
    output_theta = torch.cat((prediction, output_derivs), dim=1)
    
    strain_t = input_theta[:, 1:2] # Extract the first time derivative of strain
    strain = torch.cat((input_theta[:, 0:1], input_theta[:, 2:]), dim=1) # remove this before it gets put into theta
    strain *= -1
    theta = torch.cat((strain, output_theta), dim=1)
    
    return [strain_t], theta


def stress_input_library(data, prediction, library_config):    
    
    # Begin by computing the values of the terms corresponding to the input, for which an analytical expression is given.
    # This only needs to be done for the very first epoch, after which the values are known and stored in the library_config dictionary.
    # The first condition checks if input_theta was previously already calculated and saved.
    # It also checks that the previous saved input_theta was not one that corresponded to a different data tensor.
    if ('input_theta' in library_config) and (library_config['input_theta'].shape[0] == data.shape[0]):
        input_theta = library_config['input_theta']
    else:
        input_data = library_config['input_expr'](data)
        input_derivs = library_deriv(data, input_data, library_config)
        input_data, input_derivs = input_data.detach(), input_derivs.detach()
        input_theta = torch.cat((input_data, input_derivs), dim=1)
        library_config['input_theta'] = input_theta
    
    #Next use the result of the feedforward pass of the NN to calculate derivatives of your prediction with respect to time. 
    output_derivs = library_deriv(data, prediction, library_config)
    output_theta = torch.cat((prediction, output_derivs), dim=1)
    
    strain_t = output_theta[:, 1:2] # Extract the first time derivative of strain
    strain = torch.cat((output_theta[:, 0:1], output_theta[:, 2:]), dim=1) # remove this before it gets put into theta
    strain *= -1
    theta = torch.cat((strain, input_theta), dim=1)
    
    return [strain_t], theta


# def mech_library_group(data, prediction, library_config):
#     '''
#     Here we define a library that contains first and second order derivatives to construct a list of libraries to study a set of different advection diffusion experiments.
#     '''
#     time_deriv_list = []
#     theta_list = []
    
#     # Creating lists for all outputs
#     for output in torch.arange(prediction.shape[1]):
#         time_deriv, theta = mech_library(data, prediction[:, output:output+1], library_config)
#         time_deriv_list.extend(time_deriv)
#         theta_list.append(theta)
        
#     return time_deriv_list, theta_list



def library_deriv(data, prediction, library_config):
    '''
    data and prediction must be single columned tensors.
    If it is desired to calculate the derivatives of different predictions wrt different data, this function must be called multiple times.
    This function does not return a column with the zeroth derivative (the prediction).
    '''
    max_order = library_config['diff_order']
    
    # First derivative builds off prediction.
    derivs = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    for _ in range(max_order-1):
        # Higher derivatives chain derivatives from first derivative.
        derivs = torch.cat((derivs, grad(derivs[:, -1:], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]), dim=1)
            
    return derivs


def library_deriv_old(data, prediction, library_config):
    '''
    Calculates derivative of function u up to order M of given input, including M=0. Each column corresponds to power, i.e.
    the columns correspond to [1, u_x, u_xx... , u_{x^M}].

    Parameters
    ----------
    data : tensor of size (N x 2)
        coordinates to whose respect the derivatives of prediction are calculated. First column is time, space second column.
    prediction : tensor of size (N x 1)
        dataset whose derivatives are to be calculated.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    time_deriv: tensor of size (N x 1)
        First temporal derivative of prediction.
    u : tensor of (N X (M+1))
        Tensor containing derivatives.
    '''
    max_order = library_config['diff_order']
    
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]

    if dy.shape[1] == 1:
        #nonsense result
        time_deriv = torch.ones_like(prediction)
        diff_column = 0
    else:
        time_deriv = dy[:, 0:1]
        diff_column = 1
    
    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(dy[:, 0:1]), dy[:, diff_column:diff_column+1]), dim=1)
        if max_order >1:
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, diff_column:diff_column+1]), dim=1)

    return time_deriv, du



def library_ODE(data, prediction, library_config):
    '''
    Calculates polynomials of function u up to order M of given input, including M=0. Each column corresponds to power, i.e.
    the columns correspond to [1, u, u^2... , u^M].

    Parameters
    ----------
    prediction : tensor of size (N x 1)
        dataset whose polynomials are to be calculated.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    u : tensor of (N X (M+1))
        Tensor containing polynomials.
    '''
    time_deriv_list = []
    theta_list = []
       
    # Creating lists for all outputs
    
    for output in torch.arange(prediction.shape[1]):
        time_deriv, theta = library_1D_in(data, prediction[:, :], library_config)
        time_deriv_list.extend(time_deriv)
        theta_list.append(theta)
    return time_deriv_list, theta_list


def string_matmul(list_1, list_2):
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod

def library_1D_in(data, prediction, library_config):
    '''
    Calculates a library function for a 1D+1 input for M coupled equations consisting of all polynomials up to order K and derivatives up to order
    L and all possible combinations (i.e. combining two terms) of these.

    Parameters
    ----------
    data : tensor of size (N x 2)
        coordinates to whose respect the derivatives of prediction are calculated. First column is time, space second column.
    prediction : tensor of size (N x M)
        dataset from which the library is constructed.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    time_deriv_list : tensor list of length of M
        list containing the time derivatives, each entry corresponds to an equation.
    theta : tensor
        library matrix tensor.
    '''
    
    poly_list = []
    deriv_list = []
    time_deriv_list = []
    # Creating lists for all outputs

    for output in torch.arange(prediction.shape[1]):
        time_deriv, du = library_deriv(data, prediction[:, output:output+1], library_config)
        u = library_poly(prediction[:, output:output+1], library_config)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)

    samples = time_deriv_list[0].shape[0]
    total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]
    
    # Calculating theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:

        theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1) # calculate all unique combinations of derivatives
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives
        theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
    return time_deriv_list, theta



def library_2Din_1Dout(data, prediction, library_config):
        '''
        Constructs a library graph in 1D. Library config is dictionary with required terms.
        '''

        # Polynomial
        
        max_order = library_config['poly_order']
        u = torch.ones_like(prediction)

        for order in np.arange(1, max_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

        # Gradients
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]
 
        du = torch.cat((torch.ones_like(u_x), u_x, u_y , u_xx, u_yy, u_xy), dim=1)

        samples= du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples,-1)
        
        return [u_t], theta

def library_2D_in_1D_out_group(data, prediction, library_config):
    '''
    Here we define a library that contains first and second order derivatives to construct a list of libraries to study a set of different advection diffusion experiments.
    '''
    time_deriv_list = []
    theta_list = []
    
    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, theta = library_2Din_1Dout(data, prediction[:, output:output+1], library_config)
        time_deriv_list.extend(time_deriv)
        theta_list.append(theta)
        
    return time_deriv_list, theta_list



def library_2Din_1Dout_PINN(data, prediction, library_config):
        '''
        Constructs a library to infer the advection and diffusion term from 2D data 
        '''       
        u = torch.ones_like(prediction)
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]
 
        du = torch.cat((u_x, u_y , u_xx, u_yy), dim=1)

        samples= du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples,-1)
        
        return [u_t], theta

def library_2D_in_PINN_group(data, prediction, library_config):
    '''
    Here we define a library that contains first and second order derivatives to construct a list of libraries to study a set of different advection diffusion experiments.
    '''
    time_deriv_list = []
    theta_list = []
    
    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, theta = library_2Din_1Dout_PINN(data, prediction[:, output:output+1], library_config)
        time_deriv_list.extend(time_deriv)
        theta_list.append(theta)
        
    return time_deriv_list, theta_list


def library_1D_in_group_b(data, prediction, library_config):
    '''
    Calculates a library function for a 1D+1 input for M coupled equations consisting of all polynomials up to order K and derivatives up to order
    L and all possible combinations (i.e. combining two terms) of these.

    Parameters
    ----------
    data : tensor of size (N x 2)
        coordinates to whose respect the derivatives of prediction are calculated. First column is time, space second column.
    prediction : tensor of size (N x M)
        dataset from which the library is constructed.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    time_deriv_list : tensor list of length of M
        list containing the time derivatives, each entry corresponds to an equation.
    theta : tensor
        library matrix tensor.
    '''

    time_deriv_list = []
    theta_list = []
    
    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, theta = library_1D_in(data, prediction[:, output:output+1], library_config)
        time_deriv_list.extend(time_deriv)
        theta_list.append(theta)
        
    return time_deriv_list, theta_list
