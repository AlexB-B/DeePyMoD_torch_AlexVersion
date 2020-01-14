import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product
from functools import reduce
import sys

sys.path.append('../../')
import data.Generation.VE_DataGen_Functions as vedg

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
    # Not sure if having 2 additional arguements will cause an issue for the deepmod implementation. Might need to build these into the library_config dictionary instead
    
    '''
    Constructs a library graph in 1D. Library config is dictionary with required terms.
    
    data in this case is strictly the time data
    prediction can be either stress or strain, but must be the data calculated as a result of the feedfoward run of the NN.
    Input_Expression is the analytical functional form of the stress or the strain, whichever is input. It must be differentiatable. This is a SymPy expression to allow for analytical differentiation.
    library_config should be a dictionary stating the max order of differential and need specify nothing more
    Input_Type should be either stress or strain and will determine primarily which data are placed into Strain_t (derived from input expression or prediction) and which data is placed in the first columns of theta.
    
    '''
    
    max_order = library_config['diff_order']
    
    #Begin by computing the values of the terms corresponding to the input, for which an analytical expression is given. This only needs to be done for the very first epoch, after which the values are known and stored in the library_config dictionary.
    if ('input_theta' in library_config) and (library_config['input_theta'].shape[0] == data.shape[0]):
        input_theta = library_config['input_theta']
    else:
        input_data = library_config['input_expr'](data)
        _, input_derivs = library_deriv(data, input_data, library_config)
        
        input_data, input_derivs = input_data.detach(), input_derivs.detach()
        
        input_theta = torch.cat((input_data, input_derivs[:, 1:]), dim=1)#indexing is to remove constant column of ones from the beginning of other_input_derivs
        library_config['input_theta'] = input_theta
    
    #Next use the result of the feedforward pass of the NN to calculate derivatives of your prediction with respect to time. 
    _, output_derivs = library_deriv(data, prediction, library_config)
    output_theta = torch.cat((prediction, output_derivs[:, 1:]), dim=1)
    
    #Next identify the Output/Input as Stress/Strain and organise into returned variables
    Input_Type = library_config['input_type']
    if not (Input_Type == ('Strain' or 'Stress')):
        print('Improper description of input choice. Defaulting to \'Strain\'')
        Input_Type = 'Strain'
    
    if Input_Type == 'Strain':
        Strain = input_theta
        Stress = output_theta
    else:
        Strain = output_theta
        Stress = input_theta
    
    Strain_t = Strain[:, 1:2] # Extract the first time derivative of strain
    Strain = torch.cat((Strain[:, 0:1], Strain[:, 2:]), dim=1) # remove this before it gets put into theta
    Strain *= -1 # The coefficient of all strain terms will always be negative. rather than hoping deepmod will find these negative terms, we assume the negative factor here and later on DeepMoD will just find positive coefficients
    theta = torch.cat((Strain, Stress), dim=1) # I have arbitrarily set the convention of making Strain the first columns of data
    
    return [Strain_t], theta

def mech_library_group(data, prediction, library_config):
    '''
    Here we define a library that contains first and second order derivatives to construct a list of libraries to study a set of different advection diffusion experiments.
    '''
    time_deriv_list = []
    theta_list = []
    
    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, theta = mech_library(data, prediction[:, output:output+1], library_config)
        time_deriv_list.extend(time_deriv)
        theta_list.append(theta)
        
    return time_deriv_list, theta_list


def library_deriv(data, prediction, library_config):
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
