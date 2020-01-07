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


def mech_library(data, prediction, Input_Expression, library_config):
    '''
    Constructs a library graph in 1D. Library config is dictionary with required terms.
    
    data in this case is strictly the time data
    prediction can be either stress or strain, but must be the data calculated as a result of the feedfoward run of the NN.
    Input_Expression is the analytical functional form of the stress or the strain, whichever is input. It must be differentiatable. Initially i am imagining this as a SymPy expression but perhaps it is possible to have it as a lambda function.
    library_config should be a dictionary stating the max order of differential and need specify nothing more
    
    '''
    
    max_order = library_config['diff_order']
    
    
    #This should not have sigma hardcoded as it could be epslion. If we want a type-ambivalent name, could just call it x.
    #If Input_Expression is a sympy thing, can it take torch tensors? Better to convert to Numpy and back?
    x = Input_Expression(data) #this won't work, its just a stand in for the idea of using the sympu expression to return a numpy array....
    # yh, that may need to be converted back into a tensor
    Exp_t = Input_Expression.diff(t) #here i am using a way of doinf differentiation in sympy
    x_t = Exp_t(data)
    Exp_tt = Exp_t.diff(t)
    x_tt = Exp_tt(data)
    '''
    ...
    or
    '''
    
    sigma_t = (data*torch.cos(data) - torch.sin(data))/(data**2)
    sigma_tt = -((data**2-2)*torch.sin(data)+2*data*torch.cos(data))/(data**3)
    sigma_ttt = (3*(data**2-2)*torch.sin(data)-data*(data**2-6)*torch.cos(data))/(data**4)
    sigma_tttt = (4*(data**2-6)*torch.cos(data)+(data**4-12*data**2+24)*torch.sin(data))/(data**5)
    
        du_2 = prediction.clone().detach()
    for order in np.arange(1, max_order):
        du_2 = torch.cat((du_2, grad(du_2[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)
    #Not sure where the grad_output comes in
    #y signifies output whatever that may be
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    y_t = dy[:, 0:1]
    dyy = grad(y_t, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    y_tt = dyy[:, 0:1]
    dyyy = grad(y_tt, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    y_ttt = dyyy[:, 0:1]
    
    #Set up an if statement here that asks which of x and y is strain and then ensures that this is the former series of values in the final 'du'. It does not have to be this way around, but it makes more sense to me.
    
    # Will therefore concatenate more like du = torch.cat((du_1, du_2), dim=1) assuming strain was teh input parmeter.
    # will also need to make sure in code above that the strain derivative in time (1st order is not part of this. As a resuklt, will need to understand teh nature of x and y a little earlier than just here
    
    du = torch.cat((sigma, sigma_t, sigma_tt,sigma_ttt, -prediction, -y_tt,-y_ttt), dim=1)
    samples= du.shape[0]
    theta = du.view(samples,-1)# I'm not convinced this line is not redundant and couldn't just have theta = du (maybe du.clone() or du[:,:] or something to avoid pointer issues)
    return [y_t], theta

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
    time_deriv = dy[:, 0:1]
    
    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order >1:
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

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
