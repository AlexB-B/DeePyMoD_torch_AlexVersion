import numpy as np
import scipy.integrate as integ
import sympy as sym

import deepymod_torch.VE_params as VE_params


#Data generation routines
def calculate_strain_stress(input_type, time_array, input_expr, E_mods, viscs, D_input_lambda=None):
    
    if D_input_lambda:
        input_lambda = input_expr
    else:
        t = sym.symbols('t', real=True)
        D_input_expr = input_expr.diff(t)
        
        input_lambda = sym.lambdify(t, input_expr)
        D_input_lambda = sym.lambdify(t, D_input_expr)
    
    relax_creep_lambda = relax_creep_kelvin(E_mods, viscs, input_type)
    
    if relax_creep_lambda == False:
        return False, False
    
    start_time_point = time_array[0]
    
    integrand_lambda = lambda x, t: relax_creep_lambda(t-x)*D_input_lambda(x)
    integral_lambda = lambda t: integ.quad(integrand_lambda, start_time_point, t, args=(t))[0]
    
    output_array = np.array([])
    input_array = np.array([])
    for time_point in time_array:
        first_term = input_lambda(start_time_point)*relax_creep_lambda(time_point-start_time_point)
        second_term = integral_lambda(time_point)
        output_array = np.append(output_array, first_term + second_term)
        input_array = np.append(input_array, input_lambda(time_point))
    
    if input_type == 'Strain':
        strain_array = input_array
        stress_array = output_array
    else:
        strain_array = output_array
        stress_array = input_array
        
    strain_array = strain_array.reshape(time_array.shape)
    stress_array = stress_array.reshape(time_array.shape)
    
    return strain_array, stress_array


def relax_creep_kelvin(E_mods, viscs, input_type):
    
    E_mods_1plus_array = np.array(E_mods[1:]).reshape(-1,1)
    viscs_array = np.array(viscs).reshape(-1,1)
    
    taus = viscs_array/E_mods_1plus_array
    
    if input_type == 'Strain':
        relax_creep_lambda = lambda t: E_mods[0] + np.sum(np.exp(-t/taus)*E_mods_1plus_array)
    elif input_type == 'Stress':
        relax_creep_lambda = lambda t: 1/E_mods[0] + np.sum((1-np.exp(-t/taus))/E_mods_1plus_array)
    else:
        print('Incorrect input_type')
        relax_creep_lambda = False
    
    return relax_creep_lambda


def calculate_strain_stress_alt(input_type, time_array, input_lambda, E_mods, viscs):
    
    relax_creep_lambda, D_relax_creep_lambda = relax_creep_kelvin_alt(E_mods, viscs, input_type)
    
    if relax_creep_lambda == False:
        return False, False
    
    start_time_point = time_array[0]
    
    integrand_lambda = lambda x, t: D_relax_creep_lambda(t-x)*input_lambda(x)
    integral_lambda = lambda t: integ.quad(integrand_lambda, start_time_point, t, args=(t))[0]
    
    output_array = np.array([])
    input_array = np.array([])
    for time_point in time_array:
        first_term = input_lambda(start_time_point)*relax_creep_lambda(time_point)#-start_time_point)
        second_term = integral_lambda(time_point)
        output_array = np.append(output_array, first_term + second_term)
        input_array = np.append(input_array, input_lambda(time_point))
    
    if input_type == 'Strain':
        strain_array = input_array
        stress_array = output_array
    else:
        strain_array = output_array
        stress_array = input_array
        
    strain_array = strain_array.reshape(time_array.shape)
    stress_array = stress_array.reshape(time_array.shape)
    
    return strain_array, stress_array


def relax_creep_kelvin_alt(E_mods, viscs, input_type):
    
    E_mods_1plus_array = np.array(E_mods[1:]).reshape(-1,1)
    viscs_array = np.array(viscs).reshape(-1,1)
    
    taus = viscs_array/E_mods_1plus_array
    
    if input_type == 'Strain':
        relax_creep_lambda = lambda t: E_mods[0] + np.sum(np.exp(-t/taus)*E_mods_1plus_array)
        D_relax_creep_lambda = lambda t: -np.sum(np.exp(-t/taus)*E_mods_1plus_array/taus)
    elif input_type == 'Stress':
        relax_creep_lambda = lambda t: 1/E_mods[0] + np.sum((1-np.exp(-t/taus))/E_mods_1plus_array)
        D_relax_creep_lambda = lambda t: np.sum(np.exp(-t/taus)/(E_mods_1plus_array*taus))
    else:
        print('Incorrect input_type')
        relax_creep_lambda = False
        D_relax_creep_lambda = False
    
    return relax_creep_lambda, D_relax_creep_lambda

#Data Validation routines
def equation_residuals(time_array, strain_array, stress_array, E_mods, viscs):
    
    diff_order = len(viscs)

    strain_theta = num_derivs(strain_array, time_array, diff_order)
    stress_theta = num_derivs(stress_array, time_array, diff_order)
    num_theta = np.concatenate((strain_theta, stress_theta), axis=1)
    
    coeffs = VE_params.coeffs_from_model_params(E_mods, viscs)
    coeffs_strain_array = np.array([coeffs[0]] + [1] + coeffs[1:diff_order])
    coeffs_stress_array = np.array(coeffs[diff_order:])
    coeffs_array = np.concatenate((-coeffs_strain_array, coeffs_stress_array)).reshape(-1,1)
    
    residuals = num_theta @ coeffs_array
    
    return residuals


def num_derivs(dependent_data, independent_data, diff_order):
    
    data_derivs = dependent_data.copy()
    data_derivs = data_derivs.reshape(-1, 1)
    for _ in range(diff_order):
        data_derivs = np.append(data_derivs, np.gradient(data_derivs[:, -1].flatten(), independent_data.flatten()).reshape(-1,1), axis=1)
    
    return data_derivs