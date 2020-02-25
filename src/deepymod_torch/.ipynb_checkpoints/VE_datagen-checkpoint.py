import numpy as np
import scipy.integrate as integ
import sympy as sym
import torch

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
    
    # The following function interprets the provided model parameters differently depending on the input_type.
    # If the input_type is 'Strain' then the parameters are assumed to refer to a Maxwell model, whereas
    # if the input_type is 'Stress' then the parameters are assumed to refer to a Kelvin model.
    relax_creep_lambda = relax_creep(E_mods, viscs, input_type)
    
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


def relax_creep(E_mods, viscs, input_type):
    
    # The following function interprets the provided model parameters differently depending on the input_type.
    # If the input_type is 'Strain' then the parameters are assumed to refer to a Maxwell model, whereas
    # if the input_type is 'Stress' then the parameters are assumed to refer to a Kelvin model.
    # The equations used thus allow the data to be generated according to the model now designated.
    
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


def calculate_strain_finite_difference(time_array, input_expr, E_mods, viscs):
    # input is stress
    
    E_mods_1plus_array = np.array(E_mods[1:]).reshape(-1,1)
    viscs_array = np.array(viscs).reshape(-1,1)
    
    delta_t = time_array[1] - time_array[0]
    
    stress = np.array([])
    strain_i = np.zeros(viscs_array.shape)
    strain = np.array([])
    for t in time_array:
        stress = np.append(stress, input_expr(t))
        # In below line, need to remember that stress[-1] refers to stress(t) whereas strain_i[:, -1] refers to strain_i(t-delta_t) as stress is one element longer after previous line
        strain_i = np.append(strain_i, (delta_t*stress[-1] + viscs_array*strain_i[:, -1])/(delta_t*E_mods_1plus_array + viscs_array), axis=1)
        # Now strain_i[:, -1] refers to strain_i(t)
        strain = np.append(strain, stress[-1]/E_mods[0] + np.sum(strain_i[:, -1]))
    
    stress = stress.reshape(-1,1)
    strain = strain.reshape(-1,1)
    
    return strain, stress


def calculate_stress_finite_difference(time_array, input_expr, E_mods, viscs):
    # input is strain
    
    E_mods_1plus_array = np.array(E_mods[1:]).reshape(-1,1)
    viscs_array = np.array(viscs).reshape(-1,1)
    taus = viscs_array/E_mods_1plus_array
    
    delta_t = time_array[1] - time_array[0]

    strain = np.zeros(1)
    stress = np.array([])
    stress_i = np.zeros(viscs_array.shape)
    for t in time_array:
        strain = np.append(strain, input_expr(t))
        # In below line, need to remember that strain[-1] refers to strain(t) and strain[-2] refers to strain(t-delta_t), whereas stress_i[:, -1] refers to stress_i(t-delta_t) as strain is one element longer after previous line.
        stress_i = np.append(stress_i, (E_mods_1plus_array*(strain[-1] - strain[-2]) + stress_i[:, -1])/(1 + delta_t/taus), axis=1)
        # Now stress_i[:, -1] refers to stress_i(t)
        stress = np.append(stress, strain[-1]*E_mods[0] + np.sum(stress_i[:, -1]))
    
    stress = stress.reshape(-1,1)
    strain = strain[1:].reshape(-1,1)
    
    return strain, stress


def wave_packet_lambdas_sum(freq_max, freq_step, std_dev):
    
    # changing freq_max changes the 'detail' in the wave packet.
    # Changing the freq_step changes the seperation of the wave packets.
    # changing the std_dev changes the size of the wavepacket.
    # replacing the gaussian weighting of the discrete waves with a constant makes the wavepacket look like a sinc function.
    
    # In terms of time to calculate, this scales much more poorly than wave_packet_lambdas_integ.
    # small freq_step slows these lambdas down in a linear fashion.
    # For sufficiently small freq_step and large evaluation range, the two can take equally long.
    
    mean = freq_max/2
    
    omega_array = np.arange(0, freq_max, freq_step)
    
    output_lambda = lambda t: freq_step*sum([np.exp(-((omega-mean)**2)/(2*std_dev**2))*np.sin(omega*t) for omega in omega_array])
    d_output_lambda = lambda t: freq_step*sum([omega*np.exp(-((omega-mean)**2)/(2*std_dev**2))*np.cos(omega*t) for omega in omega_array])
    
    torch_output_lambda = lambda t: freq_step*sum([np.exp(-((omega-mean)**2)/(2*std_dev**2))*torch.sin(omega*t) for omega in omega_array])
    
    return output_lambda, d_output_lambda, torch_output_lambda


def wave_packet_lambdas_integ(freq_max, std_dev):
    
    # changing freq_max changes the 'detail' in the wave packet.
    # changing the std_dev changes the size of the wavepacket.
    # replacing the gaussian weighting of the discrete waves with a constant makes the wavepacket look like a sinc function.
    
    # This method using numerical integration arguably produces a closer approximation of the ideal wavepacket [than wave_packet_lambdas_sum] but:
    # - is arguably less transparent
    # - In terms of time to calculate, this generally takes longer than wave_packet_lambdas_sum as a result mostly of the integration (np.gradient is fast).
    #   Evaluating the lambdas over large time ranges slows these lambdas down.
    #   For sufficiently small freq_step and large evaluation range, the two can take equally long.
    # - If the additional arguement of freq_step is small, is indistinguishable from from lambda from wave_packet_lambdas_sum.
    # - generates a warning flag at large evaluation ranges.
    
    mean = freq_max/2
    
    half_deriv_range = 5
    deriv_steps = 101
    middle = deriv_steps // 2
    
    integrand_lambda = lambda omega, t: np.exp(-((omega-mean)**2)/(2*std_dev**2)) * np.sin(omega*t)
    output_lambda = lambda t: integ.quad(integrand_lambda, 0, freq_max, args=(t))[0]
    array_output_lambda = lambda t_array: np.array([integ.quad(integrand_lambda, 0, freq_max, args=(t))[0] for t in t_array])
    d_output_lambda = lambda t: np.gradient(array_output_lambda(np.linspace(t-half_deriv_range, t+half_deriv_range, deriv_steps)), 
                                            np.linspace(t-half_deriv_range, t+half_deriv_range, deriv_steps))[middle]
    
    torch_integrand_lambda = lambda omega, t: torch.exp(-((omega-mean)**2)/(2*std_dev**2)) * torch.sin(omega*t)
    torch_output_lambda = lambda t_tensor: integ.quad(torch_integrand_lambda, 0, freq_max, args=(t))[0]
    
    return output_lambda, d_output_lambda, torch_output_lambda


#Data Validation routines
def equation_residuals(time_array, strain_array, stress_array, E_mods, viscs, input_type):
    
    diff_order = len(viscs)

    strain_theta = num_derivs(strain_array, time_array, diff_order)
    stress_theta = num_derivs(stress_array, time_array, diff_order)
    num_theta = np.concatenate((strain_theta, stress_theta), axis=1)
    
    if input_type == 'Stress':
        coeffs = VE_params.coeffs_from_model_params_kelvin(E_mods, viscs)
    elif input_type == 'Strain':
        coeffs = VE_params.coeffs_from_model_params_maxwell(E_mods, viscs)
    else:
        print('input_type is wrong')
        return None
    
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