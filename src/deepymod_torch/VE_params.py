import numpy as np
import sympy as sym
import IPython.display as dis


def model_params_from_coeffs(coeff_vector, model, print_expressions=False):
    
    order = len(coeff_vector)//2
    
    if model == 'GMM':
        coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(order)
    else: # model == 'GKM'
        coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(order)
    
    if print_expressions:
        coeff_equations_list = [sym.Eq(coeff_expression, coeff_value) for coeff_expression, coeff_value in zip(coeff_expression_list, coeff_vector)]
        dis.display(*coeff_equations_list)
    
    coeff_equations_list = [coeff_expression - coeff_value for coeff_expression, coeff_value in zip(coeff_expression_list, coeff_vector)]
    
    model_params_value_list = sym.solve(coeff_equations_list, model_params_mask_list)
    
    if len(model_params_value_list) == 0:
        model_params_value_list = [()] # Preserve structure for indexing consistency
        print('No solution possible for coefficient values and model complexity arrived at.')
    
    # Note, the returned solution may still contain 1 (or more?) symbols. This does not mean sym.solve() has failed.
    # Instead, it means that more than 1 set of model params can give rise to the given coeffs.
    # The symbol(s) remaining describe the common pattern in the model params for the same coeffs.
    return model_params_value_list, model_params_mask_list


def coeffs_from_model_params(E_mod_list, visc_list, model, print_expressions=False):
    
    order = len(visc_list)
    
    if model == 'GMM':
        coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(order)
    else: # model == 'GKM'
        coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(order)
    
    if print_expressions:
        dis.display(*coeff_expression_list)
    
    coeff_value_list = [coeff_expression.subs(zip(model_params_mask_list, E_mod_list+visc_list)) for coeff_expression in coeff_expression_list]
    
    return coeff_value_list


# KELVIN MODEL

def kelvin_coeff_expressions(order):
        
    E_0 = sym.symbols('E^K_0', real=True, negative=False)
    
    #Create and organise Kelvin general equation
    summation, model_params_list = kelvin_sym_sum(order)
    model_params_list = [E_0] + model_params_list
    RHS = 1/E_0 + summation
    RHS = RHS.together()
    stress_side, strain_side = sym.fraction(RHS)
    strain_side, stress_side = strain_side.expand(), stress_side.expand() # required for .coeff() to work.
    
    dt = sym.symbols('dt')
    
    # coeff of strain_t NOT 1.
    # Strain coeff expressions always first by convention.
    coeff_expressions_list = [strain_side.coeff(dt, strain_order) for strain_order in range(order+1)]
    coeff_expressions_list += [stress_side.coeff(dt, stress_order) for stress_order in range(order+1)]
    
    # Fix coeffs to a reference version of equation, where the coeff of strain_t is 1.
    coeff_expressions_list = [sym.simplify(coeff_expression/coeff_expressions_list[1]) for i, coeff_expression in enumerate(coeff_expressions_list) if i != 1]
    
    return coeff_expressions_list, model_params_list


def kelvin_sym_sum(order):
    
    E_Syms = [sym.symbols('E^K_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols('eta^K_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    dt = sym.symbols('dt')
    
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(E_Syms[Branch_Index] + Eta_Syms[Branch_Index]*dt)
        
    return Expression, all_syms


# MAXWELL MODEL

def maxwell_coeff_expressions(order):
        
    E_0 = sym.symbols('E^M_0', real=True, negative=False)
    
    #Create and organise Kelvin general equation
    summation, model_params_list = maxwell_sym_sum(order)
    model_params_list = [E_0] + model_params_list
    RHS = E_0 + summation
    RHS = RHS.together()
    strain_side, stress_side = sym.fraction(RHS)
    strain_side, stress_side = strain_side.expand(), stress_side.expand() # required for .coeff() to work.

    dt = sym.symbols('dt')
    
    # coeff of strain_t NOT 1.
    # Strain coeff expressions always first by convention.
    coeff_expressions_list = [strain_side.coeff(dt, strain_order) for strain_order in range(order+1)]
    coeff_expressions_list += [stress_side.coeff(dt, stress_order) for stress_order in range(order+1)]
    
    # Fix coeffs to a reference version of equation, where the coeff of strain_t is 1.
    coeff_expressions_list = [sym.simplify(coeff_expression/coeff_expressions_list[1]) for i, coeff_expression in enumerate(coeff_expressions_list) if i != 1]
    
    return coeff_expressions_list, model_params_list


def maxwell_sym_sum(order):
    
    E_Syms = [sym.symbols('E^M_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols('eta^M_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    dt = sym.symbols('dt')
    
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(1/E_Syms[Branch_Index] + 1/(Eta_Syms[Branch_Index]*dt))
        
    return Expression, all_syms



# CONVERT BETWEEN MODELS

def convert_between_models(E_mod_list, visc_list, origin_model, print_expressions=False):
    
    if origin_model == 'GMM':
        dest_model = 'GKM'
    else: # origin_model == 'GKM'
        dest_model = 'GMM'
    
    if print_expressions:
        print(f'Universal coefficients from {origin_model} parameters:')
    
    coeff_value_list = coeffs_from_model_params(E_mod_list, visc_list, origin_model, print_expressions=print_expressions)
    
    if print_expressions:
        print(f'{dest_model} parameters from universal coefficients:')
    
    dest_model_value_list = model_params_from_coeffs(coeff_value_list, dest_model, print_expressions=print_expressions)[0]
    
    # Absurd line for converting sympy objects back. [0] needed due to format of dest_model_value_list.
    params_result = [float(param) for param in dest_model_value_list[0]]
    E_mod_list_result = params_result[:len(E_mod_list)]
    visc_list_result = params_result[len(E_mod_list):]
    
    return E_mod_list_result, visc_list_result


# SCALE COEFFS DUE TO 'NORMALISATION'

def scaled_coeffs_from_true(true_coeffs, time_sf, strain_sf, stress_sf):
    
    true_coeffs_array = np.array(true_coeffs).flatten()
    alpha_array = coeff_scaling_values(true_coeffs_array, time_sf, strain_sf, stress_sf)
    scaled_coeff_guess = true_coeffs_array*alpha_array
    
    return list(scaled_coeff_guess)


def true_coeffs_from_scaled(scaled_coeffs, time_sf, strain_sf, stress_sf):
    
    scaled_coeffs_array = np.array(scaled_coeffs).flatten()
    alpha_array = coeff_scaling_values(scaled_coeffs_array, time_sf, strain_sf, stress_sf)
    true_coeffs = scaled_coeffs_array/alpha_array
    
    return list(true_coeffs)


def coeff_scaling_values(coeffs, time_sf, strain_sf, stress_sf):
    
    middle_index = len(coeffs)//2

    # calculate alpha_n for each term on RHS
    alpha_n_array = np.ones(coeffs.shape)

    # split into subarrays that modify in place original due to mutability of arrays
    # but allows for easier indexing.
    strain_subarray = alpha_n_array[:middle_index]
    stress_subarray = alpha_n_array[middle_index:]

    # apply dependant variable scaling of alpha_n
    strain_subarray *= strain_sf
    stress_subarray *= stress_sf

    # apply independant variable scaling of alpha_n
    stress_subarray[1] /= time_sf
    for idx in range(2, middle_index + 1):
        strain_subarray[idx-1] /= time_sf**idx
        stress_subarray[idx] /= time_sf**idx

    alpha_LHS = strain_sf/time_sf

    alpha_array = alpha_LHS/alpha_n_array
    
    return alpha_array