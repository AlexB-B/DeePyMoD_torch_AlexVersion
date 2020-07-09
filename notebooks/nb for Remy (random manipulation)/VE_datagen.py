import numpy as np
import scipy.integrate as integ
import sympy as sym
import torch
import torch.autograd as auto

import deepymod_torch.VE_params as VE_params


# Data generation using Boltzmann superposition integrals.
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


# Data generation using finite difference approx of models.
def calculate_strain_finite_difference(time_array, input_expr, E_mods, viscs):
    # input is stress, model is GKM
    
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
    
    stress = stress.reshape(time_array.shape)
    strain = strain.reshape(time_array.shape)
    
    return strain, stress


def calculate_stress_finite_difference(time_array, input_expr, E_mods, viscs):
    # input is strain, model is GMM
    
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
    
    stress = stress.reshape(time_array.shape)
    strain = strain[1:].reshape(time_array.shape)
    
    return strain, stress


# Data generation from differential equation
def calculate_int_diff_equation(time, response, input_lambda_or_network, coeff_vector, sparsity_mask, library_diff_order, input_type):
    
    # time, response and coeff_vector should either all be tensors, or all be arrays.
    if type(coeff_vector) is torch.Tensor:
        coeff_vector = coeff_vector.detach()
        time_array = np.array(time.detach())
    else: # else numpy array
        time_array = time
    
    coeff_array = np.array(coeff_vector)
    mask_array = np.array(sparsity_mask)
    
    # Function returns masks and arrays in format such that mask values correspond to diff order etc.
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeff_array, mask_array, library_diff_order)
    
    if input_type == 'Strain':
        input_coeffs, input_mask = strain_coeffs_mask
        response_coeffs, response_mask = stress_coeffs_mask
    else: # else 'Stress'
        response_coeffs, response_mask = strain_coeffs_mask
        input_coeffs, input_mask = stress_coeffs_mask
    
    # Coeffs as stated refer to an equation with all stress terms on one side, and all strain on the other.
    # Depending on which coeffs are paired with the response, they must be moved across to the other side.
    # This is accomplished by making them negative, and concatenating carefully to prepare for alignment with correct terms.
    # The coeff for the highest derivative of response variable is left behind, and moved later.
    neg_response_coeffs = -response_coeffs[:-1]
    coeffs_less_dadt_array = np.concatenate((input_coeffs, neg_response_coeffs))
        
    # Don't skip derivative orders, but avoids calculating higher derivatives than were eliminated.
    max_input_diff_order = max(input_mask)
    max_response_diff_order = max(response_mask)
    
    def calc_dU_dt(U, t):
        # U is list (seems to be converted to array before injection) of response and increasing orders of derivative of response.
        # Returns list of derivative of each input element in U.
        
        if type(input_lambda_or_network) is type(lambda:0):
            # Calculate numerical derivatives of manipualtion variable by spooling a dummy time series around t.
            input_derivs = num_derivs_single(t, input_lambda_or_network, max_input_diff_order)
        else: # network
            t_tensor = torch.tensor([t], dtype=torch.float32, requires_grad=True)
            input_derivs = [input_lambda_or_network(t_tensor)[0]] # if I ever supply module, is because voltage is 1 of 2 outputs of NN. The [0] here selects the voltage
            for _ in range(max_input_diff_order):
                input_derivs += [auto.grad(input_derivs[-1], t_tensor, create_graph=True)[0]]

            input_derivs = np.array([input_deriv.item() for input_deriv in input_derivs])
        
        # Use masks to select manipulation terms from numerical work above...
        # ...and response terms from function parameter, considering ladder of derivative substitions.
        # Concatenate carefully to align with coefficient order.
#         input_terms = np.array([input_derivs[mask_value] for mask_value in input_mask])
        input_terms = input_derivs[input_mask]
#         response_terms = np.array([U[mask_value] for mask_value in response_mask[:-1]])
        response_terms = U[response_mask[:-1]]
        terms_array = np.concatenate((input_terms, response_terms))
        
        # Multiply aligned coeff-term pairs and divide by coeff of highest order deriv of response variable.
        da_dt = np.sum(coeffs_less_dadt_array*terms_array)/response_coeffs[-1]
        
        dU_dt = list(U[1:]) + [da_dt]
        
        return dU_dt
    
    start_index = max_response_diff_order # If want to start recalc at different point modify start_index NOT max_response_diff_order. Default behaviour is to take same value but 2 different purposes.
    # Initial values of derivatives. IVs taken a few steps into the response curve to avoid edge effects.
    if type(time) is torch.Tensor:
        # Initial values of response and response derivatives determined using torch autograd.
        IVs = [response[start_index]]
        for _ in range(max_response_diff_order-1):
            IVs += [auto.grad(IVs[-1], time, create_graph=True)[0][start_index]]

        IVs = [IV.item() for IV in IVs]
        
        # The few skipped values from edge effect avoidance tacked on again.
        calculated_response_array_initial = np.array(response[:start_index].detach()).flatten()
    else: # else numpy array
        # Initial values of response and response derivatives determined using numpy gradient.
        response_derivs = num_derivs(response, time_array, max_response_diff_order-1)[start_index, :]
        IVs = list(response_derivs)
        
        # The few skipped values from edge effect avoidance tacked on again.
        calculated_response_array_initial = response[:start_index].flatten()
    
    reduced_time_array = time_array[start_index:].flatten()
    
    calculated_response_array = integ.odeint(calc_dU_dt, IVs, reduced_time_array)[:, 0]
    
    calculated_response_array = np.concatenate((calculated_response_array_initial, calculated_response_array)).reshape(-1, 1)
    
    return calculated_response_array


def calculate_int_diff_equation_initial(time_array, input_lambda, E, eta, input_type, model):
    
    shape = time_array.shape
    time_array = time_array.flatten()
    input_array = input_lambda(time_array)
    
    if model == 'GMM':
        if input_type == 'Strain':
            instant_response = input_array[0]*sum(E)
        else: # else 'Stress'
            instant_response = input_array[0]/sum(E)
    else: # model = 'GKM'
        if input_type == 'Strain':
            instant_response = input_array[0]*E[0]
        else: # else 'Stress'
            instant_response = input_array[0]/E[0]
    
    coeff_array = np.array(VE_params.coeffs_from_model_params(E, eta, model))
    mask_array = np.arange(len(coeff_array))
    library_diff_order = len(coeff_array) // 2
    
    # Function returns masks and arrays in format such that mask values correspond to diff order etc.
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeff_array, mask_array, library_diff_order)
    
    if input_type == 'Strain':
        input_coeffs, input_mask = strain_coeffs_mask
        response_coeffs, response_mask = stress_coeffs_mask
    else: # else 'Stress'
        response_coeffs, response_mask = strain_coeffs_mask
        input_coeffs, input_mask = stress_coeffs_mask
    
    # Coeffs as stated refer to an equation with all stress terms on one side, and all strain on the other.
    # Depending on which coeffs are paired with the response, they must be moved across to the other side.
    # This is accomplished by making them negative, and concatenating carefully to prepare for alignment with correct terms.
    # The coeff for the highest derivative of response variable is left behind, and moved later.
    neg_response_coeffs = -response_coeffs[:-1]
    coeffs_less_dadt_array = np.concatenate((input_coeffs, neg_response_coeffs))
    
    def calc_dU_dt(U, t):
        # U is list (seems to be converted to array before injection) of response and increasing orders of derivative of response.
        # Returns list of derivative of each input element in U.
        
        # Calculate numerical derivatives of manipualtion variable by spooling a dummy time series around t.
        input_derivs = num_derivs_single(t, input_lambda, library_diff_order)
        
        terms_array = np.concatenate((input_derivs, U))

        # Multiply aligned coeff-term pairs and divide by coeff of highest order deriv of response variable.
        da_dt = np.sum(coeffs_less_dadt_array*terms_array)/response_coeffs[-1]
        
        dU_dt = list(U[1:]) + [da_dt]
        
        return dU_dt
        
    IVs = [instant_response] + [0]*(library_diff_order-1)
    # A scalable method for getting accurate IVs for higher than zeroth order derivative would reuire sympy implemented differentiation and would be (symbolic_input_expr*symbolic_relax_or_creep_expr).diff()[.diff().diff() etc] evaluated at 0.
    calculated_response_array = integ.odeint(calc_dU_dt, IVs, time_array)[:, 0:1]
    
    if input_type == 'Strain':
        strain_array = input_array
        stress_array = calculated_response_array
    else: # else 'Stress'
        strain_array = calculated_response_array
        stress_array = input_array
        
    strain_array, stress_array = strain_array.reshape(shape), stress_array.reshape(shape)
        
    return strain_array, stress_array


def calculate_finite_difference_diff_equation(time_array, strain_array, stress_array, coeff_vector, sparsity_mask, library_diff_order, input_type):
    
    # MAKE SENSE OF MASKS
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeff_vector, sparsity_mask, library_diff_order)
    strain_coeffs, strain_mask = list(strain_coeffs_mask[0]), list(strain_coeffs_mask[1])
    stress_coeffs, stress_mask = list(stress_coeffs_mask[0]), list(stress_coeffs_mask[1])
    
    # GENERATE FINITE DIFFERENCE EXPRESSIONS FOR STRAIN AND STRESS
    # Avoid dealing with higher order derivatives that were eliminated for both stress and strain.
    max_remaining_diff_order = max(strain_mask+stress_mask)
    
    # Recover strain symbols and time step symbol
    eps_syms, delta = generate_finite_difference_approx_deriv('epsilon', max_remaining_diff_order)[1:]
    # Build strain expression by generating finite difference approximation and combining with coeffs.
    strain_expr = sym.S(0)
    for coeff_index, mask_value in enumerate(strain_mask):
        term_approx_expr = generate_finite_difference_approx_deriv('epsilon', mask_value)[0]
        strain_expr += strain_coeffs[coeff_index]*term_approx_expr
    
    # Recover stress symbols
    sig_syms = generate_finite_difference_approx_deriv('sigma', max_remaining_diff_order)[1]
    # Build stress expression by generating finite difference approximation and combining with coeffs.
    stress_expr = sym.S(0)
    for coeff_index, mask_value in enumerate(stress_mask):
        term_approx_expr = generate_finite_difference_approx_deriv('sigma', mask_value)[0]
        stress_expr += stress_coeffs[coeff_index]*term_approx_expr
    
    # DETERMINE EXPRESSION TO RETURN RESPONSE
    # Subsitute time step symbol for value. This also simplifies expressions to sums of coeff*unique_symbol terms.
    delta_t = float(time_array[1] - time_array[0])
    strain_expr = strain_expr.subs(delta, delta_t)
    stress_expr = stress_expr.subs(delta, delta_t)
    
    if input_type == 'Strain':
        input_array = strain_array
        response_array = stress_array
        input_expr = strain_expr
        response_expr = stress_expr
        input_syms = eps_syms
        response_syms = sig_syms
    else: # else 'Stress'
        input_array = stress_array
        response_array = strain_array
        input_expr = stress_expr
        response_expr = strain_expr
        input_syms = sig_syms
        response_syms = eps_syms
    
    # Rearrange expressions to create equation for response.
    # The coeff of the zeroth order of any symbol is the coeff a constant wrt to that symbol.
    # The below line thus produces an expression of everything in stress_expr but coeff*stress(t).
    response_side_to_subtract = response_expr.coeff(response_syms[0], 0)
    input_side = input_expr - response_side_to_subtract
    response_coeff = response_expr.coeff(response_syms[0]) # no order means 1st order, ie only coeff of stress(t).
    evaluate_response = input_side/response_coeff
    
    # EVALUATE RESPONSE FOR ALL TIME POINTS
    # Evaluation requires the use of some initial values for stress and strain.
    # The higher the order of derivative, the more 'initial values' needed.
    # We pick from the full array of the controlled variable, but response builds off only initial values.
    initial_index = max_remaining_diff_order
    flat_input_array = input_array.flatten()
    calculated_response_array = response_array[:initial_index].flatten()
        
    # Evaluate for each time point beyond initial values.
    for t_index in range(initial_index, len(time_array)):
        # Dictionaries created mapping symbol to stress and strain values at correct historic time point.
        # Reverse order slicing of symbols to match values correctly.
        # Always chooses the most recent stress and strain values wrt current time point.
        # Avoids including response(t) symbol.
        input_subs_dict = dict(zip(input_syms[::-1], flat_input_array[t_index-initial_index:t_index+1]))
        response_subs_dict = dict(zip(response_syms[:0:-1], calculated_response_array[-initial_index:]))
        subs_dict = {**input_subs_dict, **response_subs_dict} # combine dictionaries
        
        # Evaluate expression using dictionary as guide for all substitutions. Append to stress so far calculated.
        calculated_response_array = np.append(calculated_response_array, evaluate_response.evalf(subs=subs_dict))
    
    calculated_response_array = calculated_response_array.reshape(time_array.shape)
    
    # returned array is the opposite quantity to the input as specified by input type.
    return calculated_response_array


def generate_finite_difference_approx_deriv(sym_string, diff_order):
    
    # Each symbol refers to the dependant variable at previous steps through independant variable values.
    # Starts from the current variable value and goes backwards.
    syms = [sym.symbols(sym_string+'_{t-'+str(steps)+'}', real=True) for steps in range(diff_order+1)]
    # Symbol represents the step change in independant variable.
    delta = sym.symbols('Delta', real=True, positive=True)
    
    # Correct coeff for each symbol for historic value of dependant variable can be determined by analogy.
    # The coeffs of x following expansion yield the desired coeffs, with polynomial order and number of steps back exchanged. 
    x = sym.symbols('x') # Dummy variable
    signed_pascal_expression = (1-x)**diff_order
    signed_pascal_expression = signed_pascal_expression.expand()
    
    # Numerator of expression for finite approx.
    expr = sym.S(0)
    for poly_order in range(diff_order+1):
        expr += signed_pascal_expression.coeff(x, poly_order)*syms[poly_order]
    
    # Divide numerator by denominator of expression for finite approx.
    expr /= delta**diff_order
    
    return expr, syms, delta


def align_masks_coeffs(coeff_vector, sparsity_mask, library_diff_order):
    
    # Create boolean arrays to slice mask into strain and stress parts
    first_stress_mask_value = library_diff_order
    is_strain = sparsity_mask < first_stress_mask_value
    is_stress = sparsity_mask >= first_stress_mask_value
    
    # Slice mask and coeff values and shift stress mask so that mask values always refer to diff order.
    strain_mask = sparsity_mask[is_strain]
    strain_coeffs = list(coeff_vector[is_strain].flatten())
    stress_mask = list(sparsity_mask[is_stress] - first_stress_mask_value)
    stress_coeffs = list(coeff_vector[is_stress].flatten())
    
    # Adjust strain mask and coeffs to account for missing first strain derivative.
    # Mask values above 0 are shifted up and a mask value of 1 added so that mask values always refer to diff order.
    strain_mask_stay = list(strain_mask[strain_mask < 1])
    strain_mask_shift = list(strain_mask[strain_mask > 0] + 1)
    strain_t_idx = len(strain_mask_stay)
    strain_mask = strain_mask_stay + [1] + strain_mask_shift
    # A coeff of 1 is added for the coeff of the first strain derivative.
    strain_coeffs.insert(strain_t_idx, 1)
    
    # Arrays in, arrays out.
    strain_coeffs_mask = np.array(strain_coeffs), np.array(strain_mask, dtype=int)
    stress_coeffs_mask = np.array(stress_coeffs), np.array(stress_mask, dtype=int)
    
    return strain_coeffs_mask, stress_coeffs_mask


# Wave packet lambda generation
def wave_packet_lambdas_sum(freq_max, freq_step, std_dev, amp):
    
    # changing freq_max changes the 'detail' in the wave packet.
    # Changing the freq_step changes the seperation of the wave packets.
    # changing the std_dev changes the size of the wavepacket.
    # replacing the gaussian weighting of the discrete waves with a constant makes the wavepacket look like a sinc function.

    mean = freq_max/2
    
    omega_array = np.arange(freq_step, freq_max+(freq_step/2), freq_step)
    
    output_lambda = lambda t: amp*freq_step*sum([np.exp(-((omega-mean)**2)/(2*std_dev**2))*np.sin(omega*t) for omega in omega_array])
    d_output_lambda = lambda t: amp*freq_step*sum([omega*np.exp(-((omega-mean)**2)/(2*std_dev**2))*np.cos(omega*t) for omega in omega_array])
    
    torch_output_lambda = lambda t: amp*freq_step*sum([np.exp(-((omega-mean)**2)/(2*std_dev**2))*torch.sin(omega*t) for omega in omega_array])
    
    return output_lambda, d_output_lambda, torch_output_lambda


def wave_packet_lambdas_integ(freq_max, std_dev, amp):
    
    # changing freq_max changes the 'detail' in the wave packet.
    # changing the std_dev changes the size of the wavepacket.
    # replacing the gaussian weighting of the discrete waves with a constant makes the wavepacket look like a sinc function.
    
    # This method using numerical integration arguably produces a closer approximation of the ideal wavepacket [than wave_packet_lambdas_sum] but:
    # - Cannot be implemented to handle PyTorch tensors
    # - is arguably less transparent
    # - In terms of time to calculate, this generally takes longer than wave_packet_lambdas_sum as a result mostly of the integration (np.gradient is fast).
    # - generates a warning flag at large evaluation ranges.
    
    mean = freq_max/2
    
    integrand_lambda = lambda omega, t: np.exp(-((omega-mean)**2)/(2*std_dev**2)) * np.sin(omega*t)
    output_lambda_single = lambda t: integ.quad(integrand_lambda, 0, freq_max, args=(t))[0]
    output_lambda = lambda t_array: amp*np.array([output_lambda_single(t) for t in t_array])
    d_output_lambda = lambda t_array: np.array([num_derivs_single(t, output_lambda, 1)[1] for t in t_array])
    
    # The below code will likely not work, untested so far, as a torch tensor will likely need to be converted to ...
    # ... a numpy array to put put into quad, and therefore lose its history. (will throw up error as no .detach())
    torch_integrand_lambda = lambda omega, t: torch.exp(-((omega-mean)**2)/(2*std_dev**2)) * torch.sin(omega*t)
    torch_output_lambda_single = lambda t: integ.quad(torch_integrand_lambda, 0, freq_max, args=(t))[0]
    torch_output_lambda = lambda t_tensor: amp*torch.stack([torch_output_lambda_single(t) for t in t_tensor])
    
    return output_lambda, d_output_lambda, torch_output_lambda


#Data Validation routine
def equation_residuals(time_array, strain_array, stress_array, coeffs, sparsity_mask='full', diff_order='full'):
    
    if diff_order == 'full': # this and sparsity_mask should either both be default, or both be specified.
        sparsity_mask = np.arange(len(coeffs))
        diff_order = len(coeffs)//2
    
    # In case they are tensors. Tensors must still be detached as arguements.
    time_array, strain_array, stress_array = np.array(time_array), np.array(strain_array), np.array(stress_array)
    coeffs, sparsity_mask = np.array(coeffs).astype(float), np.array(sparsity_mask)
    
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeffs, sparsity_mask, diff_order)
    strain_coeffs, strain_mask = strain_coeffs_mask[0], strain_coeffs_mask[1]
    stress_coeffs, stress_mask = stress_coeffs_mask[0], stress_coeffs_mask[1]
    
    coeffs_array = np.concatenate((-strain_coeffs, stress_coeffs)).reshape(-1,1)
    
    strain_theta = num_derivs(strain_array, time_array, diff_order)
    stress_theta = num_derivs(stress_array, time_array, diff_order)
    
    num_theta = np.concatenate((strain_theta[:, strain_mask], stress_theta[:, stress_mask]), axis=1)

    residuals = num_theta @ coeffs_array
        
    return residuals


def equation_residuals_auto(theta, strain_t, coeffs, sparsity_mask='full', diff_order='full'):
    
    if diff_order == 'full': # this and sparsity_mask should either both be default, or both be specified.
        sparsity_mask = np.arange(len(coeffs))
        diff_order = len(coeffs)//2
    
    # In case they are tensors. Tensors must still be detached as arguements.
    theta, strain_t = np.array(theta), np.array(strain_t)
    coeffs, sparsity_mask = np.array(coeffs).astype(float), np.array(sparsity_mask)
    
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeffs, sparsity_mask, diff_order)
    strain_coeffs, strain_mask = strain_coeffs_mask[0], strain_coeffs_mask[1]
    stress_coeffs, stress_mask = stress_coeffs_mask[0], stress_coeffs_mask[1]
    
    coeffs_array = np.concatenate((-strain_coeffs, stress_coeffs)).reshape(-1,1)
    
    strain_theta = np.concatenate((-theta[:, 0:1], strain_t, -theta[:, 1:diff_order]), axis=1)
    stress_theta = theta[:, diff_order:]
    
    reduced_strain_theta = [strain_theta[:, mask_value:mask_value+1] for mask_value in strain_mask]
    reduced_stress_theta = [stress_theta[:, mask_value:mask_value+1] for mask_value in stress_mask]
    num_theta = np.concatenate(reduced_strain_theta + reduced_stress_theta, axis=1)
    
    residuals = num_theta @ coeffs_array
        
    return residuals


# Numerical derivatives using NumPy
def num_derivs(dependent_data, independent_data, diff_order):
    
    data_derivs = dependent_data.copy()
    data_derivs = data_derivs.reshape(-1, 1)
    for _ in range(diff_order):
        data_derivs = np.append(data_derivs, np.gradient(data_derivs[:, -1].flatten(), independent_data.flatten()).reshape(-1,1), axis=1)
    
    return data_derivs


def num_derivs_single(t, input_lambda, diff_order, num_girth=1, num_depth=101):
    
    # Higher derivs need further reaching context for accuracy.
    mod_num_girth = num_girth*diff_order
    
    # num_depth must end up odd. If an even number is provided, num_depth ends up as provided+1.
    num_half_depth = num_depth // 2
    num_depth = 2*num_half_depth + 1
    
    # To calculate numerical derivs, spool out time points around point of interest, calculating associated input values...
    t_temp = np.linspace(t-mod_num_girth, t+mod_num_girth, num_depth)
    input_array = input_lambda(t_temp) # Divide by zeros will probably result in Inf values which could cause derivative issues

    # ... Then calc all num derivs for all spooled time array, but retain only row of interest.
    input_derivs = num_derivs(input_array, t_temp, diff_order)[num_half_depth, :]
    
    return input_derivs