import numpy as np
import scipy.integrate as integ
import sympy as sym
import torch
import torch.autograd as auto

import deepymod_torch.VE_params as VE_params


######## MAJOR FUNCTIONALITY ########

# Data generation using Boltzmann superposition integrals.
def calculate_strain_stress(input_type, time_array, input_expr, E_mods, viscs, D_input_lambda=None):
    '''
    Main function for generating accurate viscoelastic response to provided manipulation for given mechanical model.
    Uses the principle of Boltzmann superposition and as such is only valid for linear viscoelasticity.
    In addition, only GMMs can be used in this framework to calculate responses to strain manipulations and...
    ... only GKMs can be used to calculate responses to stress manipulations.
    As such, the model parameters provided will be interpreted as defining a GMM if the specified input_type is 'Strain' and...
    ... the model parameters provided will be interpreted as defining a GKM if the specified input_type is 'Stress'.
    Solutions are obtained using numerical integration from the SciPy package.
    
    Parameters
        input_type: string
            Must be 'Strain' or 'Stress'. Defines the manipulation type and mechanical model.
        time_array: Nx1 array
            Time series previously defined.
            More time points does not equal greater accuracy but does equal greater computation time.
        input_expr: SymPy Expression OR function
            By default, the analytical expression for the manipulation is defined as a SymPy expression here.
            Alternatively, if the kwarg D_input_lambda does not take its default, a function can be provided that...
            ...returns the result of an analytical definition for the manipulation for a given time point.
        E_mods: list
            The elastic moduli partially defining the mechanical model being manipulated.
            All but the first value are paired with a corresponding viscosity.
        viscs: list
            The viscosities partially defining the mechanical model being manipulated.
            Always one element shorter than E_mods.
        D_input_lambda: function; OPTIONAL
            Returns the result of the first derivative of the expression used to define the manipulation profile for any time point.
            If None, input_expr is assumed to have been symbolically defined and as such...
            ...the derivative is determined in this fashion.
    
    Returns
        strain_array: array of same shape as time_array
        stress_array: array of same shape as time_array
    '''
    
    if D_input_lambda: # Checks if provided
        input_lambda = input_expr
    else: # Default behavior is to reach an analytical expression for the first time derivative of manipulation using SymPy
        t = sym.symbols('t', real=True)
        D_input_expr = input_expr.diff(t)
        
        input_lambda = sym.lambdify(t, input_expr)
        D_input_lambda = sym.lambdify(t, D_input_expr)
    
    # Relaxation and creep functions occupy identical positions in mathematics. Whichever is needed depending on input_type...
    # ... is created as a lambda function with input time, and explicit use of model parameters.
    relax_creep_lambda = relax_creep(E_mods, viscs, input_type)
    
    start_time_point = time_array[0]
    
    integrand_lambda = lambda x, t: relax_creep_lambda(t-x)*D_input_lambda(x) # x is t', or dummy variable of integration.
    integral_lambda = lambda t: integ.quad(integrand_lambda, start_time_point, t, args=(t))[0] # integral to perform at each time point.
    
    output_array = np.array([])
    input_array = np.array([])
    for time_point in time_array:
        # Term outside integral, corrects for discontinuity between assumed zero manipulation history and beginning of here defined manipulation.
        first_term = input_lambda(start_time_point)*relax_creep_lambda(time_point-start_time_point)
        
        # Integral term. Response to here defined manipulation.
        second_term = integral_lambda(time_point)
        
        output_array = np.append(output_array, first_term + second_term)
    
    input_array = input_lambda(time_array)
    
    input_array = input_array.reshape(time_array.shape)
    output_array = output_array.reshape(time_array.shape)
    
    # Purely arrangement of returned objects.
    if input_type == 'Strain':
        strain_array, stress_array = input_array, output_array
    else:
        strain_array, stress_array = output_array, input_array
    
    return strain_array, stress_array


def relax_creep(E_mods, viscs, input_type):
    '''
    Incorporates mechanical model definition and manipulation type into function for kernal within Boltzmann superposition integral.
    Function returned is either that called the relaxation function (input_type='Strain') or the creep function (input_type='Stress'), the result being used analagously.
    If the input_type is 'Strain' then the parameters are assumed to refer to a Maxwell model, whereas
    if the input_type is 'Stress' then the parameters are assumed to refer to a Kelvin model.
    
    Parameters
        E_mods: list
            The elastic moduli partially defining the mechanical model being manipulated.
            All but the first value are paired with a corresponding viscosity.
        viscs: list
            The viscosities partially defining the mechanical model being manipulated.
            Always one element shorter than E_mods.
        input_type: string
            Must be 'Strain' or 'Stress'. Defines the manipulation type and mechanical model.
            
    Returns
        relax_creep_lambda: lambda function
            With single parameter of time.
    '''
    
    # Converted to arrays for easy computation of relevant tau (characteristic times) values
    E_mods_1plus_array = np.array(E_mods[1:]).reshape(-1,1) # So-called 'equillibrium' constant incorporated differently.
    viscs_array = np.array(viscs).reshape(-1,1)
    
    taus = viscs_array/E_mods_1plus_array
    
    if input_type == 'Strain':
        # Relaxation formulation
        relax_creep_lambda = lambda t: E_mods[0] + np.sum(np.exp(-t/taus)*E_mods_1plus_array)
    else: # input_type == 'Stress'
        # Creep formulation
        relax_creep_lambda = lambda t: 1/E_mods[0] + np.sum((1-np.exp(-t/taus))/E_mods_1plus_array)
    
    return relax_creep_lambda


# Data generation from differential equation
def calculate_int_diff_equation(time, response, input_lambda_or_network, coeff_vector, sparsity_mask, library_diff_order, input_type):
    '''
    Alternative function for generating viscoelastic response to provided manipulation for given mechanical model.
    Compared to calculate_strain_stress, this function is more versatile but less accurate.
    Solves differential equation (GDM) directly using numerical methods.
    Not totally from first principles as prior of some initial values of the response are required.
    
    Parameters
        time: Nx1 Tensor OR array (must match response)
            Time series over which response should be calculated.
            More time points BOTH equals greater accuracy and greater computation time.
            Specify as Tensor if graph then exists to response, allowing automatic differentiation to be used when calculating initial values.
            Otherwise numerical derivatives will be used. Tensors are preferred.
        response: Nx1 Tensor OR array (must match time)
            Already defined response series used PURELY for initial values.
        input_lambda_or_network: function OR nn.Module from PyTorch with first output as manipulation fit
            Method of calculating manipulation profile and manipulation derivatives.
            Preferred is function, providing direct analytical description of manipulation, derivatives obtained numerically.
            In case of noisy manipulation, neural network mediated fit can be used instead, with automatic derivatives.
        coeff_vector: Mx1 array OR detached Tensor
            Coefficients partially defining model discovered.
        sparsity_mask: M element array OR detached Tensor
            Mask identifying the terms associated with each coefficient.
        library_diff_order: int
            The maximum order of derivative calculated for both strain and stress to calculate the library in the model discovery process.
            Allows interpretation of sparsity_mask by providing understanding of terms associated with mask values before threshold.
        input_type: string
            Must be 'Strain' or 'Stress'. Unlike calculate_strain_stress, no mechanical model is assumed.
        
    Returns
        calculated_response_array: Nx1 array
    '''
    
    # time and response should be either both tensors, or both arrays.
    if type(time) is torch.Tensor:
        time_array = np.array(time.detach()) # time as tensor retained for initial values
    else: # else numpy array
        time_array = time
    
    coeff_array = np.array(coeff_vector)
    mask_array = np.array(sparsity_mask)
    
    # Function returns coeffs and masks in standardized format such that mask values correspond to diff order etc.
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
    
    # Defines ODE to solve. Function as required by odeint to determine derivative of each variable for a given time point.
    # The 'variables' here are not strain and stress.
    # Each derivative of the response up to and not including the highest is treated as an independant variable from the perspective of odeint.
    # The relationship between them must be specified in the function.
    # The derivatives for the manipulation are independantly calculated.
    def calc_dU_dt(U, t):
        # U is an array of response and increasing orders of derivative of response.
        # t is a time point decided by odeint, it does not necessarily come from time.
        # Returns list of derivative of each input element in U.
        
        # Manipulation derivatives
        if type(input_lambda_or_network) is type(lambda:0):
            # Calculate numerical derivatives of manipulation variable by spooling a dummy time series around t.
            input_derivs = num_derivs_single(t, input_lambda_or_network, max_input_diff_order)
        else: # network
            t_tensor = torch.tensor([t], dtype=torch.float32, requires_grad=True)
            input_derivs = [input_lambda_or_network(t_tensor)[0]] # The [0] here selects the manipulation.
            for _ in range(max_input_diff_order):
                # Calculate automatic derivatives of manipulation variable
                input_derivs += [auto.grad(input_derivs[-1], t_tensor, create_graph=True)[0]]

            input_derivs = np.array([input_deriv.item() for input_deriv in input_derivs])
        
        # Use masks to select manipulation terms ...
        # ...and response terms from function argument, considering ladder of derivative substitions.
        # Concatenate carefully to align with coefficient order.
        input_terms = input_derivs[input_mask]
        response_terms = U[response_mask[:-1]]
        terms_array = np.concatenate((input_terms, response_terms))
        
        # Multiply aligned coeff-term pairs and divide by coeff of highest order deriv of response variable.
        da_dt = np.sum(coeffs_less_dadt_array*terms_array)/response_coeffs[-1]
        
        dU_dt = list(U[1:]) + [da_dt]
        
        return dU_dt
    
    # To avoid edge effects, increasingly pronounced in higher derivatives, initial values are picked a few elements from the extremes.
    start_index = max_response_diff_order
    
    # Initial values of derivatives.
    if type(time) is torch.Tensor:
        # Initial values of response and response derivatives determined using torch autograd.
        IVs = [response[start_index]]
        for _ in range(max_response_diff_order-1):
            IVs += [auto.grad(IVs[-1], time, create_graph=True)[0][start_index]] # result of autograd will have a single non-zero element at start_index

        IVs = [IV.item() for IV in IVs]
        
        # The few skipped values from edge effect avoidance tacked on again - prepped.
        calculated_response_array_initial = np.array(response[:start_index].detach()).flatten()
    else: # else numpy array
        # Initial values of response and response derivatives determined using numpy gradient.
        response_derivs = num_derivs(response, time_array, max_response_diff_order-1)[start_index, :] # Keep only row of start_index
        IVs = list(response_derivs)
        
        # The few skipped values from edge effect avoidance tacked on again - prepped.
        calculated_response_array_initial = response[:start_index].flatten()
    
    # odeint is blind to clipped initial extreme
    reduced_time_array = time_array[start_index:].flatten()
    
    calculated_response_array = integ.odeint(calc_dU_dt, IVs, reduced_time_array)[:, 0] # Want only first column (response) not series for derivatives of response
    
    # The few skipped values from edge effect avoidance tacked on again - done.
    calculated_response_array = np.concatenate((calculated_response_array_initial, calculated_response_array)).reshape(-1, 1)
    
    return calculated_response_array


def align_masks_coeffs(coeff_vector, sparsity_mask, library_diff_order):
    '''
    Restructures given set of coeffs wrt an understanding of the associated terms.
    Result is a coeffs vector and mask vector for each of strain and stress where...
    ...the mask values indicate precisely the order of derivative of the associated term.
    The strain part of this also contains the coeff of 1 associated with the first derivative.
    
    Parameters
        coeff_vector: 1D or 2D array of N elements
            Coefficients partially defining model of interest.
        sparsity_mask: 1D array of N elements
            Mask identifying the terms associated with each coefficient.
        library_diff_order: int
            The maximum order of derivative calculated for both strain and stress to calculate the library of terms.
            Allows interpretation of sparsity_mask by providing understanding of terms associated with mask values.
            
    Returns
        strain_coeffs_mask: 2-tuple
            Tuple like (coeffs, mask) where each element is a 1D array.
        stress_coeffs_mask: 2-tuple
            As strain_coeffs_mask.
    '''
    
    # Create boolean arrays to slice mask into strain and stress parts
    first_stress_mask_value = library_diff_order
    is_strain = sparsity_mask < first_stress_mask_value
    is_stress = sparsity_mask >= first_stress_mask_value
    
    # Slice mask and coeff values and shift stress mask so that mask values always refer to diff order. (Only complete for Stress here.)
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


#Data Validation routine
def equation_residuals(time_array, strain_array, stress_array, coeffs, sparsity_mask='full', diff_order='full'):
    '''
    Quantifies the agreement of a given strain/stress differential equation with a data series at each point in the data series.
    All derivatives specified by the model are calculated numerically and the products of each coefficient and associated term...
    ...are summed or subtracted as appropriate to determine the the degree to which the stated equality is invalid at each point.
    The default behavior is to assume adherence to the GDM structure with no skipped orders of derivative and...
    ...the same highest order of derivative for both strain and stress. In this case, coeffs is understood without the help of the kwargs.
    
    Parameters
        time_array: Nx1 array
            Series of time stamps for each data point. Must be consecutive.
        strain_array: Nx1 array
            Series of strain values for each point in time.
        stress_array: Nx1 array
            Series of stress values for each point in time.
        coeffs: 1D or 2D M element array
            Coefficients partially defining model of interest. Is suffcient to effectively fully define model if no contradictory mask is specified.
        sparsity_mask: 1D M element array; OPTIONAL
             Mask identifying the terms associated with each coefficient.
        diff_order: int; OPTIONAL
            The maximum order of derivative calculated for both strain and stress to calculate the library of terms.
            Allows interpretation of sparsity_mask by providing understanding of terms associated with mask values.
        
    Returns
        residuals: Nx1 array
    '''
    
    # If default, the mask and diff_order appropriate to coeffs adhering to a GDM is generated.
    if diff_order == 'full': # this and sparsity_mask should either both be default, or both be specified.
        sparsity_mask = np.arange(len(coeffs))
        diff_order = len(coeffs)//2
    
    # In case they are tensors. Tensors must still be detached as arguements.
    time_array, strain_array, stress_array = np.array(time_array), np.array(strain_array), np.array(stress_array)
    coeffs, sparsity_mask = np.array(coeffs, dtype=float), np.array(sparsity_mask)
    
    # Function returns coeffs and masks in standardized format such that mask values correspond to diff order etc.
    strain_coeffs_mask, stress_coeffs_mask = align_masks_coeffs(coeffs, sparsity_mask, diff_order)
    strain_coeffs, strain_mask = strain_coeffs_mask
    stress_coeffs, stress_mask = stress_coeffs_mask
    
    # strain coeff-term products will be subtracted and so the negatives of the strain coeffs are used allowing everything to be summed.
    coeffs_array = np.concatenate((-strain_coeffs, stress_coeffs)).reshape(-1,1)
    
    # Calculate all numerical derivatives for full library (all derivatives at each point in time series).
    strain_theta = num_derivs(strain_array, time_array, diff_order)
    stress_theta = num_derivs(stress_array, time_array, diff_order)
    
    # Build sparse library only including derivatives specified by masks.
    num_theta = np.concatenate((strain_theta[:, strain_mask], stress_theta[:, stress_mask]), axis=1)
    
    # Matrix multiplication to calculate all coeff-term products and sum at each time point.
    residuals = num_theta @ coeffs_array
        
    return residuals


# Numerical derivatives using NumPy
def num_derivs(dependent_data, independent_data, diff_order):
    '''
    Utility function for calculating increasing orders of numerical derivatives for a given independant and dependant data series.
    
    Parameters
        dependent_data: 1D N or 2D Nx1 array
            Data corresponding to the independant values at each point.
        independent_data: 1D N or 2D Nx1 array
            Derivatives will be calculated across this range. Must be consecutive values.
        diff_order: int
            Specified maximum order of derivative to be calculated and incorporated into returned array.
        
    Returns
        data_derivs: Nx(diff_order+1) array
            Includes zeroth order of derivative (dependent_data) as first column in matrix returned.
    '''
    
    data_derivs = dependent_data.reshape(-1, 1)
    
    # Calculate array of derivatives and append as additional column to previous array to build up matrix to return.
    # Recursively calculate derivatives on previous derivatives to acheive higher order derivatives.
    for _ in range(diff_order):
        data_derivs = np.append(data_derivs, np.gradient(data_derivs[:, -1].flatten(), independent_data.flatten()).reshape(-1,1), axis=1)
    
    return data_derivs


def num_derivs_single(t, input_lambda, diff_order, num_girth=1, num_depth=101):
    '''
    Calculates numerical derivatives for a single point on a defined curve.
    num_derivs relies on a detailed series of points to accurately calculate derivatives numerically, especially higher derivatives.
    If an analytical expression is known, and derivatives are required for a single independant point, this method spools out...
    ...a series of points on the curve defined by the expression around the single point to calculate derivatives at this point.
    If an analytical expression for each order of derivative desired is known, this method is unnecessary and relatively inaccurate.
    However if these expressions are not known, this method can still be used.
    
    Parameters
        t: float
            The independant time point for which the derivatives are desired.
        input_lambda: function (1->1)
            Returns result of evaluating analytical expression describing curve for which derivatives will be calculated.
        diff_order: int
            Specified maximum order of derivative to be calculated and incorporated into returned array.
        num_girth: int; OPTIONAL
            Increasing improves accuracy of derivatives at cost of computation time.
            Specifies absolute range around t to evaluate input_lambda.
            Is modified by diff_order for real range used.
        num_depth: int; OPTIONAL
            Increasing improves accuracy of derivatives at cost of computation time.
            Specifies number of points within range for which evaluation of input_lambda is performed.
            Should be odd, but will be adjusted if not.
        
    Returns
        input_derivs: 1D (diff_order+1) element array
            Includes zeroth order of derivative (input_lambda(t)) as first element in vector returned.
    '''
    
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







######## EXTENDED FUNCTIONALITY ########

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
