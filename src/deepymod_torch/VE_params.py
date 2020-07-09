import numpy as np
import sympy as sym
import IPython.display as dis


def model_params_from_coeffs(coeff_vector, model, print_expressions=False): 
    '''
    Handles symbolic conversion from coefficients within GDM to meaningful mechanical constants of mechanical model of choice.
    Reverse process of coeffs_from_model_params but much more challenging.
    Will only work as expected if coeff_vector provides coefficients that form part of the description of a model that...
    ...conforms to the pattern of a GDM precisely.
    Function does not read mask to interpret validity of attempt and simply exceptions or unending rounds of computation can occur.
    Currently only tested up to 2nd order GDMs. Computation difficulty increases significantly with order.
    
    Parameters
        coeff_vector: N or Nx1 array
            Must conform to GDM structure. Should not include coefficient of the first derivative of strain.
        model: string
             Must be either 'GMM' or 'GKM'. Specifies framework of interpretation. Usually, either is possible.
        print_expressions: bool; OPTIONAL
            SymPy expressions can be displayed to display the simultaneous equations to solve.
            
    Returns
        model_params_value_list: N element list
            The values associated with the mechanical parameters arrived at.
            Will always be all elastic moduli, then all viscosities.
            The number of viscosities is one fewer than the number of elastic moduli.
        model_params_mask_list: N element list
            SymPy symbols associated with each value returned in model_params_value_list.
            
    Notes
        A minimal successful attempt to solve should in theory contain a number of solutions equal to A! where ...
        ... A is the maximum derivative order in the GDM. This is because for a set of mechanical units, each ...
        ... containing E_i and eta_i, there is no set order in which to define them.
        The max order is equal to the number of units. These solutions will be redundant with regards to each other.
    '''
    
    # Easy catch for non-conformity with GDM
    if len(coeff_vector) % 2 == 0:
        raise ValueError('No viable mech model discoverable from an even number of coefficients.')
    
    order = len(coeff_vector)//2
    
    # Returns expressions for each coefficient specified model complexity and type.
    # Expressions are in simple or 'natural' form, with no symbolic denominators.
    # This means the coefficient of the first derivative of strain is not 1.
    # There will be one more expression than there are elements in coeff_vector.
    if model == 'GMM':
        coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(order)
    else: # model == 'GKM'
        coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(order)
    
    # Treat the simultaneous equation problem to be more amenable to SymPy by ...
    # ... recognising that coefficients with assumed constraint that the coefficient of the first derivative of strain is 1 (coeff_vector) are ...
    # ... equal to the 'natural' expressions for each coefficient divided by the expression for the 'natural' coefficient of the first derivative of strain.
    # And therefore that the coefficients with assumed constraint multiplied by the 'natural' coefficient of the first derivative of strain are ...
    # ... equal to the 'natural' expressions for each coefficient.
    # Then explicitely include the value 1 as the coefficient of the first derivative of strain ...
    # ... and solve as a series of equations with one additional symbol, and one additional equation.
    coeff_vector = coeff_vector[0:1] + [1] + coeff_vector[1:]
    natural_strain_t_coeff_sym = sym.symbols('c^n_{\epsilon_t}', real=True, positive=True) # Additional symbol. Represents the 'natural' coefficient of the first derivative of strain.
    coeff_expression_list = [sym.simplify(coeff_expression) for coeff_expression in coeff_expression_list]
        
    # Form list of simultaneous expressions to solve.
    coeff_equations_list = [sym.Eq(coeff_expression, coeff_value*natural_strain_t_coeff_sym) for coeff_expression, coeff_value in zip(coeff_expression_list, coeff_vector)]
    
    # Use of dis.display to maximize readability of displayed simultaneous equations.
    if print_expressions:
        dis.display(*coeff_equations_list)
            
    # When solving, additional symbol to evaluate is added just in time.
    # Returned is a list of tuples, each tuple being a different solution. If no solution, returns empty list.
    model_params_value_list = sym.solve(coeff_equations_list, model_params_mask_list + [natural_strain_t_coeff_sym])
    
    # No need to report value of natural_strain_t_coeff_sym
    model_params_value_list = [model_params_values[:-1] for model_params_values in model_params_value_list]
    
    if len(model_params_value_list) == 0:
        model_params_value_list = [()] # Preserve structure for indexing consistency
        print('No solution possible for coefficient values and model complexity arrived at.')
    
    # Note, the returned solution may still contain 1 (or more?) symbols. This does not mean sym.solve() has failed.
    # Instead, it means that more than 1 set of model params can give rise to the given coeffs.
    # The symbol(s) remaining describe the common pattern in the model params for the same coeffs.
    return model_params_value_list, model_params_mask_list


def coeffs_from_model_params(E_mod_list, visc_list, model, print_expressions=False):
    '''
    Handles symbolic conversion from meaningful mechanical constants of mechanical model of choice to coefficients within GDM.
    Reverse process of model_params_from_coeffs but much simpler.
    
    Parameters
        E_mod_list: list
            list of values of elastic moduli partially defining mechanical model.
            Should be one longer than visc_list as first elastic modulus corresponds to 'equillibrium' spring.
        visc_list: list
            list of values of viscosities partially defining mechanical model.
        model: string
             Must be either 'GMM' or 'GKM'. Specifies framework of interpretation.
        print_expressions: bool; OPTIONAL
            SymPy expressions can be displayed to display the equations into which the appropriate substitutions are made.

    Returns
        coeff_value_list: list
            GDM coefficients under constraint of the first derivative of strain having a coefficient of 1.
            list will have length equal to the number of elastic moduli + the number of viscosities.
    '''
    
    order = len(visc_list)
    
    # Returns expressions for each coefficient for specified model complexity and type.
    # Expressions are in simple or 'natural' form, with no symbolic denominators.
    # This means the coefficient of the first derivative of strain is not 1.
    # There will be one more expression than the number of coefficients sought.
    if model == 'GMM':
        coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(order)
    else: # model == 'GKM'
        coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(order)
    
    # Fix coeff of first derivative of strain to 1 by ...
    # ... dividing all 'natural' coefficients, including that of the first derivative of strain, ...
    # ... by the expression for the coefficient of the first derivative of strain.
    # Arrive at list with one less expression.
    coeff_expression_list = [sym.simplify(coeff_expression/coeff_expression_list[1]) for i, coeff_expression in enumerate(coeff_expression_list) if i != 1]
    
    if print_expressions:
        dis.display(*coeff_expression_list)
    
    # Substitute in all mechanical values into expressions.
    coeff_value_list = [coeff_expression.subs(zip(model_params_mask_list, E_mod_list+visc_list)) for coeff_expression in coeff_expression_list]
    
    return coeff_value_list


# KELVIN MODEL

def kelvin_coeff_expressions(order):
    '''
    Arranges symbolic expressions for each coefficient in a GDM based of of symbols referring to a generalized Kelvin model.
    Expressions are in simple or 'natural' form, with no symbolic denominators.
    This means the coefficient of the first derivative of strain is not 1.
    
    Parameters
        order: int
            Maximum order of derivative in GDM OR the number of viscosities in the GKM, both are equivalent.
            
    Returns
        coeff_expressions_list: list
            list contains SymPy expressions for each coefficient. The number of elements will be equal to (order+1)*2
            The first expressions refer to the coefficients of increasing order of derivative for strain.
            The second half of expressions refer to the equivalent for stress.
        model_params_list: list
            Symbol objects for all the assumed mechanical parameters incorporated into the expressions in coeff_expressions_list.
            There will be one less symbol than there are elements in coeff_expressions_list.
            The first symbols are all the elastic moduli will the next symbols are all the viscosities.
            There will always be one more elastic modulus beyond the number of viscosities.
    '''
    
    # Create and organise Kelvin general equation relating total strain and total stress.
    summation, model_params_list = kelvin_sym_sum(order) # Majority of expression created here.
    
    # Additional symbol for lone parameter not following 'sum' pattern.
    E_0 = sym.symbols('E^K_0', real=True, positive=True)
    model_params_list = [E_0] + model_params_list # Added to symbols list
    RHS = 1/E_0 + summation # Complete expression
    
    # Arrange into form where coeffcients of each term can be extracted.
    # The combination of .together(), sym.fraction() and .expand() allow for the coeffcients to be seperated.
    # .together() pulls all of RHS into one fraction by finding a common denominator
    # sym.fraction() seperates numerator from denominator
    # .expand() prevents factorisation from obscuring full expressions multiplying each term.
    # Formally strain = RHS*stress. There is no need to define symbols for strain and stress however if the two sides of the equality are identified.
    # The numerator is associated with all stress terms and the denominator with all strain terms.
    RHS = RHS.together()
    stress_side, strain_side = sym.fraction(RHS)
    strain_side, stress_side = strain_side.expand(), stress_side.expand()
    
    # Symbol stands in for differential operator.
    dt = sym.symbols('dt')
    
    # Identify expressions for each coeff. Each power on 'dt' is a higher order derivative.
    # Strain coeff expressions listed first.
    coeff_expressions_list = [strain_side.coeff(dt, strain_order) for strain_order in range(order+1)]
    coeff_expressions_list += [stress_side.coeff(dt, stress_order) for stress_order in range(order+1)]
    
    return coeff_expressions_list, model_params_list


def kelvin_sym_sum(order):
    '''
    Builds the majority of the expression relating total stress and total strain in a generalized Kelvin model.
    The part built does not involve the lone spring, but all units are incorporated by a symbolic sum with a pattern.
    
    Parameters
        order: int
            Maximum order of derivative in GDM OR the number of viscosities in the GKM, both are equivalent.
            
    Returns
        Expression: SymPy Expression
            Main result, expression incorporating all symbols needed for a GKM of complexity specified by order, save for the symbol for the lone spring.
        all_syms: list
            Symbol objects for all the assumed mechanical parameters incorporated into Expression.
            The first symbols are all the elastic moduli save for the lone elastic modulus while the next symbols are all the viscosities.
            There will always be the same number of elastic moduli as viscosities.
    '''
    
    # Definition of all symbols incorporated into this expression.
    # K superscript refers to 'Kelvin'.
    E_Syms = [sym.symbols(f'E^K_{Branch_Index}', real=True, positive=True) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols(f'eta^K_{Branch_Index}', real=True, positive=True) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    # Symbol stands in for differential operator.
    dt = sym.symbols('dt')
    
    # Builds symbolic sum
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(E_Syms[Branch_Index] + Eta_Syms[Branch_Index]*dt)
        
    return Expression, all_syms


# MAXWELL MODEL

def maxwell_coeff_expressions(order):
    '''
    Arranges symbolic expressions for each coefficient in a GDM based of of symbols referring to a generalized Maxwell model.
    Expressions are in simple or 'natural' form, with no symbolic denominators.
    This means the coefficient of the first derivative of strain is not 1.
    
    Parameters
        order: int
            Maximum order of derivative in GDM OR the number of viscosities in the GMM, both are equivalent.
            
    Returns
        coeff_expressions_list: list
            list contains SymPy expressions for each coefficient. The number of elements will be equal to (order+1)*2
            The first expressions refer to the coefficients of increasing order of derivative for strain.
            The second half of expressions refer to the equivalent for stress.
        model_params_list: list
            Symbol objects for all the assumed mechanical parameters incorporated into the expressions in coeff_expressions_list.
            There will be one less symbol than there are elements in coeff_expressions_list.
            The first symbols are all the elastic moduli will the next symbols are all the viscosities.
            There will always be one more elastic modulus beyond the number of viscosities.
    '''
        
    # Create and organise Kelvin general equation relating total strain and total stress.
    summation, model_params_list = maxwell_sym_sum(order)
    
    # Additional symbol for lone parameter not following 'sum' pattern.
    E_0 = sym.symbols('E^M_0', real=True, positive=True)
    model_params_list = [E_0] + model_params_list # Added to symbols list
    RHS = E_0 + summation # Complete expression
    
    # Arrange into form where coeffcients of each term can be extracted.
    # The combination of .together(), sym.fraction() and .expand() allow for the coeffcients to be seperated.
    # .together() pulls all of RHS into one fraction by finding a common denominator
    # sym.fraction() seperates numerator from denominator
    # .expand() prevents factorisation from obscuring full expressions multiplying each term.
    # Formally stress = RHS*strain. There is no need to define symbols for strain and stress however if the two sides of the equality are identified.
    # The numerator is associated with all strain terms and the denominator with all stress terms.
    RHS = RHS.together()
    strain_side, stress_side = sym.fraction(RHS)
    strain_side, stress_side = strain_side.expand(), stress_side.expand()

    # Symbol stands in for differential operator.
    dt = sym.symbols('dt')
    
    # Identify expressions for each coeff. Each power on 'dt' is a higher order derivative.
    # Strain coeff expressions listed first.
    coeff_expressions_list = [strain_side.coeff(dt, strain_order) for strain_order in range(order+1)]
    coeff_expressions_list += [stress_side.coeff(dt, stress_order) for stress_order in range(order+1)]
    
    return coeff_expressions_list, model_params_list


def maxwell_sym_sum(order):
    '''
    Builds the majority of the expression relating total stress and total strain in a generalized Maxwell model.
    The part built does not involve the lone spring, but all units are incorporated by a symbolic sum with a pattern.
    
    Parameters
        order: int
            Maximum order of derivative in GDM OR the number of viscosities in the GMM, both are equivalent.
            
    Returns
        Expression: SymPy Expression
            Main result, expression incorporating all symbols needed for a GMM of complexity specified by order, save for the symbol for the lone spring.
        all_syms: list
            Symbol objects for all the assumed mechanical parameters incorporated into Expression.
            The first symbols are all the elastic moduli save for the lone elastic modulus while the next symbols are all the viscosities.
            There will always be the same number of elastic moduli as viscosities.
    '''
    
    # Definition of all symbols incorporated into this expression.
    # M superscript refers to 'Maxwell'.
    E_Syms = [sym.symbols('E^M_'+str(Branch_Index), real=True, positive=True) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols('eta^M_'+str(Branch_Index), real=True, positive=True) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    # Symbol stands in for differential operator.
    dt = sym.symbols('dt')
    
    # Builds symbolic sum
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(1/E_Syms[Branch_Index] + 1/(Eta_Syms[Branch_Index]*dt))
        
    return Expression, all_syms



# CONVERT BETWEEN MODELS

def convert_between_models(E_mod_list, visc_list, origin_model, print_expressions=False):
    '''
    Mostly a convenience function for finding an equivalent description of a GKM based off a GMM, or vice versa.
    Converts from GMM/GKM (mech params) -> GDM (coefficients) -> GKM/GMM (alternate mech params)
    
    Parameters
        E_mod_list: list
            list of values of elastic moduli partially defining mechanical model.
            Should be one longer than visc_list as first elastic modulus corresponds to 'equillibrium' spring.
        visc_list: list
            list of values of viscosities partially defining mechanical model.
        origin_model: string
             Must be either 'GMM' or 'GKM'. Specifies framework of interpretation for the mechanical model converting from.
             Thus also defines model converting to.
        print_expressions: bool; OPTIONAL
            Turns on or off the display of equations for substitution and solving in called functions, as well as ...
            ... adding some guiding text to structure what is displayed.
            
    Returns
        solutions: list
            Elements are each 2-tuples containing the elastic moduli and viscosities for each solution found during conversion.
            Elastic moduli and viscosities are themselves tuples, with the tuple of viscosities one shorter than the tuple of elastic moduli.
        dest_model_syms: list
            Symbol objects for all mechanical parameters given values in solutions.
            Length is always equal to the combined lengths of E_mod_list and visc_list.
    '''
    
    dest_model = 'GKM' if origin_model == 'GMM' else 'GMM'

    if print_expressions:
        print(f'Universal coefficients from {origin_model} parameters:')
    
    # Convert from mechanical parameters to coefficients
    coeff_value_list = coeffs_from_model_params(E_mod_list, visc_list, origin_model, print_expressions=print_expressions)
    
    if print_expressions:
        print(f'{dest_model} parameters from universal coefficients:')
    
    # Convert from coefficients to mechanical parameters for destination mechanical model.
    dest_model_values, dest_model_syms = model_params_from_coeffs(coeff_value_list, dest_model, print_expressions=print_expressions)
    
    # For each tupled element of dest_model_values, split into 2-tuple of tuples containing seperated elastic moduli and viscosities.
    solutions = [(solution[:len(E_mod_list)], solution[len(E_mod_list):]) for solution in dest_model_values]
    
    return solutions, dest_model_syms


# SCALE COEFFS DUE TO 'NORMALISATION'

def scaled_coeffs_from_true(true_coeffs, time_sf, strain_sf, stress_sf):
    '''
    Converts coefficients within a GDM to those that would need to be true to explain data series that have been scaled.
    When stress, strain and time are all scaled, the values of the terms at each point is predictably altered.
    Without changing the coefficients, the equality described by the GDM will no longer be true.
    If each coefficient is adjusted to reverse the affect of the changed terms, the equality will be recovered.
    Beyond this, the coefficient of the first derivative of strain must be reconstrained to 1, so all coefficients must effectively be divided by this new value.
    This recovers a set of coefficients that should be discovered from the scaled data series.
    
    Parameters
        true_coeffs: N or Nx1 array
            The coefficients known to be accurate before the scale factors that follow were applied.
            Coefficients begin with those associated increasing orders of derivative of strain ...
            ... and then those associated with increasing orders of derivative of stress.
        time_sf: float
            Scale factor applied to the time series for the viscoelastic data set.
        strain_sf: float
            As time_sf but for strain.
        stress_sf: float
            As time_sf but for stress.
    
    Returns
        list(scaled_coeff_guess): list with N elements
            New coeffcients that describe a GDM that explains the data series after the scale factors are applied.
            The structure is identical to true_coeffs besides being a list.
    '''
    
    true_coeffs_array = np.array(true_coeffs).flatten()
    
    # array of all scale factors (different for each coeff) needed to reverse the affect that scaling will have on calculated terms within the GDM.
    alpha_array = coeff_scaling_values(true_coeffs_array, time_sf, strain_sf, stress_sf)
    
    # piece-wise multiplication with original coeffs.
    scaled_coeff_guess = true_coeffs_array*alpha_array
    
    return list(scaled_coeff_guess)


def true_coeffs_from_scaled(scaled_coeffs, time_sf, strain_sf, stress_sf, mask='full', library_diff_order='full'):
    '''
    Converts coefficients within a GDM to those that would need to be true to explain data series that has had a stated scaling reversed.
    A less rigid reversal of the process from scaled_coeffs_from_true(...).
    Differs from scaled_coeffs_from_true in that assumption regarding conformity with the GDM structure can be overridden.
    The sparsity mask and maximum derivative order in the library are used to identify which terms are associated with which coefficients ...
    ... so that the scaling occurs appropriately. 
    Default behavior is to assume that a GDM structure is present.
    
    Parameters
        scaled_coeffs: N or Nx1 array
            The coefficients discovered or calculated for data series after the scale factors that follow were applied.
            Coefficients begin with those associated increasing orders of derivative of strain ...
            ... and then those associated with increasing orders of derivative of stress.
        time_sf: float
            Scale factor applied to the time series for the viscoelastic data set.
        strain_sf: float
            As time_sf but for strain.
        stress_sf: float
            As time_sf but for stress.
        mask: N or Nx1 array; OPTIONAL
            Identifies the terms associated with each coefficient.
        library_diff_order: int; OPTIONAL
            The maximum order of derivative calculated for both strain and stress to calculate the library in the model discovery process.
            Allows interpretation of mask by providing understanding of which mask values begin to correspond to stress terms.
    
    Returns
        list(true_coeffs): list with N elements
            Coeffcients that describe a GDM that explains the data series after the effects of the scale factors are reversed.
            The structure is identical to scaled_coeffs besides being a list.
    '''
    
    scaled_coeffs_array = np.array(scaled_coeffs).flatten()
    if mask is not 'full':
        mask = np.array(mask).flatten()
        # library_diff_order allows knowledge of the full potential mask if all terms in the library were present.
        # If mask values are missing, the associated terms effectively have coeffcients of 0.
        # Returned by the function below are coeffs with these zeros explicitely resupplied.
        # These zeros will remain zeros in the next steps but it is a convenient way to ensure the correct scale factors are applied to the correct coefficients.
        scaled_coeffs_array = include_zero_coeffs(scaled_coeffs_array, mask, library_diff_order)
    
    # array of all scale factors (different for each coeff) needed to reverse the affect that FORWARD scaling had on calculated terms within the GDM.
    alpha_array = coeff_scaling_values(scaled_coeffs_array, time_sf, strain_sf, stress_sf)
    
    # piece-wise division for coeffs found for scaled data series.
    true_coeffs = scaled_coeffs_array/alpha_array
    
    # removes zero elements if any present due to non-full mask
    true_coeffs = true_coeffs[true_coeffs != 0]
    
    return list(true_coeffs)


def coeff_scaling_values(coeffs, time_sf, strain_sf, stress_sf):
    '''
    Calculates the 'alpha array' to scale coefficients based on the scaling of the associated data series ...
    ... such that the equation residuals are unaffected.
    This function assumes coefficients have been provided in a format that can be interpreted as a GDM.
    
    Paramters
        coeffs: N or Nx1 array
            The coefficients discovered or calculated for data series after the scale factors that follow were applied.
            Coefficients begin with those associated increasing orders of derivative of strain ...
            ... and then those associated with increasing orders of derivative of stress.
            This function is only interested in the shape of this array.
        time_sf: float
            Scale factor applied to the time series for the viscoelastic data set.
        strain_sf: float
            As time_sf but for strain.
        stress_sf: float
            As time_sf but for stress.
    
    Returns
        alpha_array: array of same shape as coeffs
            The scale factors necassary to apply aligned with the coeffcient in the equivalent array, coeffs.
            
    Notes
        For a derivative of a certain order, the magnitude of this term will be larger by the factor applied to the dependant variable (call t) ...
        ... and smaller by the factor applied to the independant variable (call y), this factor applied a number of times equal to the order.
        To make the coefficient term pair unchanged by the scaled data series, the reverse of this must occur to the coefficient.
        Hence, alpha, the scale factor for the coeff is (t_sf)^diff_order / y_sf where _sf means scale factor.
        However, The coefficient of the first derivative of strain must be 1, so all alpha so far calculated ...
        ... must be divided by the alpha for the coefficient of this term, alpha_LHS.
    '''
    
    middle_index = len(coeffs)//2

    # calculate alpha_n for each term on RHS. Prepare at initial scale factor of 1.
    alpha_n_array = np.ones(coeffs.shape)

    # split into subarrays that modify in place original due to mutability of arrays
    # but allows for easier indexing.
    strain_subarray = alpha_n_array[:middle_index]
    stress_subarray = alpha_n_array[middle_index:]

    # apply dependant variable scaling of alpha_n
    strain_subarray /= strain_sf
    stress_subarray /= stress_sf

    # apply independant variable scaling of alpha_n
    # No such scaling for zeroth order derivatives (index 0)
    # alpha_LHS handled seperately so only 1 first order derivative
    # Higher order derivatives have more consistant pattern hence loop.
    stress_subarray[1] *= time_sf
    for idx in range(2, middle_index + 1):
        strain_subarray[idx-1] *= time_sf**idx
        stress_subarray[idx] *= time_sf**idx

    alpha_LHS = time_sf/strain_sf

    alpha_array = alpha_n_array/alpha_LHS
    
    return alpha_array


def include_zero_coeffs(coeff_vector, sparsity_mask, library_diff_order):
    '''
    Terms eliminated from the library will not have associated coeffs. They effectively have coeffcients of 0.
    This function explicitely fills these zero coefficients into the coeff_vector.
    
    Parameters
        coeff_vector: N or Nx1 array
            The coefficients discovered or calculated for data series after the scale factors that follow were applied.
            Coefficients begin with those associated increasing orders of derivative of strain ...
            ... and then those associated with increasing orders of derivative of stress.
        sparsity_mask: N or Nx1 array
            Identifies the terms associated with each coefficient.
        library_diff_order: int
            The maximum order of derivative calculated for both strain and stress to calculate the library in the model discovery process.
            Allows interpretation of mask by providing understanding of which mask values begin to correspond to stress terms.
    
    Returns
        full_coeff_array: M or Mx1 array with M = 2*library_diff_order + 1
            This array still does not include a coefficient for the first derivative of strain.
            Zero coeffs are inserted into coeff_vector wherever an effective coeff of 0 is present for teh full library of terms.
    '''
    
    full_coeff_list = list(coeff_vector)
    # The generator used in this loop is the full sparsity mask for the entire library
    for term_id in range(library_diff_order*2 + 1):
        if term_id not in sparsity_mask: # If term eliminated
            full_coeff_list.insert(term_id, 0)
            
    full_coeff_array = np.array(full_coeff_list)
    
    return full_coeff_array