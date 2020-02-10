import sympy as sym


# KELVIN MODEL

def model_params_from_coeffs_kelvin(coeff_vector, print_expressions=False):
    
    coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(len(coeff_vector))
    
    if print_expressions:
        print(coeff_expression_list)
    
    coeff_equations_list = [coeff_expression - coeff_value for coeff_expression, coeff_value in zip(coeff_expression_list, coeff_vector)]
    
    model_params_value_list = sym.solve(coeff_equations_list, model_params_mask_list)
    
    if len(model_params_value_list) == 0:
        print('No solution possible for coefficient values and model complexity arrived at.')
    
    return model_params_value_list, model_params_mask_list


def coeffs_from_model_params_kelvin(E_mod_list, visc_list, print_expressions=False):
    
    coeff_count = len(E_mod_list)*2 - 1
    
    coeff_expression_list, model_params_mask_list = kelvin_coeff_expressions(coeff_count)
    
    if print_expressions:
        print(coeff_expression_list)
    
    coeff_value_list = [coeff_expression.subs(zip(model_params_mask_list, E_mod_list+visc_list)) for coeff_expression in coeff_expression_list]
    
    return coeff_value_list


def kelvin_coeff_expressions(coeff_count):
    
    order = coeff_count // 2
    
    eps, sig = sym.symbols('epsilon,sigma', real=True)
    E_0 = sym.symbols('E_0', real=True, negative=False)
    
    #Create and organise Kelvin general equation
    summation, model_params_list = kelvin_sym_sum(order)
    model_params_list = [E_0] + model_params_list
    RHS = (1/E_0 + summation)*sig
    RHS = sym.together(RHS)
    RHS, denom = sym.fraction(RHS)
    
    full_expression = RHS - eps*denom
    
    #find coeffs - prep
    expanded = sym.expand(full_expression)
    dt = sym.symbols('dt')
    
    #fix expressions by making coefficient of first derivative of strain equal to 1
    Strain_t_coeff_expr = -expanded.coeff(eps, 1).coeff(dt, 1)
    
    #Strain coeffs
    coeff_expression = -expanded.coeff(eps, 1).coeff(dt, 0)/Strain_t_coeff_expr
    coeff_expressions_list = [sym.simplify(coeff_expression)]
    for strain_order in range(2, order+1):
        coeff_expression = -expanded.coeff(eps, 1).coeff(dt, strain_order)/Strain_t_coeff_expr
        coeff_expressions_list += [sym.simplify(coeff_expression)]
        
    #Stress coeffs
    for stress_order in range(order+1):
        coeff_expression = expanded.coeff(sig, 1).coeff(dt, stress_order)/Strain_t_coeff_expr
        coeff_expressions_list += [sym.simplify(coeff_expression)]
    
    return coeff_expressions_list, model_params_list


def kelvin_sym_sum(order):
    
    E_Syms = [sym.symbols('E_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols('eta_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    dt = sym.symbols('dt')
    
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(E_Syms[Branch_Index] + Eta_Syms[Branch_Index]*dt)
        
    return Expression, all_syms


# MAXWELL MODEL

def model_params_from_coeffs_maxwell(coeff_vector, print_expressions=False):
    
    coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(len(coeff_vector))
    
    if print_expressions:
        print(coeff_expression_list)
    
    coeff_equations_list = [coeff_expression - coeff_value for coeff_expression, coeff_value in zip(coeff_expression_list, coeff_vector)]
    
    model_params_value_list = sym.solve(coeff_equations_list, model_params_mask_list)
    
    if len(model_params_value_list) == 0:
        print('No solution possible for coefficient values and model complexity arrived at.')
    
    return model_params_value_list, model_params_mask_list


def coeffs_from_model_params_maxwell(E_mod_list, visc_list, print_expressions=False):
    
    coeff_count = len(E_mod_list)*2 - 1
    
    coeff_expression_list, model_params_mask_list = maxwell_coeff_expressions(coeff_count)
    
    if print_expressions:
        print(coeff_expression_list)
    
    coeff_value_list = [coeff_expression.subs(zip(model_params_mask_list, E_mod_list+visc_list)) for coeff_expression in coeff_expression_list]
    
    return coeff_value_list


def maxwell_coeff_expressions(coeff_count):
    
    order = coeff_count // 2
    
    eps, sig = sym.symbols('epsilon,sigma', real=True)
    E_0 = sym.symbols('E_0', real=True, negative=False)
    
    #Create and organise Kelvin general equation
    summation, model_params_list = maxwell_sym_sum(order)
    model_params_list = [E_0] + model_params_list
    RHS = (E_0 + summation)*eps
    RHS = sym.together(RHS)
    RHS, denom = sym.fraction(RHS)
    
    full_expression = RHS - sig*denom
    
    #find coeffs - prep
    expanded = sym.expand(full_expression)
    dt = sym.symbols('dt')
    
    #fix expressions by making coefficient of first derivative of strain equal to 1
    Strain_t_coeff_expr = -expanded.coeff(eps, 1).coeff(dt, 1)
    
    #Strain coeffs
    coeff_expression = -expanded.coeff(eps, 1).coeff(dt, 0)/Strain_t_coeff_expr
    coeff_expressions_list = [sym.simplify(coeff_expression)]
    for strain_order in range(2, order+1):
        coeff_expression = -expanded.coeff(eps, 1).coeff(dt, strain_order)/Strain_t_coeff_expr
        coeff_expressions_list += [sym.simplify(coeff_expression)]
        
    #Stress coeffs
    for stress_order in range(order+1):
        coeff_expression = expanded.coeff(sig, 1).coeff(dt, stress_order)/Strain_t_coeff_expr
        coeff_expressions_list += [sym.simplify(coeff_expression)]
    
    return coeff_expressions_list, model_params_list


def maxwell_sym_sum(order):
    
    E_Syms = [sym.symbols('E_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    Eta_Syms = [sym.symbols('eta_'+str(Branch_Index), real=True, negative=False) for Branch_Index in range(1, order+1)]
    
    all_syms = E_Syms + Eta_Syms
    
    dt = sym.symbols('dt')
    
    Expression = sym.S(0)
    for Branch_Index in range(len(E_Syms)):
        Expression += 1/(1/E_Syms[Branch_Index] + 1/(Eta_Syms[Branch_Index]*dt))
        
    return Expression, all_syms