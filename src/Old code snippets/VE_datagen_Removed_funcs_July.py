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