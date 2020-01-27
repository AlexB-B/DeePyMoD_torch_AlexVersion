import torch


def scaling(weight_vector, library, time_deriv):
    '''
    Rescales the weight vector according to vec_rescaled = vec * |library|/|time_deriv|.
    Columns in library correspond to elements of weight_vector.

    Parameters
    ----------
    weight_vector : tensor of size (Mx1).
        The weight vector to be rescaled.
    library : tensor of size (NxM)
        The library matrix used to rescale weight_vector.
    time_deriv : tensor of size (Nx1)
        The time derivative vector used to rescale weight_vector.

    Returns
    -------
    tensor of size (Mx1)
        Rescaled weight vector.
    '''

    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(library, dim=0)[:, None]
    scaled_weight_vector = weight_vector * (scaling_theta / scaling_time)
    return scaled_weight_vector


def threshold(scaled_coeff_vector, coeff_vector, sparsity_mask, library_config):
    '''
    Performs thresholding of coefficient vector based on the scaled coefficient vector.
    Components greater than the standard deviation of scaled coefficient vector are maintained, rest is set to zero.
    Also returns the location of the maintained components.

    Parameters
    ----------
    scaled_coeff_vector : tensor of size (Mx1)
        scaled coefficient vector, used to determine thresholding.
    coeff_vector : tensor of size (Mx1)
        coefficient vector to be thresholded.

    Returns
    -------
    tensor of size (Nx1)
        vector containing remaining values of weight vector.
    tensor of size(N)
        tensor containing index location of non-zero components.
    '''
    
    reduced_sparse_coeff_vector = torch.where(torch.abs(scaled_coeff_vector) > torch.std(scaled_coeff_vector, dim=0), coeff_vector, torch.zeros_like(scaled_coeff_vector))
    Indices_To_Keep = torch.nonzero(reduced_sparse_coeff_vector)[:, 0]
    Overode = False
    if library_config.get('input_type', None) == ('Strain' or 'Stress'):        
        sparsity_mask_trial = sparsity_mask[Indices_To_Keep]
        if check_need_overide(sparsity_mask_trial, library_config['diff_order']):
            Indices_To_Keep = torch.arange(len(sparse_coeff_vector))
            Overode = True
            '''
            Indices_To_Keep = remove_high_order(sparse_coeff_vector)
            if len(Indices_To_Keep) == 3:
                print('Defaulted to minimum library size')
            elif len(Indices_To_Keep) < 3:#Only possible to trigger if 'diff_order: 1' used and DeepMoD wants to remove a term.
                Indices_To_Keep = torch.arange(3)
            else:
                Overode = True
            '''
                
    sparsity_mask = sparsity_mask[Indices_To_Keep].detach()  # detach it so it doesn't get optimized and throws an error
    sparse_coeff_vector = sparse_coeff_vector[Indices_To_Keep].clone().detach().requires_grad_(True)  # so it can be optimized

    return sparse_coeff_vector, sparsity_mask, Overode

        
def check_need_overide(sparsity_mask_trial, original_diff_order):
    Index_Gap = original_diff_order+1
    
    #First check zeroth and first derivatives are present. In no model is there ever the situation where either of these should  be removed, except for a single spring
    for coeff_index in [0, Index_Gap-1, Index_Gap]:
        if not coeff_index in sparsity_mask_trial:
            return True
    
    #Next check if all pairs are like
    for coeff_index in range(1, original_diff_order):
        if (coeff_index in sparsity_mask_trial) != (coeff_index+Index_Gap in sparsity_mask_trial):
            return True
    
    #Next check that no derivatives skipped
    Expected_State = False
    for coeff_index in range(1, original_diff_order):
        if (coeff_index in sparsity_mask_trial) == Expected_State:
            if Expected_State == True:
                return True
            
            Expected_State = True
                
    return False


def remove_high_order(sparse_coeff_vector):
    
    Number_Of_Coeffs = len(sparse_coeff_vector)
    Low_Index = (Number_Of_Coeffs // 2) - 1
    
    Indices_To_Keep = torch.cat((torch.arange(Low_Index), torch.arange(Low_Index+1, Number_Of_Coeffs-1)))
    
    return Indices_To_Keep
