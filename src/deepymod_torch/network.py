import torch
import torch.nn as nn


class Library(nn.Module):
    def __init__(self, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args

    def forward(self, input):
        self.time_deriv_list, self.theta = self.library_func(input, **self.library_args)
        return self.time_deriv_list, self.theta


class Fitting(nn.Module):
    def __init__(self, n_equations, n_terms, library_config):
        super().__init__()
        tensor_list = [torch.rand((n_terms, 1), dtype=torch.float32) for _ in torch.arange(n_equations)]
        if 'coeff_sign' in library_config:
            tensor_list = [library_config['coeff_sign']*abs(tensor) for tensor in tensor_list]
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(tensor) for tensor in tensor_list])
        self.sparsity_mask = [torch.arange(n_terms) for _ in torch.arange(n_equations)]
        self.coeff_vector_history = []
        self.sparsity_mask_history = []

    def forward(self, input):
        sparse_theta = self.apply_mask(input)
        return sparse_theta, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
