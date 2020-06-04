# Use command %run -i "/home/working/data/make_all_plots.py"
# or %run -i "/home/alex/DeePyMoD_torch_AlexVersion/data/make_all_plots.py"
# Run from console with cwd as folder containing csv files of interest etc.

import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

cwd = os.getcwd()
git_top_path = cwd[:cwd.index("data")]
sys.path.append(git_top_path+'src')
import deepymod_torch.VE_datagen as VE_datagen

# Directories
try: # Check if variables already defined in ipython console (requires "-i" flag)
    main_train_path
    post_thresh_train_path
except NameError:
    main_train_path = input('Name of folder containing main training tensorboard')
    post_thresh_train_path = input('Name of folder containing post threshold training tensorboard')
if not os.path.isdir('Figures'):
    os.makedirs('Figures')
save_path = 'Figures/'

# Load data
with open('config_dict_list.txt', 'r') as file:
    for line in file:
        dict_start_index, dict_end_index = line.index("{"), line.index("}")
        dict_str = line[dict_start_index:dict_end_index+1]
        dict_str = dict_str.replace("<lambda>", "lambda").replace("<", "'").replace(">", "'")
        line_dict = eval(dict_str)
        if line.startswith('library'):
            library_config = line_dict
#         elif line.startswith('network'):
#             network_config = line_dict
#         elif line.startswith('optim'):
#             optim_config = line_dict
#         elif line.startswith('report'):
#             report_config = line_dict

with open('treatment_info_list.txt', 'r') as file:
    for line in file:
        value = re.search(r'(: )(.+)', line).group(2)
        if line.startswith('time_sf'):
            time_sf = float(value)
        elif line.startswith('strain_sf') or line.startswith('voltage_sf'):
            strain_sf = float(value)
        elif line.startswith('stress_sf') or line.startswith('current_sf'):
            stress_sf = float(value)

with open('DG_info_list.txt', 'r') as file:
    file_string = file.read()
    omega = float(re.search(r'(omega: )(.+)', file_string).group(2))
    Amp = int(re.search(r'(Amp: )(.+)', file_string).group(2))
            
dg_data = np.loadtxt('DG_series_data.csv', delimiter=',')
fit_data = np.loadtxt('NN_series_data.csv', delimiter=',')
full_pred = np.loadtxt('full_prediction.csv', delimiter=',')
expected_coeffs = np.loadtxt('expected_coeffs.csv', delimiter=',').flatten()
final_coeffs_data = np.loadtxt('final_coeffs_data.csv', delimiter=',')

# Extract Tensorboard data
path = cwd + '/' + main_train_path
event_file = next(filter(lambda filename: filename[:6] =='events', os.listdir(path)))
summary_iterator = EventAccumulator(str(path + '/' + event_file)).Reload()
tags_main = summary_iterator.Tags()['scalars']
steps_main = np.array([event.step for event in summary_iterator.Scalars(tags_main[0])])
data_main = np.array([[event.value for event in summary_iterator.Scalars(tag)] for tag in tags_main]).T

path = cwd + '/' + post_thresh_train_path
event_file = next(filter(lambda filename: filename[:6] =='events', os.listdir(path)))
summary_iterator = EventAccumulator(str(path + '/' + event_file)).Reload()
tags_pt = summary_iterator.Tags()['scalars']
steps_pt = np.array([event.step for event in summary_iterator.Scalars(tags_pt[0])])
data_pt = np.array([[event.value for event in summary_iterator.Scalars(tag)] for tag in tags_pt]).T


            
            
# plot final fit
time = fit_data[:, 0]
number_graphs = fit_data[:, 1:].shape[1]//2

fig, axes = plt.subplots(ncols=number_graphs, squeeze=False, figsize=(6*number_graphs, 5))
axes = axes.flatten()
for tar, ax in enumerate(axes):
    ax.set_title(input(f'Target title {tar}. Like: Scaled measured voltage manipulation.'))
    ax.set_xlabel('Scaled time')
    ax.plot(time, fit_data[:, 1+(2*tar)], linestyle='None', marker='.', markersize=1, color='blue', label='Target')
    ax.plot(time, fit_data[:, 2+(2*tar)], linestyle='None', marker='.', markersize=1, color='red', label='Prediction')
    ax.legend(numpoints=3, markerscale=5)
    
plt.tight_layout()
plt.savefig(save_path+'target_prediction_fit.png', bbox_inches='tight')






# Plot losses graphs
plot_ratio = len(steps_pt)/len(steps_main)
fig_width = 7*(1+plot_ratio)
fig, axes = plt.subplots(ncols=2, figsize=(fig_width, 5), sharey=True, gridspec_kw={'width_ratios': [1, plot_ratio], 'wspace': 0.13})
# the extra 'wspace' in the dict above I think confuses tight_layout() so the y kwarg for suptitle() must be different

mod = number_graphs - 1

ax1 = axes[0]
ax1.set_xlabel('                Epochs')
ax1.set_ylabel('Loss Magnitudes')
ax1.semilogy(steps_main, data_main[:, 1], color='blue', linestyle='None', marker='.', markersize=1, label='MSE')
if number_graphs == 2:
    ax1.semilogy(steps_main, data_main[:, 2], color='deepskyblue', linestyle='None', marker='.', markersize=1, label='MSE_1')
ax1.semilogy(steps_main, data_main[:, 2+mod], color='orange', linestyle='None', marker='.', markersize=1, label='PI')
ax1.semilogy(steps_main, data_main[:, 3+mod], color='green', linestyle='None', marker='.', markersize=1, label='L1')
ax1.semilogy(steps_main, data_main[:, 4+mod], color='purple', linestyle='None', marker='.', markersize=1, label='Sign')
ax1.semilogy(steps_main, data_main[:, 0], color='red', linestyle='None', marker='.', markersize=1, label='Total')
ax1.legend(numpoints=3, markerscale=5)
ax1.set_xlim(right=1.01*steps_main[-1])

ax2 = axes[1]
ax2.semilogy(steps_pt, data_pt[:, 1], color='blue', linestyle='None', marker='.', markersize=1, label='MSE')
if number_graphs == 2:
    ax2.semilogy(steps_pt, data_pt[:, 2], color='deepskyblue', linestyle='None', marker='.', markersize=1, label='MSE_1')
ax2.semilogy(steps_pt, data_pt[:, 2+mod], color='orange', linestyle='None', marker='.', markersize=1, label='PI')
ax1.semilogy(steps_pt, data_pt[:, 4+mod], color='purple', linestyle='None', marker='.', markersize=1, label='Sign')
ax2.semilogy(steps_pt, data_pt[:, 0], color='red', linestyle='None', marker='.', markersize=1, label='Total')
# ax2.set_xlim(left=0)
ax2.tick_params(axis='y', which='both', left=False)

plt.tight_layout()
plt.savefig(save_path+'loss_evolution.png', bbox_inches='tight')






# Plot coeffs graphs
first_coeff_column_idx = 4 + number_graphs
library_diff_order = library_config['diff_order']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
strain_labels = ['$\epsilon$', ' ', '$\epsilon_{tt}$', '$\epsilon_{ttt}$']
stress_labels = ['$\sigma$', '$\sigma_{t}$', '$\sigma_{tt}$', '$\sigma_{ttt}$']


fig, axes = plt.subplots(ncols=2, figsize=(fig_width, 5), sharey=True, gridspec_kw={'width_ratios': [1, plot_ratio], 'wspace': 0.13})

start_strain_coeffs_mask, start_stress_coeffs_mask = list(range(library_diff_order+1)), list(range(library_diff_order+1))

ax1 = axes[0]
ax1.set_xlabel('                Epochs')
ax1.set_ylabel('Associated Coefficient Magnitudes')

series_idx = first_coeff_column_idx # Initial value
for idx_diff_order in start_strain_coeffs_mask:
    if idx_diff_order == 1:
        ax1.plot(0, alpha=0, label=strain_labels[idx_diff_order]) # dummy to space legend in nice way
        continue
    ax1.plot(steps_main, data_main[:, series_idx], color=colors[idx_diff_order], linestyle='--', label=strain_labels[idx_diff_order])
    series_idx += 1

# series_idx carried forth to this loop.
for idx_diff_order in start_stress_coeffs_mask:
    ax1.plot(steps_main, data_main[:, series_idx], color=colors[idx_diff_order], label=stress_labels[idx_diff_order])
    series_idx += 1

ax1.legend(ncol=2)
ax1.set_xlim(right=1.01*steps_main[-1])

final_coeffs = final_coeffs_data[:, 0]
final_mask = final_coeffs_data[:, 1]
strain_coeffs_mask, stress_coeffs_mask = VE_datagen.align_masks_coeffs(final_coeffs, final_mask, library_diff_order)

strain_mask_aligned = strain_coeffs_mask[1]
stress_mask_aligned = stress_coeffs_mask[1]

ax2 = axes[1]

series_idx = first_coeff_column_idx # Initial value
for idx_diff_order in strain_mask_aligned:
    if idx_diff_order == 1:
        ax2.plot(0, alpha=0, label=strain_labels[idx_diff_order]) # dummy to space legend in nice way
        continue
    ax2.plot(steps_pt, data_pt[:, series_idx], color=colors[idx_diff_order], linestyle='--', label=strain_labels[idx_diff_order])
    series_idx += 1

# series_idx carried forth to this loop.
for idx_diff_order in stress_mask_aligned:
    ax2.plot(steps_pt, data_pt[:, series_idx], color=colors[idx_diff_order], label=stress_labels[idx_diff_order])
    series_idx += 1

# ax2.set_xlim(left=0)
ax2.tick_params(axis='y', which='both', left=False)

plt.tight_layout()
plt.savefig(save_path+'coeff_evolution.png', bbox_inches='tight')







# DV plots
expected_coeffs = list(expected_coeffs)

input_type = library_config['input_type']

scaled_time_array = dg_data[:, 0]*time_sf
scaled_strain_array = dg_data[:, 1]*strain_sf
scaled_stress_array = dg_data[:, 2]*stress_sf

if number_graphs == 1: # Tell tale sign that not using real data, so target comp allowed.
    errors_exp_tar = VE_datagen.equation_residuals(scaled_time_array, scaled_strain_array, scaled_stress_array, expected_coeffs)
    errors_DM_tar = VE_datagen.equation_residuals(scaled_time_array, scaled_strain_array, scaled_stress_array, final_coeffs, sparsity_mask=final_mask, diff_order=library_diff_order)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title('Agreement with target')
    ax.set_xlabel('Scaled time')
    ax.semilogy(scaled_time_array, abs(errors_exp_tar), linestyle='None', marker='.', markersize=1, color='green', label='Expected coefficients')
    ax.semilogy(scaled_time_array, abs(errors_DM_tar), linestyle='None', marker='.', markersize=1, color='red', label='Discovered coefficients')
    ax.legend(numpoints=3, markerscale=5)
    
    plt.tight_layout()
    plt.savefig(save_path+'target_residuals.png', bbox_inches='tight')


if number_graphs == 1: # Tell tale sign that not using real data, so target comp allowed.
    full_pred = full_pred.flatten()
    if input_type == 'Strain':
        errors_exp_pred = VE_datagen.equation_residuals(scaled_time_array, scaled_strain_array, full_pred, expected_coeffs)
        errors_DM_pred = VE_datagen.equation_residuals(scaled_time_array, scaled_strain_array, full_pred, final_coeffs, sparsity_mask=final_mask, diff_order=library_diff_order)
    else:
        errors_exp_pred = VE_datagen.equation_residuals(scaled_time_array, full_pred, scaled_stress_array, expected_coeffs)
        errors_DM_pred = VE_datagen.equation_residuals(scaled_time_array, full_pred, scaled_stress_array, final_coeffs, sparsity_mask=final_mask, diff_order=library_diff_order)
else:
    strain_pred, stress_pred = full_pred[:, 0], full_pred[:, 1]
    
    errors_exp_pred = VE_datagen.equation_residuals(scaled_time_array, strain_pred, stress_pred, expected_coeffs)
    errors_DM_pred = VE_datagen.equation_residuals(scaled_time_array, strain_pred, stress_pred, final_coeffs, sparsity_mask=final_mask, diff_order=library_diff_order)
        
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title('Agreement with prediction')
ax.set_xlabel('Scaled time')
ax.semilogy(scaled_time_array, abs(errors_exp_pred), linestyle='None', marker='.', markersize=1, color='green', label='Expected coefficients')
ax.semilogy(scaled_time_array, abs(errors_DM_pred), linestyle='None', marker='.', markersize=1, color='red', label='Discovered coefficients')
ax.legend(numpoints=3, markerscale=5)
    
plt.tight_layout()
plt.savefig(save_path+'prediction_residuals.png', bbox_inches='tight')






# Regen plot
if number_graphs == 1:
    full_pred = full_pred.flatten()
    input_expr = lambda t: Amp*np.sin(omega*t)/(omega*t)
    if input_type == 'Strain':
        scaled_input_expr = lambda t: strain_sf*input_expr(t/time_sf)
        target_array = scaled_stress_array
        response_type = 'Stress'
    else:
        scaled_input_expr = lambda t: stress_sf*input_expr(t/time_sf)
        target_array = scaled_strain_array
        response_type = 'Strain'
        
    response_recalc = VE_datagen.calculate_int_diff_equation(scaled_time_array, full_pred, scaled_input_expr, final_coeffs, final_mask, library_diff_order, input_type)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(response_type+' response reformulation\nfrom discovered coefficients')
    ax.set_xlabel('Scaled time')
    ax.plot(scaled_time_array, target_array, label='Target')
    ax.plot(scaled_time_array, response_recalc.flatten(), label='Reformulation', marker='.', markersize=1, linestyle='None')
    ax.legend(numpoints=3, markerscale=5)
    
    plt.tight_layout()
    plt.savefig(save_path+'recalculation_from_coeffs.png', bbox_inches='tight')