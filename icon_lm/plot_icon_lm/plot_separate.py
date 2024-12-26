'Plot the relative error of different types of problems with different captions'
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from jax.config import config
import jax
config = jax.config
# import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# tf.config.set_visible_devices([], device_type='GPU')
from collections import OrderedDict
from pprint import pprint
import jax.tree_util as tree
from absl import app, flags, logging


FLAGS = flags.FLAGS


label_map = {
	"mfc_gparam_hj_forward11": {'legend': "MFC $g$-parameter 1D -> 1D", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"mfc_gparam_hj_forward12": {'legend': "MFC $g$-parameter 1D -> 2D", 'linestyle': '--', 'marker': 'o', 'xlabel': r"$x$"},
	"mfc_gparam_hj_forward22": {'legend': "MFC $g$-parameter 2D -> 2D", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$x$"},
	"mfc_rhoparam_hj_forward11": {'legend': r"MFC $\rho_0$-parameter 1D -> 1D", 'linestyle': '-', 'marker': 's', 'xlabel': r"$x$"},
	"mfc_rhoparam_hj_forward12": {'legend': r"MFC $\rho_0$-parameter 1D -> 2D", 'linestyle': ':', 'marker': 's', 'xlabel': r"$x$"},
	"mfc_gparam_solve_forward": {'legend': "MFC 1", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"mfc_rhoparam_solve_forward": {'legend': "MFC 2", 'linestyle': '--', 'marker': 'o', 'xlabel': r"$x$"},
	"ode_auto_const_forward": {'legend': "Forward problem of ODE 1", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$t$"},
	"ode_auto_const_inverse": {'legend': "Inverse problem of ODE 1", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$t$"},
	"ode_auto_linear1_forward": {'legend': "Forward problem of ODE 2", 'linestyle': '-', 'marker': 's', 'xlabel': r"$t$"},
	"ode_auto_linear1_inverse": {'legend': "Inverse problem of ODE 2", 'linestyle': ':', 'marker': 's', 'xlabel': r"$t$"},
	"ode_auto_linear2_forward": {'legend': "Forward problem of ODE 3", 'linestyle': '-', 'marker': 'd', 'xlabel': r"$t$"},
	"ode_auto_linear2_inverse": {'legend': "Inverse problem of ODE 3", 'linestyle': ':', 'marker': 'd', 'xlabel': r"$t$"},
	"pde_poisson_spatial_forward": {'legend': "Forward Poisson equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"pde_poisson_spatial_inverse": {'legend': "Inverse Poisson equation", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$x$"},
	"pde_porous_spatial_forward" : {'legend': "Forward linear reaction-diffusion", 'linestyle': '-', 'marker': 's', 'xlabel': r"$x$"},
	"pde_porous_spatial_inverse": {'legend': "Inverse linear reaction-diffusion", 'linestyle': ':', 'marker': 's', 'xlabel': r"$x$"},
	"pde_cubic_spatial_forward":  {'legend': "Forward nonlinear reaction-diffusion", 'linestyle': '-', 'marker': 'd', 'xlabel': r"$x$"},
	"pde_cubic_spatial_inverse": {'legend': "Inverse nonlinear reaction-diffusion", 'linestyle': ':', 'marker': 'd', 'xlabel': r"$x$"},
	"series_damped_oscillator": {'legend': "time series prediction", 'linestyle': '-', 'marker': '*', 'xlabel': r"$t$"},
	"series_damped_oscillator_forward": {'legend': "Forward damped oscillator", 'linestyle': '-', 'marker': '*', 'xlabel': r"$t$"},
	"series_damped_oscillator_inverse": {'legend': "Inverse damped oscillator", 'linestyle': ':', 'marker': '*', 'xlabel': r"$t$"},
	"pde_heat_homog_forward": {'legend': "Forward homogeneous heat equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"pde_heat_homog_inverse": {'legend': "Inverse homogeneous heat equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"pde_heat_nonhomog_forward": {'legend': "Forward non-homogeneous heat equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
	"pde_heat_nonhomog_inverse": {'legend': "Inverse non-homogeneous heat equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"}
}


figure_config = {0: {'title': 'ODE 1', 'xlabel': 'x', 'ylabel': 'y', 'ylim': [0.004, 1]},
				1: {'title': 'ODE 2', 'xlabel': 'x', 'ylabel': 'y', 'ylim': [0.004, 1]},
				2: {'title': 'ODE 3', 'xlabel': 'x', 'ylabel': 'y', 'ylim': [0.004, 1]},
				}

def calculate_error(pred, label, mask):
	'''
	pred: [..., len, 1]
	label: [..., len, 1]
	mask: [..., len]
	'''
	mask = mask.astype(bool)
	error = np.linalg.norm(pred - label, axis = -1) # [..., len]
	error = np.mean(error, where = mask)
	gt_norm_mean = np.mean(np.linalg.norm(label, axis = -1), where = mask)
	relative_error = error/gt_norm_mean
	return error, relative_error

def pattern_match(patterns, name):
	'check if name contains any of the patterns'
	for pattern in patterns:
		if pattern in name:
			return True
	return False

def get_error_from_dict(result_dict, key, demo_num, caption_id):
	'get error from result_dict'
	error, relative_error = calculate_error(result_dict[(key, 'pred', demo_num, caption_id)], result_dict[(key, 'ground_truth')], result_dict[(key, 'mask')])
	return error, relative_error

def draw_seperate(folder, demo_num_list, caption_id, title):
	'draw separate plots for different types of problems'
	patterns_list = ['pde']

	with open(f"{folder}/result_dict.pkl", "rb") as file:
		result_dict = pickle.load(file)

	print("")
	for key, value in {k:v for k,v in result_dict.items() if k[1] != 'equation'}.items():
		print(key, np.mean(value), np.std(value), len(value), flush=True)

	for fi, patterns in enumerate(patterns_list):
		plot_key = [i[0] for i in result_dict.keys() if pattern_match(patterns, i[0])]
		plot_key = list(OrderedDict.fromkeys(sorted(plot_key)))
		print(plot_key)
		fig, ax = plt.subplots(figsize=(4,5))
		# fig, ax = plt.subplots(figsize=(6,3))

		for key in plot_key:
			relative_error_list = [get_error_from_dict(result_dict, key, demo_num, caption_id)[1] for demo_num in demo_num_list]
			print(f"Key: {key}, Errors: {relative_error_list}")
			ax.semilogy(demo_num_list, relative_error_list, label=label_map[key]['legend'], linestyle= label_map[key]['linestyle'], marker= label_map[key]['marker'], markersize=7)

		ax.set_xticks(demo_num_list)
		ax.set_xlabel('number of examples')
		ax.set_ylabel('relative error')
		ax.set_ylim(figure_config[fi]['ylim'])

		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize = 10)
		# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 10)

		ax.set_title(title)
		# grid on
		ax.grid(True, which='both', axis='both', linestyle=':')
		plt.subplots_adjust(bottom=0.46, left = 0.2, right = 0.95, top = 0.95)
		# plt.tight_layout()
		file_path = f'{folder}/ind_err_{title.replace(" ", "_")}_sub_{fi}_err.pdf'
		plt.savefig(file_path)
		print(f'saved to {file_path}')
		plt.close('all')

def plot_hero_sep(caption=False):
	'plot separate plots for different types of problems'
	stamp = FLAGS.ckpt
	if caption:
		draw_seperate(folder=f"{FLAGS.analysis_dir}/{stamp}-nocap", demo_num_list=[0, 1, 2, 3, 4, 5], caption_id=-1, title='no caption')
		draw_seperate(folder=f"{FLAGS.analysis_dir}/{stamp}-vague", demo_num_list=[0, 1, 2, 3, 4, 5], caption_id=0, title='vague caption')
		draw_seperate(folder=f"{FLAGS.analysis_dir}/{stamp}-precise", demo_num_list=[0, 1, 2, 3, 4, 5], caption_id=0, title='precise caption')
	else:
		draw_seperate(folder=f"{FLAGS.analysis_dir}/{stamp}", demo_num_list=[1,2,3,4,5], caption_id=-1, title='no caption')

def main(argv):
	'main function'
	del argv
	plot_hero_sep()

if __name__ == '__main__':

	FLAGS = flags.FLAGS
	flags.DEFINE_string('analysis_dir', "/Users/jamiemahowald/in-context-operator-networks/icon_lm/analysis", 'directory to load/save analysis')
	flags.DEFINE_string('ckpt', "icon_lm_learn_20240716-143836-heat-forward", 'checkpoint name')

	app.run(main)
