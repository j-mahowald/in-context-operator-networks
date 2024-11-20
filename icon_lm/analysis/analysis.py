'Analysis for ICON-LM'
import os
import sys
sys.path.append('../')

import gc
import glob
import pickle
from datetime import datetime
from collections import namedtuple
from functools import partial
from pprint import pprint

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import torch
import tensorflow as tf
import haiku as hk
import optax
import pytz
from absl import app, flags, logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
print("working directory: ", os.getcwd())
import utils
import plot

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')

def append_dict_list(dict_input, key, value):
	'appends value to the list of dict[key]'
	if key not in dict_input:
		dict_input[key] = []
	dict_input[key].append(value)
	return dict_input

def get_key(task, eqn_name):
	'''
	get the key for the eqn_name
	@param
		task: string, the task name, 'ind', 'ood'
		eqn_name: string, the name of the equation
	@return key: tuple
	'''
	eqn_name = eqn_name.numpy().decode('utf-8')

	if 'weno' in task:
		eqn_name_split = eqn_name.split("_")
		eqn_name_clean = "_".join(eqn_name_split[:4])
		eqn_params = [float(i) for i in eqn_name_split[4:]]
		key = (eqn_name_clean, *eqn_params)
		return key

	if task == 'ind' or task == 'len':
		eqn_name_split = eqn_name.split("_")
		eqn_name_clean = "_".join(eqn_name_split[:4])
		key = (eqn_name_clean,)
	elif task == 'ood':
		if "ode_auto_const" in eqn_name:
			coeffs = eqn_name.split("_")
			coeff1_buck = int(np.floor(float(coeffs[4])* 10))/10 # buck size 0.1
			coeff2_buck = int(np.floor(float(coeffs[5])* 5))/5 # buck size 0.2
			key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
		elif "ode_auto_linear1" in eqn_name:  # e.g., ode_auto_linear1_forward_0.200_-0.091
			coeffs = eqn_name.split("_")
			coeff1_buck = int(np.floor(float(coeffs[4])* 10))/10 # buck size 0.1
			coeff2_buck = int(np.floor(float(coeffs[5])* 5))/5 # buck size 0.2
			key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
		elif "ode_auto_linear2" in eqn_name:  # e.g., ode_auto_linear2_forward_0.128_0.556_1.156
			coeffs = eqn_name.split("_")
			coeff1_buck = int(np.floor(float(coeffs[5])* 10))/10 # buck size 0.1
			coeff2_buck = int(np.floor(float(coeffs[6])* 5))/5 # buck size 0.2
			key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
		elif "pde_porous_spatial" in eqn_name:  # e.g., pde_porous_spatial_forward_0.128_0.560_-5.248_0.404, four num: ul, ur, c, a
			coeffs = eqn_name.split("_")
			coeff1_buck = int(np.floor(float(coeffs[7])* 10))/10  # coeffa, buck size 0.1
			coeff2_buck = int(np.floor(float(coeffs[6])* 2.5))/2.5 # coeffc, buck size 0.4
			key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
		elif "series_damped_oscillator" in eqn_name:  # e.g., series_damped_oscillator_forward_-0.624
			coeffs = eqn_name.split("_")
			coeff1_buck = int(np.floor(float(coeffs[4])* 6.66))/6.66  # decay, buck size 0.15
			key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff1_buck)
		elif "ode_auto_linear3" in eqn_name:
			coeff = float(eqn_name.split("_")[5]) # use the second coeff as key
			key =  ("_".join(eqn_name.split("_")[:4]), coeff)
		else:
			raise NotImplementedError
	return key


def write_into_dict(result_dict, task, runner, equation, all_caption, all_data, label, test_demo_num_list, test_caption_id_list, split_data):
	'Write the error, ground truth, mask, and caption into dict'
	FLAGS = flags.FLAGS
	if FLAGS.backend == 'jax':
		# flatten the batch dimension
		all_data_flat = tree.tree_map(lambda x: einshape('ij...->(ij)...', x), all_data)
		label_flat = einshape('ij...->(ij)...', label)
	else:
		all_data_flat = all_data
		label_flat = label

	test_caption_id_list_in_use = test_caption_id_list.copy()
	# write ground truth and mask into dict
	for i in range(label_flat.shape[0]):
		eqn_key = get_key(task, equation[i])
		append_dict_list(result_dict, (*eqn_key, 'ground_truth'), label_flat[i,0])
		append_dict_list(result_dict, (*eqn_key, 'mask'), all_data_flat.quest_qoi_mask[i,0])

	if 'quest' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(task, equation[i])
			append_dict_list(result_dict, (*eqn_key, 'cond_k'), all_data_flat.quest_cond_k[i,0])
			append_dict_list(result_dict, (*eqn_key, 'cond_v'), all_data_flat.quest_cond_v[i,0])
			append_dict_list(result_dict, (*eqn_key, 'cond_mask'), all_data_flat.quest_cond_mask[i,0])
			append_dict_list(result_dict, (*eqn_key, 'qoi_k'), all_data_flat.quest_qoi_k[i,0])

	if 'demo' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(task, equation[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_k'), all_data_flat.demo_cond_k[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_v'), all_data_flat.demo_cond_v[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_cond_mask'), all_data_flat.demo_cond_mask[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_k'), all_data_flat.demo_qoi_k[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_v'), all_data_flat.demo_qoi_v[i])
			append_dict_list(result_dict, (*eqn_key, 'demo_qoi_mask'), all_data_flat.demo_qoi_mask[i])

	if 'equation' in FLAGS.write:
		for i in range(label_flat.shape[0]):
			eqn_key = get_key(task, equation[i])
			append_dict_list(result_dict, (*eqn_key, 'equation'), equation[i])

	# write error into dict, test_caption_id_list_in_use = [-1] or [0] or [-1,0]

	# -1 indicates no caption
	if -1 in test_caption_id_list_in_use:
		test_caption_id_list_in_use.remove(-1) # remove -1 from the list
		for demo_num, caption_id, caption, data in split_data(all_caption, all_data, test_demo_num_list, [0]):
			this_error, this_pred = runner.get_error(data, label, with_caption = False, return_pred = True)
			if FLAGS.backend == 'jax':
				this_error = einshape('ij...->(ij)...', this_error)
				this_pred = einshape('ij...->(ij)...', this_pred)
			for i in range(this_error.shape[0]):
				eqn_key = get_key(task, equation[i])
				append_dict_list(result_dict, (*eqn_key, 'error', demo_num, -1), this_error[i]) # -1 means no caption
				append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, -1), this_pred[i])

	# count len(test_caption_id_list_in_use other than -1), usually 0 or 1
	for demo_num, caption_id, caption, data in split_data(all_caption, all_data, test_demo_num_list, test_caption_id_list_in_use):
		this_error, this_pred = runner.get_error(data, label, with_caption = True, return_pred = True)
		if FLAGS.backend == 'jax':
			this_error = einshape('ij...->(ij)...', this_error)
			this_pred = einshape('ij...->(ij)...', this_pred)
		for i in range(this_error.shape[0]):
			eqn_key = get_key(task, equation[i])
			append_dict_list(result_dict, (*eqn_key, 'error', demo_num, caption_id), this_error[i])
			append_dict_list(result_dict, (*eqn_key, 'pred', demo_num, caption_id), this_pred[i])


def run_analysis():
	'Run analysis'
	FLAGS = flags.FLAGS
	utils.set_seed(FLAGS.seed)
	from dataloader import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader

	test_data_dirs = FLAGS.test_data_dirs
	test_file_names = [f"{i}/{j}" for i in test_data_dirs for j in FLAGS.test_data_globs]

	print("test_file_names: ", flush=True)
	pprint(test_file_names)

	test_config = utils.load_json("../config_data/" + FLAGS.test_config_filename)
	model_config = utils.load_json("../config_model/" + FLAGS.model_config_filename)

	if 'cap' not in FLAGS.loss_mode:
		model_config['caption_len'] = 0
		test_config['load_list'] = []


	print('==============data config==============', flush = True)
	print("test_config: ", flush=True)
	pprint(test_config)
	print('==============data config end==============', flush = True)

	print('-----------------------model config-----------------------')
	print("model_config: ", flush=True)
	pprint(model_config)
	print('-----------------------model config end-----------------------')


	if FLAGS.backend == 'jax':
		optimizer = optax.adamw(0.0001) # dummy optimizer
		data_num_devices = len(jax.devices())
		opt_config = {}
	elif FLAGS.backend == 'torch':
		optimizer = optax.adamw(0.0001) # dummy optimizer
		opt_config = {'peak_lr': 0.001,
					'end_lr': 0,
					'warmup_steps': 10,
					'decay_steps': 100,
					'gnorm_clip': 1,
					'weight_decay': 0.0001,
					}
		data_num_devices = 0
	else:
		raise ValueError(f"backend {FLAGS.backend} not supported")

	test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
	test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

	test_data = DataProvider(seed = FLAGS.seed + 10,
							config = test_config,
							file_names = test_file_names,
							batch_size = FLAGS.batch_size,
							deterministic = True,
							drop_remainder = False,
							shuffle_dataset = False,
							num_epochs=1,
							shuffle_buffer_size=10,
							num_devices=data_num_devices,
							real_time = True,
							caption_home_dir = '../data_preparation',
							)

	equation, caption, data, label = test_data.get_next_data()
	print_eqn_caption(equation, caption, decode = False)
	print(tree.tree_map(lambda x: x.shape, data))

	if FLAGS.model in ['icon', 'icon_scale', 'icon_scale_surrogate']:
		from runners.runner_jax import Runner_vanilla
		runner = Runner_vanilla(seed = FLAGS.seed,
						model = FLAGS.model,
						data = data,
						model_config = model_config,
						optimizer = optimizer,
						trainable_mode = 'all',
						)
	elif FLAGS.model in ['icon_lm']:
		from runners.runner_jax import Runner_lm
		runner = Runner_lm(seed = FLAGS.seed,
						model = FLAGS.model,
						data = data,
						model_config = model_config,
						optimizer = optimizer,
						trainable_mode = 'all',
						loss_mode = FLAGS.loss_mode,
						)
	elif FLAGS.model in ['gpt2']:
		from runners.runner_torch import Runner
		runner = Runner(data, model_config, opt_config = opt_config,
						model_name = FLAGS.model, pretrained = True,
						trainable_mode = 'all',
						loss_mode = FLAGS.loss_mode,
						)
	else:
		raise ValueError(f"model {FLAGS.model} not supported")


	runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
	result_dict = {}

	test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
	test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

	write_into_dict(result_dict, FLAGS.task, runner, equation, caption, data, label,
					test_demo_num_list, test_caption_id_list, split_data)

	read_step = 0
	while True:
		utils.print_dot(read_step)
		read_step += 1
		try:
			equation, caption, data, label = test_data.get_next_data()
		except StopIteration:
			break
		write_into_dict(result_dict, FLAGS.task, runner, equation, caption, data, label,
						test_demo_num_list, test_caption_id_list, split_data)

	for key, value in result_dict.items():
		result_dict[key] = np.array(value)
	return result_dict


def main(argv):
	'Entry point'
	FLAGS = flags.FLAGS
	for key, value in FLAGS.__flags.items():
		print(value.name, ": ", value._value, flush=True)

	utils.set_seed(FLAGS.seed + 123456)
	result_dict = run_analysis()

	print("")
	for key, value in result_dict.items():
		print(key, value.shape, end = ", ", flush=True)
		try:
			print(np.mean(value), np.std(value), flush=True)
		except Exception as e:
			print("", flush=True)

	if not os.path.exists(FLAGS.analysis_dir):
		os.makedirs(FLAGS.analysis_dir)
	with open(f"{FLAGS.analysis_dir}/result_dict{FLAGS.results_name}.pkl", "wb") as f:
		pickle.dump(result_dict, f)

	print(f"result_dict saved to {FLAGS.analysis_dir}", flush=True)

	if 'weno' in FLAGS.task:
		import analysis_weno_aug
		if FLAGS.task == 'weno_quadratic':
			eqn_name = 'conservation_weno_quadratic_backward'
			analysis_weno_aug.write_quadratic_consistency_error(FLAGS.analysis_dir, eqn_name)
		elif FLAGS.task == 'weno_cubic':
			eqn_name = 'conservation_weno_cubic_backward'
			analysis_weno_aug.write_cubic_consistency_error(FLAGS.analysis_dir, eqn_name)
		elif FLAGS.task == 'weno_sin':
			eqn_name = 'conservation_weno_sin_backward'
			analysis_weno_aug.write_sin_consistency_error(FLAGS.analysis_dir, eqn_name)
		else:
			raise ValueError(f"task {FLAGS.task} not supported")
	else:
		pass

if __name__ == '__main__':

	FLAGS = flags.FLAGS
	flags.DEFINE_boolean('tfboard', False, 'dump into tfboard')

	flags.DEFINE_enum('task', 'ind', ['ind', 'ood', 'len', 'weno_quadratic', 'weno_cubic'], 'task type')
	flags.DEFINE_enum('backend', 'torch', ['jax','torch'], 'backend of runner')

	flags.DEFINE_integer('seed', 44, 'random seed')

	flags.DEFINE_list('test_data_dirs', '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon_lm/data', 'directories of testing data')
	flags.DEFINE_list('test_data_globs', ['test*'], 'filename glob patterns of testing data')
	flags.DEFINE_string('test_config_filename', 'test_input_id_config.json', 'config file for testing')
	flags.DEFINE_list('test_demo_num_list', [1,2,3,4,5], 'demo number list for testing')
	flags.DEFINE_list('test_caption_id_list', [-1], 'caption id list for testing')

	flags.DEFINE_integer('batch_size', 32, 'batch size')
	flags.DEFINE_list('loss_mode', ['nocap'], 'loss mode')
	flags.DEFINE_list('write', ['quest', 'demo', 'equation'], 'write mode')

	flags.DEFINE_string('model', 'icon_lm', 'model name')
	flags.DEFINE_string('model_config_filename', '../config_model/model_gpt2_config.json', 'config file for model')
	flags.DEFINE_string('analysis_dir', '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon_lm/jamie/ckpts/icon_lm/20240716-143836', 'write file to dir')
	flags.DEFINE_string('results_name', '', 'additional file name for results')
	flags.DEFINE_string('restore_dir', '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon_lm/jamie/ckpts/icon_lm/20240716-143836', 'restore directory')
	flags.DEFINE_integer('restore_step', 900000, 'restore step')


	app.run(main)
