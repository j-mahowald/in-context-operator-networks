
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from pprint import pprint

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape

import sys
sys.path.append('../')
import utils
import plot
import matplotlib.pyplot as plt


import pickle 
from runner_jax import Runner_lm
import plot_weno_new as new

figsize=(4,3.5)

def main(argv):
  del argv
  Delta_t = FLAGS.dt * FLAGS.downsample # Delta_t = 0.01

  model_config = utils.load_json("../config_model/model_lm_config.json")
  model_config['caption_len'] = 0

  restore_dir = '/work2/09989/jmahowald/frontera/in-context-operator-networks/icon-lm/save/user/ckpts/icon_weno/{}'.format(FLAGS.ckpt)
  restore_step = 1000000
  real_name = 'sin_1.0_-1.0_1.0_1.0'
  name = real_name
  
  folder = f"weno_{real_name}_init_scale_{FLAGS.init_scale}_{FLAGS.groups}x{FLAGS.num}"
  if not os.path.exists(folder):
    os.makedirs(folder)
  data, runner = new.get_data_runner(model_config, restore_dir, restore_step, bs = FLAGS.num)
  rng = hk.PRNGSequence(jax.random.PRNGKey(1))

  stride = 5
  mode_list = ['vanilla', 'minmax_0.5', 'minmax_1', 'minmax_2', 'minmax_3']

  forward_sols = {}
  backward_sols = {}

  sols = forward_sols
  for mode in mode_list:
    file_name = f'{folder}/sim_{name}.npy'
    key = (name,"sim")
    print(file_name, key)
    try:
      # direct simulation
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key], fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                                      num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = sols[key][:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
      np.save(file_name, sols[key])

    file_name = f'{folder}/pred_forward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
    key = (name, name, stride, mode)
    print(file_name, key)
    try:
      # forward prediction with "name" record (t = 0 to 0.1) and "name" initial condition (at t = 0.1)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_forward(data, runner, sols[(name,"sim")], 
                                          FLAGS.ref, stride, mode, rng)
      np.save(file_name, sols[key])


  sols = backward_sols
  for mode in mode_list:
    file_name = f'{folder}/sim_{name}.npy'
    key = (name, "sim")
    print(file_name, key)
    try:
      #1 direct simulation
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key], fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                            num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = sols[key][:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
    key = (name, name, stride, mode)
    print(file_name, key)
    try:
      #2 backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.4)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_backward(data, runner, sols[(name,"sim")], 
                                           FLAGS.ref, stride, mode, rng)
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_sim_{name}_stride{stride}_{mode}.npy'
    key = (name, name, name, stride, mode)
    print(file_name, key)
    try:
      #3 backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.4)
      # then apply "name" equation and simulate to t = 0.4
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      # set steps = 1, not 0, due to jax issue
      _, fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = 1, dt = FLAGS.dt, 
                                            num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = new.get_backward_consistency(sols[(name, name, stride, mode)], FLAGS.ref, fn, grad_fn)
      np.save(file_name, sols[key])



  label_dict = {
    "vanilla": "no change of variable",
    'minmax_0.5': 'scale $u$ to [-0.5,0.5]',
    'minmax_1': 'scale $u$ to [-1,1]',
    'minmax_2': 'scale $u$ to [-2,2]',
    'minmax_3': 'scale $u$ to [-3,3]'
                }
  color_dict = {
    "vanilla": 'k-',
    'minmax_0.5': 'r--',
    'minmax_1': 'm--',
    'minmax_2': 'g--',
    'minmax_3': 'b--'
                }
  
  sols = forward_sols
  t = np.linspace(0.0, 0.5, 51, endpoint=True)
  plt.figure(figsize=figsize)
  for mode in mode_list:
    linestyle = color_dict[mode]
    plt.plot(t, new.get_error(sols[(name, "sim")], sols[(name, name, stride, mode)]), 
             linestyle, label = "{}".format(label_dict[mode]))
  
  plt.legend(loc = 'lower right')
  plt.ylim(-0,0.04)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/variable_forward_{folder}.pdf')


  sols = backward_sols
  t = np.linspace(0.5, 0, 51, endpoint=True)
  plt.figure(figsize=figsize)
  for mode in mode_list:
    error = new.get_error(sols[(name, name, name, stride, mode)], 
                          sols[(name, 'sim')][:,:,-(FLAGS.ref+1):-FLAGS.ref,:,:])
    error[:FLAGS.ref] = 0
    linestyle = color_dict[mode]
    plt.plot(t, error, linestyle, label = "{}".format(label_dict[mode]))
  
  plt.legend(loc = 'lower left')
  plt.ylim(-0,0.04)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/variable_backward_{folder}.pdf')


if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_list('mode', ["sin","tanh"], 'run mode')

  flags.DEFINE_integer('seed', 200, 'random seed')
  flags.DEFINE_integer('steps', 1000, 'training steps')
  flags.DEFINE_float('dt', 0.0005, 'dt')
  flags.DEFINE_float('init_scale', -1.0, 'the scale of initial condition')
  flags.DEFINE_integer('downsample', 20, 'downsample to build sequence')
  flags.DEFINE_integer('length', 100, 'length of grid')
  flags.DEFINE_integer('ref', 10, 'ref steps, based on downsampled sequence')
  flags.DEFINE_integer('stride', 5, 'prediction stride, based on downsampled sequence')
  flags.DEFINE_string('demo_mode', "random", 'mode for demo example selection')
  flags.DEFINE_integer('groups', 8, 'groups')
  flags.DEFINE_integer('num', 64, 'num')
  flags.DEFINE_bool('regen', False, 'regenerate data')
  flags.DEFINE_string('ckpt', "20231209-222440", 'checkpoint')


  app.run(main)
