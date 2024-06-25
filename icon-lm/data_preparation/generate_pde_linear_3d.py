import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

import data_dynamics as dyn
import data_series as series
import data_pdes as pdes
import data_mfc_hj as mfc_hj
import data_writetfrecord as datawrite
import data_utils

import tensorflow as tf

def print_dot(i):
  if i % 100 == 0:
    print(".", end = "", flush = True)

def generate_pde_linear_3d(seed, eqns, quests, length_x, length_t, dx, dt, num, name):
  '''
  Generate PDE data for linear PDEs
  a*u_xx + b*u_xt + c*u_tt + d*u_x + e*u_t + f*u = g(x,t)
  '''

  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  N_x, N_t = [length_x, length_t]
  L_x, L_t = [N_x * dx, N_t * dt]
    
  coeffs = [jax.random.uniform(next(rng), (eqns,), minval=-1, maxval=1) for _ in range(6)]
  coeffs_a, coeffs_b, coeffs_c, coeffs_d, coeffs_e, coeffs_f = coeffs

  all_xs = []; all_ts = []; all_us = []; all_gs = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c, coeffs_d, coeffs_e, coeffs_f)):
    for j in range(quests):
      xs = jnp.linspace(0.0, 1.0, N_x+1)# (N+1,)
      ts = jnp.linspace(0.0, 1.0, N_t+1)# (N+1,)
      uxt_GP = data_utils.generate_gaussian_process_3d(key = next(rng), xs = xs, ts = ts, num = num, 
                                                       kernel = data_utils.rbf_kernel_3d, k_sigma = 1.0, k_l = 0.2) # (num, N_x+1, N_t+1)
      uxt_GP = uxt_GP.reshape(num, N_t+1, N_x+1)
      coeffs = [coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f]
      # Check if uxt_GP.shape() is (num, N+1, N+1)
      g = pdes.solve_pde_linear_3d_batch(L_x, L_t, N_x, N_t, uxt_GP, coeffs) # (num, N+1, N+1)
      all_xs.append(einshape("i->jkil", xs, j=num, k=N_t+1, l=1))  # (num, N_t+1, N_x+1, 1)
      all_ts.append(einshape("i->jkil", ts, j=num, k=N_x+1, l=1))  # (num, N_x+1, N_t+1, 1)
      all_gs.append(einshape("ijk->ijkl", g, l=1))  # (num, N_t+1, N_x+1, 1)
      all_us.append(einshape("ijk->ijkl", uxt_GP, l=1))  # (num, N_t+1, N_x+1, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(*coeffs))
      all_eqn_captions.append(None)
    print_dot(i)

    for ptype in ["forward", "inverse"]:
      datawrite.write_pde_3d(name=name, eqn_type="pde_linear_3d",
                                    all_params=all_params, all_eqn_captions=all_eqn_captions,
                                    all_xs=all_xs, all_ts=all_ts, all_gs=all_gs, all_us=all_us,
                                    problem_type=ptype)

def main(argv):
  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)
  
  if 'pde_linear_3d' in FLAGS.eqn_types:
    generate_pde_linear_3d(
          seed=FLAGS.seed, eqns=FLAGS.eqns, quests=FLAGS.quests, 
          length_x=FLAGS.length_x, length_t=FLAGS.length_t, 
          dx=FLAGS.dx, dt=FLAGS.dt, num=FLAGS.num, name=name)

if __name__ == "__main__":

  import tensorflow as tf
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
  flags.DEFINE_integer('num', 6, 'number of systems in each equation')
  flags.DEFINE_integer('quests', 1, 'number of questions in each operator')
  flags.DEFINE_integer('eqns', 1, 'number of equations')
  flags.DEFINE_integer('length_x', 100, 'length of trajectory and control')
  flags.DEFINE_integer('length_t', 100, 'length of time')
  flags.DEFINE_float('dt', 0.01, 'time step in dynamics')
  flags.DEFINE_float('dx', 0.01, 'spatial step in dynamics')
  flags.DEFINE_string('name', 'train', 'name of the dataset')
  flags.DEFINE_string('dir', 'data', 'name of the directory to save the data')
  flags.DEFINE_list('eqn_types', ['pde_linear_3d'], 'list of equations for data generation')
  flags.DEFINE_list('write', [], 'list of features to write')

  flags.DEFINE_integer('seed', 1, 'random seed')
  flags.DEFINE_string('eqn_mode', 'random_-1_1', 'the mode of equation generation')
  flags.DEFINE_integer('file_split', 10, 'split the data into multiple files')
  flags.DEFINE_integer('truncate', None, 'truncate the length of each record')

  app.run(main)