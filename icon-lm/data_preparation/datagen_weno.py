import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.append('../')
sys.path.append('weno/')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

import data_writetfrecord as datawrite
import data_utils
from weno.weno_solver import generate_weno_scalar_sol, generate_weno_euler_sol

def generate_conservation_weno_cubic(seed, eqns, quests, length, steps, dt, num, name, eqn_mode):
  '''du/dt + d(a * u^2 + b * u)/dx = 0'''
  eqn_type = "conservation_weno_cubic"
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if 'random' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    coeffs_a = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_b = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_c = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
  elif 'grid' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    values = np.linspace(minval, maxval, eqns)
    coeffs_a, coeffs_b, coeffs_c = np.meshgrid(values, values, values)
    coeffs_a = coeffs_a.flatten()
    coeffs_b = coeffs_b.flatten()
    coeffs_c = coeffs_c.flatten()
  else:
    raise NotImplementedError("eqn_mode = {} is not implemented".format(FLAGS.eqn_mode))
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    print("coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(coeff_a, coeff_b, coeff_c), flush=True)
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
    grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
    for j in range(quests):
      while True:
        init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N+1)
        if jnp.max(jnp.abs(init)) < 3.0:
          break
      sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, stable_tol = 10.0) # (num, steps + 1, N, 1)
      all_xs.append(xs) # (N,)
      all_us.append(sol) # (num, steps + 1, N, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a, coeff_b, coeff_c))
      all_eqn_captions.append(['dummy caption'])
    utils.print_dot(i)
    if (i+1) % (len(coeffs_a)//FLAGS.file_split) == 0 or i == len(coeffs_a) - 1:
      for ptype in ['forward', 'backward']:
        for st in FLAGS.stride:
          sti = int(st)
          datawrite.write_evolution_tfrecord(seed = next(rng)[0], eqn_type = eqn_type, 
                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                            all_xs = all_xs, all_us = all_us, stride = sti,
                            problem_type = ptype, file_name = "{}_{}_{}_stride{}_{}.tfrecord".format(name, eqn_type, ptype, sti, i+1))
      all_xs = []; all_us = []; all_params = []; all_eqn_captions = []


def main(argv):
  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  
  name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)
  
  if 'weno_cubic' in FLAGS.eqn_types:
    generate_conservation_weno_cubic(
          seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, steps = 1000,
          dt = FLAGS.dt, num = FLAGS.num, name = name, eqn_mode = FLAGS.eqn_mode)

if __name__ == "__main__":

  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  # tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
  flags.DEFINE_integer('num', 100, 'number of systems in each equation')
  flags.DEFINE_integer('quests', 1, 'number of questions in each operator')
  flags.DEFINE_integer('eqns', 100, 'number of equations')
  flags.DEFINE_integer('length', 100, 'length of trajectory and control')
  flags.DEFINE_float('dt', 0.001, 'time step in dynamics')
  flags.DEFINE_float('dx', 0.01, 'spatial step in dynamics')
  flags.DEFINE_string('name', 'data', 'name of the dataset')
  flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
  flags.DEFINE_list('eqn_types', ['weno_cubic'], 'list of equations for data generation')
  flags.DEFINE_list('write', [], 'list of features to write')

  flags.DEFINE_list('stride', [200], 'time strides')

  flags.DEFINE_integer('seed', 1, 'random seed')
  flags.DEFINE_string('eqn_mode', 'random_-1_1', 'the mode of equation generation')
  flags.DEFINE_integer('file_split', 10, 'split the data into multiple files')
  flags.DEFINE_integer('truncate', None, 'truncate the length of each record')

  app.run(main)
