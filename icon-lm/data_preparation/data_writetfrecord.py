import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from einshape import jax_einshape as einshape
from pprint import pprint
tf.config.set_visible_devices([], device_type='GPU')
import sys
sys.path.append('..')
import utils
from absl import flags

FLAGS = flags.FLAGS

def serialize_element(equation, caption, cond_k, cond_v, qoi_k, qoi_v, count):
    '''
    equation: string describing the equation
    caption: list of strings describing the equation
    cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
    cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
    qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
    qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    write_list = FLAGS.write
    _print = count < 5

    # utils.print_dot(count, freq = 100, marker = "+")
    cond_k = cond_k.astype(np.float32)
    cond_v = cond_v.astype(np.float32)
    qoi_k = qoi_k.astype(np.float32)
    qoi_v = qoi_v.astype(np.float32)

    feature = {
      'equation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[equation.encode("utf-8")])),
      'cond_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_k).numpy()])),
      'cond_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_v).numpy()])),
      'qoi_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_k).numpy()])),
      'qoi_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_v).numpy()])),
    }

    if _print:
      print('-'*50, count, '-'*50, flush=True)
      print("equation: {}".format(equation), flush=True)
      print("cond_k.shape:  {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



def write_ode_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_ts, all_cs, all_us, problem_type):
  num = all_us[0].shape[0]
  if problem_type == "forward":
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_expand, control, traj in zip(all_params, all_ts, all_cs, all_us):
        equation_name = "{}_forward_{}".format(eqn_type, params)
        cond_k_c = jnp.pad(ts_expand[:,:-1,:], ((0,0),(0,0),(0,1)), mode = 'constant', constant_values = 0.0)
        cond_k_i = einshape('i->jki', jnp.array([0.0,1.0]), j = num, k = 1) # Delineating the term (0 for control, 1 for initial condition)
        cond_k = jnp.concatenate([cond_k_c, cond_k_i], axis = 1)
        cond_v_c = control[:,:-1,:]
        cond_v_i = traj[:,0:1,:]
        cond_v = jnp.concatenate([cond_v_c, cond_v_i], axis = 1)
        count += 1
        if np.sum(traj) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None, 
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = ts_expand, qoi_v = traj,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if problem_type == "inverse":
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_expand, control, traj in zip(all_params, all_ts, all_cs, all_us):
        equation_name = "{}_inverse_{}".format(eqn_type, params)
        count += 1
        if np.sum(traj) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None, 
                                      cond_k = ts_expand, cond_v = traj, qoi_k = ts_expand, qoi_v = control,
                                      count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

def write_series_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_ts_first, all_us_first, all_ts_second, all_us_second, problem_type):
  if problem_type == "forward":
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_first, us_first, ts_second, us_second in zip(all_params, all_ts_first, all_us_first, all_ts_second, all_us_second):
        equation_name = "{}_forward_{}".format(eqn_type, params)
        count += 1
        if jnp.sum(us_second) != jnp.nan:
          s_element= serialize_element(equation = equation_name, caption = None, 
                                       cond_k = ts_first, cond_v = us_first, qoi_k = ts_second, qoi_v = us_second,
                                      count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")
  
  if problem_type == "inverse":
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_first, us_first, ts_second, us_second in zip(all_params, all_ts_first, all_us_first, all_ts_second, all_us_second):
        equation_name = "{}_inverse_{}".format(eqn_type, params)
        count += 1
        if jnp.sum(us_second) != jnp.nan:
          s_element= serialize_element(equation = equation_name, caption = None, 
                                       cond_k = ts_second, cond_v = us_second, qoi_k = ts_first, qoi_v = us_first,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

def write_pde_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_xs, all_ks, all_us, problem_type):
  if problem_type == "forward":
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, ks, us in zip(all_params, all_xs, all_ks, all_us):
        equation_name = "{}_forward_{}".format(eqn_type, params)
        count += 1
        if np.sum(us) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None, 
                                       cond_k = xs, cond_v = ks, qoi_k = xs, qoi_v = us,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if problem_type == "inverse":
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, ks, us in zip(all_params, all_xs, all_ks, all_us):
        equation_name = "{}_inverse_{}".format(eqn_type, params)
        count += 1
        if np.sum(us) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = xs, cond_v = us, qoi_k = xs, qoi_v = ks,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")


def write_mfc_gparam_hj_tfrecord(name, eqn_type, all_params, all_eqn_captions, all_txs, all_rhos, problem_type, nt, nx):
  '''
  txs: (num, nt * nx, 2)
  rhos: (num, nt * nx, 1)
  '''
  first_half_t = nt // 2
  
  if problem_type == "forward11":
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward11")
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        equation_name = "{}_{}_{}".format(eqn_type, "forward11", params)
        cond_k = txs[:,:nx,:] # initial rho
        cond_v = rhos[:,:nx,:] # initial rho
        qoi_k = txs[:,-nx:,:] # terminal rho
        qoi_v = rhos[:,-nx:,:] # terminal rho
        count += 1
        if np.sum(rhos) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if problem_type == "forward12":
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward12")
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        equation_name = "{}_{}_{}".format(eqn_type, "forward12", params)
        cond_k = txs[:,:nx,:] # initial rho
        cond_v = rhos[:,:nx,:] # initial rho
        qoi_k = txs[:,first_half_t * nx:,:] # second half t rho
        qoi_v = rhos[:,first_half_t * nx:,:] # second half t rho
        count += 1
        if np.sum(rhos) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if problem_type == "forward22":
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward22")
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        equation_name = "{}_{}_{}".format(eqn_type, "forward22", params)
        cond_k = txs[:,:first_half_t * nx,:] # first half t rho
        cond_v = rhos[:,:first_half_t * nx,:] # first half t rho
        qoi_k = txs[:,first_half_t * nx:,:] # second half t rho
        qoi_v = rhos[:,first_half_t * nx:,:] # second half t rho
        count += 1
        if np.sum(rhos) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")


def write_mfc_rhoparam_hj_tfrecord(name, eqn_type, all_params, all_eqn_captions,
                                   all_rhos_key, all_rhos_value, all_gs_key, all_gs_value, 
                                   problem_type, nt, nx):
  '''
  rhos_key: (num, nt * nx, 2)
  rhos_value: (num, nt * nx, 1)
  gs_key: (num, nx, 1)
  gs_value: (num, nx, 1)
  '''
  first_half_t = nt // 2

  if problem_type == "forward11":
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward11")
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, rhos_key, rhos_value, gs_key, gs_value in zip(all_params, all_rhos_key, all_rhos_value, all_gs_key, all_gs_value):
        equation_name = "{}_{}_{}".format(eqn_type, "forward11", params)
        cond_k = gs_key
        cond_v = gs_value
        qoi_k = rhos_key[:,-nx:,:] # terminal rho
        qoi_v = rhos_value[:,-nx:,:] # terminal rho
        count += 1
        if np.sum(rhos_value) != np.nan:
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if problem_type == "forward12":
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward12")
    print("===========" + filename + "===========", flush=True)
    count = 0
    with tf.io.TFRecordWriter(filename) as writer:
      for params, rhos_key, rhos_value, gs_key, gs_value in zip(all_params, all_rhos_key, all_rhos_value, all_gs_key, all_gs_value):
        if np.sum(rhos_value) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward12", params)
          cond_k = gs_key
          cond_v = gs_value
          qoi_k = rhos_key[:,first_half_t * nx:,:] # second half t rho
          qoi_v = rhos_value[:,first_half_t * nx:,:] # second half t rho
          count += 1
          s_element= serialize_element(equation = equation_name, caption = None,
                                       cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v,
                                       count = count)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

def write_pde_3d(name, eqn_type, all_params, all_eqn_captions, all_xs, all_ts, all_gs, all_us, problem_type):
    if problem_type not in ["forward", "inverse"]:
        raise ValueError(f"Invalid problem_type: {problem_type}. Must be either 'forward' or 'inverse'.")

    filename = f"{name}_{eqn_type}_{problem_type}.tfrecord"
    print(f"==========={filename}===========", flush=True)
    
    with tf.io.TFRecordWriter(filename) as writer:
        count = 0
        for params, eqn_captions, xs, ts, gs, us in zip(all_params, all_eqn_captions, all_xs, all_ts, all_gs, all_us):
            equation_name = f"{eqn_type}_{problem_type}_{params}"
            caption = eqn_captions

            # Prepare coordinate arrays
            x_coords = xs[0, 0, :, 0]  # (N_x+1,)
            t_coords = ts[0, :, 0, 0]  # (N_t+1,)

            # Combine x and t coordinates
            xt_coords = np.array(np.meshgrid(x_coords, t_coords)).T.reshape(-1, 2)  # ((N_x+1)*(N_t+1), 2)

            for i in range(us.shape[0]):  # iterate over 'num' dimension
                count += 1
                if not np.isnan(np.sum(us[i])) and not np.isnan(np.sum(gs[i])):
                    g_values = gs[i].reshape(-1)  # ((N_x+1)*(N_t+1),)
                    u_values = us[i].reshape(-1)  # ((N_x+1)*(N_t+1),)

                    if problem_type == "forward":
                        cond_v, qoi_v = g_values, u_values
                    else:  # backward
                        cond_v, qoi_v = u_values, g_values

                    s_element = serialize_element(
                        equation=equation_name,
                        caption=caption,
                        cond_k=xt_coords,
                        cond_v=cond_v,
                        qoi_k=xt_coords,
                        qoi_v=qoi_v,
                        count=count
                    )
                    writer.write(s_element)