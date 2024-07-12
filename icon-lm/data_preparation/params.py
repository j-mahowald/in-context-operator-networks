import jax
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import haiku as hk
import tensorflow as tf

def serialize_element(coeffs, rep_number, count):
    '''
    equation: string describing the equation
    caption: list of strings describing the equation
    cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
    cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
    qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
    qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    _print = count < 5
    coeffs = np.array(coeffs, dtype=np.float32)
    rep_number = np.array([rep_number], dtype=np.float32)

    feature = {
      'params': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(coeffs).numpy()])),
      'representative_number': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(rep_number).numpy()])),
    }

    if _print:
      print('-'*50, count, '-'*50, flush=True)
      print("coeffs: {}".format(coeffs), flush=True)
      print("representative number: {}".format(rep_number), flush=True)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(name, all_params, all_representative_numbers):
  filename = "{}_forward.tfrecord".format(name)
  print("===========" + filename + "===========", flush=True)
  count = 0
  with tf.io.TFRecordWriter(filename) as writer:
    for (params, rep_number) in zip(all_params, all_representative_numbers):
      serialized_example = serialize_element(params, rep_number, count)
      writer.write(serialized_example)
      count += 1

def generate_params(seed, eqns, name):
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)

    # Check that next(rng) is actually working
    all_params = []
    all_representative_numbers = []
    for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
        print("coeff_a: ", coeff_a, "coeff_b: ", coeff_b)
        all_params.append((coeff_a, coeff_b))
        representative_key = next(rng)
        representative_number = jax.random.uniform(representative_key, (1,))
        print("representative_number: ", representative_number)
        all_representative_numbers.append(representative_number)

    write_tfrecord(name, all_params, all_representative_numbers)


if __name__ == "__main__":
    generate_params(seed = 0, eqns = 10, name = "params")