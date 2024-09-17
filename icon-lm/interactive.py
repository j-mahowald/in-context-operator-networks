import os
from absl import app, flags
FLAGS = flags.FLAGS
from einshape import jax_einshape as einshape
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/workspace/Jamie/anaconda3/envs/icon-tacc'
os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jax.tree_util as tree
import jax, optax
import jax.numpy as jnp
import sys
sys.path.append('../')
import utils
import haiku as hk
import data_preparation.data_dynamics as dyn
import data_preparation.data_writetfrecord as datawrite
from pprint import pprint
import data_preparation.data_utils as dutils
import data_preparation.data_pdes as pdes
import data_preparation.data_series as series   
print("Devices: ", jax.devices())

jax.config.update('jax_platform_name', 'cpu')

seed = 1

def parse_equation(equation_dict):
    equation_types = ['ode_auto_const_forward', 'ode_auto_const_inverse',
                  'ode_auto_linear1_forward', 'ode_auto_linear1_inverse',
                  'ode_auto_linear2_forward', 'ode_auto_linear2_inverse',
                  'series_damped_oscillator_forward', 'series_damped_oscillator_inverse',
                  'pde_porous_spatial_forward', 'pde_porous_spatial_inverse',
                  'pde_poisson_spatial_forward', 'pde_poisson_spatial_inverse',
                  'pde_cubic_spatial_forward', 'pde_cubic_spatial_inverse',
                  'mfc_gparam_hj_forward11', 
                  'mfc_gparam_hj_forward12',
                  'mfc_gparam_hj_forward22', 
                  'mfc_rhoparam_hj_forward11', 
                  'mfc_rhoparam_hj_forward12']

    equation_type, parameters, conditions, domain, demo_num = (
        equation_dict.get(key) for key in ('equation_type', 'parameters', 'conditions', 'domain', 'demos')
    )

    init = conditions.get('init', [])
    control = conditions.get('control', [])
    domain = domain.reshape(1, domain.shape[0], 1)

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    name = 'interactive'

    if "ode" in equation_type:
        # Create reference demos for the equation
        if "ode_auto_const" in equation_type:
            ode_batch_fn = dyn.ode_auto_const_batch_fn
        if "ode_auto_linear1" in equation_type:
            ode_batch_fn = dyn.ode_auto_linear1_batch_fn
        if "ode_auto_linear2" in equation_type:
            ode_batch_fn = dyn.ode_auto_linear2_batch_fn

        all_ts = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
        quest_control = control[None, ...]
        ts_expand, control, traj = dyn.generate_one_dyn(key = next(rng), ode_batch_fn = ode_batch_fn, 
                                                dt = 1.0/domain.shape[1], length = domain.shape[1], num = demo_num,
                                                k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                coeffs = parameters, control = None)

        quest_ts_expand, quest_control, quest_traj = dyn.generate_one_dyn(key = next(rng), ode_batch_fn = ode_batch_fn, 
                                                dt = 1.0/domain.shape[1], length = domain.shape[1], num = 1,
                                                k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                coeffs = parameters, control = quest_control, init = jnp.array(init))
        print("Control: ", quest_control)
        print("Generated solution: ", quest_traj)

        ts_expand = jnp.concatenate([ts_expand, quest_ts_expand], axis=0)
        control = jnp.concatenate([control, quest_control], axis=0)
        traj = jnp.concatenate([traj, quest_traj], axis=0)
        all_ts.append(ts_expand)
        all_cs.append(control)
        all_us.append(traj)
        all_params.append("_".join([f"{param:.8f}" for param in parameters]))
        all_eqn_captions.append(None)
        datawrite.write_ode_tfrecord(name = f"{FLAGS.dir}/{name}", eqn_type = equation_type.rsplit("_", 1)[0], 
                        all_params = all_params, all_eqn_captions = all_eqn_captions,
                        all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                        problem_type = equation_type.rsplit("_")[-1])
    
    if "damped_oscillator" in equation_type: #broken
        length = 100
        dt = 0.01
        ts = jnp.arange(length*2) * dt/2
        ts_first = einshape("i->jik", ts[:length], j = demo_num, k = 1) # (num, length, 1)
        ts_second = einshape("i->jik", ts[length:], j = demo_num, k = 1) # (num, length, 1)
        decay = parameters[0]

        all_ts_first = []; all_us_first = []; all_ts_second = []; all_us_second = []; all_params = []; all_eqn_captions = []
        amps, periods, phases = jax.random.uniform(
            next(rng), shape=(3, demo_num), minval=jnp.array([0.5, 0.1, 0.0]), maxval=jnp.array([1.5, 0.2, 2*jnp.pi])
        )

        time_series = series.generate_damped_oscillator_batch(ts, amps, periods, phases, decay)
        us_first = time_series[:, :length, None]
        us_second = time_series[:, length:, None]

        quest_amp, quest_period, quest_phase = init
        time_series = series.generate_damped_oscillator_batch(ts, jnp.array([quest_amp]), jnp.array([quest_period]), jnp.array([quest_phase]), decay)
        quest_us_first = time_series[:, :length, None]
        quest_us_second = time_series[:, length:, None]

        us_first = jnp.concatenate([us_first, quest_us_first], axis=0)
        us_second = jnp.concatenate([us_second, quest_us_second], axis=0)

        all_ts_first.append(ts_first)
        all_us_first.append(us_first)
        all_ts_second.append(ts_second)
        all_us_second.append(us_second)
        all_params.append("{:.8f}".format(decay))
        all_eqn_captions.append(None)
        datawrite.write_series_tfrecord(name = f"{FLAGS.dir}/{name}", eqn_type = equation_type.rsplit("_", 1)[0],
                        all_params = all_params, all_eqn_captions = all_eqn_captions,
                        all_ts_first = all_ts_first, all_us_first = all_us_first,
                        all_ts_second = all_ts_second, all_us_second = all_us_second,
                        problem_type = equation_type.rsplit("_")[-1])

    if "poisson" in equation_type:
        N = domain.shape[1] - 1
        L = domain[0,-1,0] - domain[0,0,0]
        all_xs = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []

        # Datagen for demos
        xs = domain[0,:,0] # (N+1,)
        cs = dutils.generate_gaussian_process(next(rng), xs, demo_num, kernel = dutils.rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
        us = pdes.solve_poisson_batch(L, N, init[0], init[1], cs[:,1:-1])

        # Taking care of input example (quest_us not strictly necessary)
        control = control[None, ...]
        quest_us = pdes.solve_poisson_batch(L, N, init[0], init[1], control[:,1:-1])


        cs = jnp.concatenate([cs, control], axis=0)
        us = jnp.concatenate([us, quest_us], axis=0) 

        all_xs.append(einshape("i->jik", xs, j = demo_num + 1, k = 1))
        all_cs.append(einshape("ij->ijk", cs, k = 1))
        all_us.append(einshape("ij->ijk", us, k = 1)) 
        all_params.append("{:.8f}_{:.8f}".format(init[0], init[1]))
        datawrite.write_pde_tfrecord(name = f"{FLAGS.dir}/{name}", eqn_type = equation_type.rsplit("_", 1)[0], 
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                      problem_type = equation_type.rsplit("_")[-1])

    if 'porous' in equation_type:
        # ul and ur being used as parameters, along with a & c

        dx = 0.01
        length = 100
        # length = domain.shape[1] - 1 
        N = length
        L = length * dx
        lamda = 0.05
        all_xs = []; all_ks = []; all_us = []; all_params = []; all_eqn_captions = []
        rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
        coeff_ul, coeff_ur, coeff_c, coeff_a = parameters

        # demos
        xs = domain[0,:,0]
        ks_GP = dutils.generate_gaussian_process(next(rng), xs, demo_num, kernel = dutils.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (demo_num, N+1)
        ks = jax.nn.softplus(ks_GP) # (demo_num, N+1)
        us = pdes.solve_porous_batch(L, N, coeff_ul, coeff_ur, coeff_a * lamda, ks[:,1:-1], coeff_c) # (num, N+1)

        # quest
        quest_ks = control[None, ...]
        quest_us = pdes.solve_porous_batch(L, N, coeff_ul, coeff_ur, coeff_a * lamda, quest_ks[:,1:-1], coeff_c) # (1, N+1)

        ks = jnp.concatenate([ks, quest_ks], axis=0)
        us = jnp.concatenate([us, quest_us], axis=0)

        all_xs.append(einshape("i->jik", xs, j = demo_num+1, k = 1)) # (demo_num, N+1, 1)
        all_ks.append(einshape("ij->ijk", ks, k = 1)) # (demo_num, N+1, 1)
        all_us.append(einshape("ij->ijk", us, k = 1)) # (demo_num, N+1, 1)
        all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
        all_eqn_captions.append(None)
        for ptype in ['forward']:
            datawrite.write_pde_tfrecord(name = f"{FLAGS.dir}/{name}", eqn_type = equation_type.rsplit("_", 1)[0], 
                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                            all_xs = all_xs, all_ks = all_ks, all_us = all_us,
                            problem_type = equation_type.rsplit("_")[-1])

    if 'cubic' in equation_type:
        length = domain.shape[1] - 1 
        N = length
        L = length * 1/N
        lamda = 0.1
        rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
        coeff_ul, coeff_ur, coeff_a, coeff_k = parameters
        all_xs = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []

        # demos
        xs = domain[0,:,0]
        us_GP = dutils.generate_gaussian_process(next(rng), xs, demo_num, kernel = dutils.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (demo_num, N+1)  
        [us,cs] = pdes.solve_cubic_batch(L, N, us_GP, coeff_ul, coeff_ur, coeff_a * lamda, coeff_k) 

        # quest
        quest_cs = control[None, ...]
        quest_us = jnp.zeros((1, N+1))
        cs = jnp.concatenate([cs, quest_cs], axis=0)
        us = jnp.concatenate([us, quest_us], axis=0)

        all_xs.append(einshape("i->jik", xs, j = demo_num+1, k = 1)) # (demo_num+1, N+1, 1)
        all_cs.append(einshape("ij->ijk", cs, k = 1)) # (demo_num+1, N+1, 1)
        all_us.append(einshape("ij->ijk", us, k = 1)) # (demo_num+1, N+1, 1)
        all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_a, coeff_k))
        all_eqn_captions.append(None)
        for ptype in ["forward","inverse"]:
            datawrite.write_pde_tfrecord(name = f"{FLAGS.dir}/{name}", eqn_type = equation_type.rsplit("_", 1)[0],
                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                            all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                            problem_type = equation_type.rsplit("_")[-1])

    return demo_num, equation_type
    
def prepare_model_input(demo_num, equation_type):
    # Prepare the input for the model
    # Return the input as a namedtuple

    utils.set_seed(FLAGS.seed)
    from dataloader import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader
    test_data_dirs= ['data']
    test_data_globs = ["_".join(["interactive", equation_type.rsplit("_", 1)[0], "*"])]
    test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in test_data_globs]

    print("test_file_names: ", flush=True)
    pprint(test_file_names)

    test_config = utils.load_json("config_data/test_lm_interactive_config.json")

    if 'ode' in equation_type:
        test_config['quest_qoi_len'] = 50

    # Updating all demo_nums in config based on user-provided demo_num
    def update_demo_nums(config, new_demo_num):
        def update_recursive(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_recursive(value)
                elif key == 'demo_num_begin':
                    d[key] = new_demo_num
                elif key == 'demo_num_end':
                    d[key] = new_demo_num + 1

        config['demo_num'] = new_demo_num
        update_recursive(config)

    update_demo_nums(test_config, demo_num)
    print('==============data config==============', flush = True)
    print("test_config: ", flush=True)
    pprint(test_config)
    model_config = utils.load_json("config_model/model_lm_config.json")

    if 'cap' not in FLAGS.loss_mode:
        model_config['caption_len'] = 0
        test_config['load_list'] = []

    test_data = DataProvider(seed = FLAGS.seed + 10,
                                config = test_config,
                                file_names = test_file_names,
                                batch_size = demo_num+1,
                                deterministic = True,
                                drop_remainder = False, 
                                shuffle_dataset = False,
                                num_epochs=1,
                                shuffle_buffer_size=0,
                                num_devices=len(jax.devices()),
                                real_time = True,
                                caption_home_dir = 'data_preparation',
                            )
    
    raw, equation, caption, data, label = test_data.get_next_data(return_raw = True)
    print_eqn_caption(equation, caption, decode = False)
    return raw, equation, caption, data, label, model_config, test_config
    
def run_inference(raw, equation, caption, data, label, model_config, test_config):

    utils.set_seed(FLAGS.seed)    
    optimizer = optax.adamw(0.0001) # dummy optimizer

    from runner_jax import Runner_lm
    runner = Runner_lm(seed = seed,
                    model = 'icon_lm',
                    data = data,
                    model_config = model_config,
                    optimizer = optimizer,
                    trainable_mode = 'all',
                    loss_mode = ['nocap'],
                    devices = jax.devices()
                    )

    runner.restore(save_dir='jamie/ckpts/icon_lm/20240716-143836', step=900000, restore_opt_state = False)
    data_replicated = jax.device_put_replicated(tree.tree_map(lambda x: x[None, ...], data), runner.devices)
    prediction = runner.get_pred(data_replicated, with_caption=False)
    return prediction

def solve_differential_equation(equation_dict):
    demo_num, equation_type = parse_equation(equation_dict)
    raw, equation, caption, data, label, model_config, test_config = prepare_model_input(demo_num, equation_type)
    prediction = run_inference(raw, equation, caption, data, label, model_config, test_config)
    os.remove('data/interactive_{}_forward.tfrecord'.format(equation_type.rsplit("_", 1)[0]))
    return prediction[0,0,:,0]

def main(argv):
    for key, value in FLAGS.__flags.items():
        print(value.name, ": ", value._value, flush=True)

    ode_domain = jnp.linspace(0, 1, 50, endpoint=False)
    ode_domain = jnp.array(ode_domain)
    rounded_ode_domain = jnp.array([float(f'{x:.2g}') for x in ode_domain])

    equation_dict = {'equation_type': 'ode_auto_const_forward',
                    'domain': rounded_ode_domain,
                    'parameters': [float(1.0), float(0.2)],
                    'conditions': {'init': [float(0.0)],
                                    'control': jnp.linspace(0, 1, 50, endpoint=False)},
                    'demos': int(3)}
    utils.set_seed(FLAGS.seed + 123456) 
    prediction = solve_differential_equation(equation_dict)
    print("Prediction on equation dict")
    print(prediction)
    print("Shape: ", prediction.shape)  

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('seed', 0, 'random seed')
    flags.DEFINE_string('test_config_filename', 'test_lm_config.json', 'test config filename')
    flags.DEFINE_string('model_config_filename', 'model_lm_config.json', 'model config filename')
    flags.DEFINE_list('loss_mode', ['nocap'], 'loss mode')
    flags.DEFINE_list('write', [], 'list of features to write')
    flags.DEFINE_string('dir', 'data', 'name of the directory to save the data')
    app.run(main) 