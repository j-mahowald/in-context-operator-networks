from flask import Flask, render_template, request, jsonify
from absl import flags, app as absl_app
import jax.numpy as jnp
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Import the necessary functions from interactive.py
from interactive import solve_differential_equation

flask_app = Flask(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_string('test_config_filename', 'test_lm_config.json', 'test config filename')
flags.DEFINE_string('model_config_filename', 'model_lm_config.json', 'model config filename')
flags.DEFINE_list('loss_mode', ['nocap'], 'loss mode')
flags.DEFINE_list('write', [], 'list of features to write')
flags.DEFINE_string('dir', 'data', 'name of the directory to save the data')

# We need to call this function to actually parse the flags
flags.FLAGS((['dummy_arg'] + []))

@flask_app.route('/')
def index():
    return render_template('index.html', 
                           seed=FLAGS.seed,
                           test_config_filename=FLAGS.test_config_filename,
                           model_config_filename=FLAGS.model_config_filename,
                           loss_mode=FLAGS.loss_mode,
                           write=FLAGS.write,
                           dir=FLAGS.dir)

@flask_app.route('/solve', methods=['POST'])
def solve():
    data = request.json

    ode_domain = jnp.linspace(0, 1, 50, endpoint=False)
    ode_domain = jnp.array(ode_domain)
    rounded_ode_domain = jnp.array([float(f'{x:.2g}') for x in ode_domain])

    # Update the ode_auto_const dictionary with user input
    # domain imported from interactive (for now)
    if data['equationType'] == 'ode-auto-const':
        equation_dict = {'equation_type': 'ode_auto_const_forward',
                          'domain': rounded_ode_domain,
                          'parameters': [float(data['param1']), float(data['param2'])],
                          'conditions': {'init': [float(data['init1'])],
                                         'control': jnp.array([float(x) for x in data['control'].split(',')])},
                          'demos': int(data['demos'])}

    if data['equationType'] == 'ode-auto-linear1':
        equation_dict = {'equation_type': 'ode_auto_linear1_forward',
                            'domain': rounded_ode_domain,
                            'parameters': [float(data['param1']), float(data['param2'])],
                            'conditions': {'init': [float(data['init1'])],
                                           'control': jnp.array([float(x) for x in data['control'].split(',')])},
                            'demos': int(data['demos'])}

    if data['equationType'] == 'ode-auto-linear2':
        equation_dict = {'equation_type': 'ode_auto_linear2_forward',
                            'domain': rounded_ode_domain,
                            'parameters': [float(data['param1']), float(data['param2']), float(data['param3'])],
                            'conditions': {'init': [float(data['init1'])],
                                           'control': jnp.array([float(x) for x in data['control'].split(',')])},
                            'demos': int(data['demos'])}

    if data['equationType'] == 'series-damped-oscillator':
        equation_dict = {'equation_type': 'series_damped_oscillator_forward',
                            'domain': jnp.linspace(0, 0.5, 101),
                            'parameters': [float(data['param1'])],
                            'conditions': {'init': [float(data['amp']), float(data['period']), float(data['phase'])],
                                            'control': []},
                            'demos': int(data['demos'])}


    if data['equationType'] == 'pde-poisson-spatial':
        equation_dict = {'equation_type': 'pde_poisson_spatial_forward',
                               'domain': jnp.linspace(0.0, 1.0, 100, endpoint=False),
                               'parameters': [],
                               'conditions': {'init': [float(data['init1']), float(data['init2'])],
                                              'control': data['control']},
                               'demos': int(data['demos'])}

    if data['equationType'] == 'pde-porous-spatial':
        equation_dict = {'equation_type': 'pde_porous_spatial_forward',
                              'domain': jnp.linspace(0.0, 1.0, 101, endpoint=True),
                              'parameters': [float(data['init1']), float(data['init2']), float(data['constant']), float(data['param1'])],
                              'conditions': {'init': [],
                                             'control': data['control']},
                              'demos': int(data['demos'])}
        

    if data['equationType'] == 'pde-cubic-spatial':
        equation_dict = {'equation_type': 'pde_cubic_spatial',
                            'domain': jnp.linspace(0.0, 1.0, 101, endpoint=True),
                            'parameters': [float(data['init1']), float(data['init2']), float(data['param1']), float(data['constant'])],
                            'conditions': {'init': [],
                                           'control': data['control']},
                            'demos': int(data['demos'])}


    # Solve the differential equation
    prediction = solve_differential_equation(equation_dict)

    # Convert the prediction to a list for JSON serialization
    prediction_list = prediction.tolist()

    return jsonify({
        'prediction': prediction_list,
        'domain': equation_dict['domain'].tolist()
    })

def main(argv):
    flask_app.run(debug=True)

if __name__ == '__main__':
    absl_app.run(main)