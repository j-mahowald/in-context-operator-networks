from flask import Flask, render_template, request, jsonify
from absl import flags, app as absl_app
import jax.numpy as jnp
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

# Import the necessary functions from interactive.py
from interactive import solve_differential_equation, ode_auto_const

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
    
    # Update the ode_auto_const dictionary with user input
    ode_auto_const['parameters'] = [float(data['param1']), float(data['param2'])]
    ode_auto_const['conditions']['init'] = [float(data['init'])]
    ode_auto_const['conditions']['control'] = jnp.array([float(x) for x in data['control'].split(',')])
    ode_auto_const['demos'] = int(data['demos'])

    # Solve the differential equation
    prediction = solve_differential_equation(ode_auto_const)

    # Convert the prediction to a list for JSON serialization
    prediction_list = prediction.tolist()

    return jsonify({
        'prediction': prediction_list,
        'domain': ode_auto_const['domain'].tolist()
    })

def main(argv):
    flask_app.run(debug=True)

if __name__ == '__main__':
    absl_app.run(main)