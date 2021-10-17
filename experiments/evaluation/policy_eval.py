"""Policy evaluation tests."""

import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from optfuncs import numpy_functions as npf

from src.environments import py_function_environment as py_fun_env
from src import config

from experiments.evaluation import utils as eval_utils

POLICIES_DIR = config.POLICIES_DIR

if __name__ == '__main__':
  function = npf.Sphere()
  dims = 30
  steps = 500

  policy_dir = os.path.join(POLICIES_DIR, 'Td3Agent')
  policy_dir = os.path.join(policy_dir, f'{dims}D')
  policy_dir = os.path.join(policy_dir, function.name)

  saved_pol = tf.compat.v2.saved_model.load(policy_dir)

  env = py_fun_env.PyFunctionEnv(function, dims)
  env = wrappers.TimeLimit(env, duration=steps)

  tf_eval_env = tf_py_environment.TFPyEnvironment(environment=env)

  # tf.config.run_functions_eagerly(True)
  eval_utils.evaluate_agent(eval_env=tf_eval_env,
                            policy_eval=saved_pol,
                            function=function,
                            dims=dims,
                            steps=steps,
                            algorithm_name='TD3',
                            save_to_file=False,
                            episodes=10,
                            save_dir=os.getcwd())
