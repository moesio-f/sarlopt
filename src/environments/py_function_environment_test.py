"""PyFunctionEnvironment validation tests."""
import numpy as np
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.environments import wrappers

from optfuncs import numpy_functions as npf
from optfuncs import tensorflow_functions as tff

from src.environments import py_function_environment as py_fun_env
from src.environments import py_env_wrappers

if __name__ == '__main__':
  function = npf.Sphere()
  dims = 500

  env = py_fun_env.PyFunctionEnv(function=function,
                                 dims=dims,
                                 bounded_actions_spec=True)
  env = py_env_wrappers.RewardClip(env=env,
                                   min_reward=-400.0,
                                   max_reward=400.0)
  env = py_env_wrappers.RewardScale(env=env,
                                    scale_factor=0.2)
  env = wrappers.TimeLimit(env=env,
                           duration=500)

  utils.validate_py_environment(env,
                                episodes=50)

  unbounded_env = py_fun_env.PyFunctionEnv(function=function,
                                           dims=dims)
  unbounded_env = py_env_wrappers.RewardClip(env=unbounded_env,
                                             min_reward=-400.0,
                                             max_reward=400.0)
  unbounded_env = py_env_wrappers.RewardScale(env=unbounded_env,
                                              scale_factor=0.2)
  unbounded_env = wrappers.TimeLimit(env=unbounded_env,
                                     duration=500)

  utils.validate_py_environment(unbounded_env,
                                episodes=50)

  tf_fun = npf.get_tf_function(function)


  def grad_fn(x: np.ndarray):
    t_x = tf.convert_to_tensor(x, dtype=x.dtype)
    grad, _ = tff.get_grads(tf_fun, t_x)
    return grad.numpy()


  env_v1 = py_fun_env.PyFunctionEnvV1(function,
                                      dims,
                                      grad_fn)

  env_v1 = wrappers.TimeLimit(env=env_v1,
                              duration=500)

  utils.validate_py_environment(env_v1,
                                episodes=50)
