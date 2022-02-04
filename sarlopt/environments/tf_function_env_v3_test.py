"""Tests for src.environments.tf_function_env_v3."""
from math import pi

import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.typing import types
from tf_agents.policies import random_tf_policy

from optfuncs import core
from optfuncs import tensorflow_functions as tff

from sarlopt.environments.tf_function_env_v3 import TFFunctionEnvV3
from sarlopt.utils.functions import distributions as fn_distributions


class SepUnimodalF1(tff.TensorflowFunction):
  def __init__(self):
    super(SepUnimodalF1, self).__init__(core.Domain(-50.0, 100.0))

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(x)


class SepUnimodalF2(tff.TensorflowFunction):
  def __init__(self):
    super(SepUnimodalF2, self).__init__(core.Domain(-200.0, 300.0))

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(tf.pow(x, 2))


class SepMultimodalF1(tff.TensorflowFunction):
  def __init__(self):
    super(SepMultimodalF1, self).__init__(core.Domain(-180.0, 250.2))

  def _call(self, x: tf.Tensor):
    return tf.reduce_prod(tf.multiply(tf.sqrt(x), tf.sin(x)), axis=-1)


class SepMultimodalF2(tff.TensorflowFunction):
  def __init__(self):
    super(SepMultimodalF2, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return -tf.divide(tf.reduce_sum(tf.pow(tf.sin(tf.multiply(x, 5 * pi)), 6),
                                    axis=-1), d)


class TFFunctionEnvV3Test(tf.test.TestCase):
  def setUp(self):
    super(TFFunctionEnvV3Test, self).setUp()

    self.seed = 100
    self.alg = tf.random.Algorithm.PHILOX

    self.dims = 2
    self.duration = 100
    self.functions = [[SepUnimodalF1(), SepUnimodalF2()],
                      [SepMultimodalF1(), SepMultimodalF2()]]
    self.hshift_bounds = (-10.0, 10.0)
    self.vshift_bounds = (-100.0, 100.0)
    self.scale_bounds = (-0.5, 2.0)
    self.action_bounds = None

  def test_run_episode(self):
    fn_dist = fn_distributions.UniformFunctionDistribution(
      functions=self.functions,
      rng_seed=self.seed,
      hshift_bounds=self.hshift_bounds,
      vshift_bounds=self.vshift_bounds,
      scale_bounds=self.scale_bounds)
    fn_dist.enable_tf_function()

    env = TFFunctionEnvV3(fn_dist=fn_dist,
                          duration=self.duration,
                          seed=self.seed,
                          dims=self.dims,
                          action_bounds=self.action_bounds)
    action = tf.expand_dims(tf.repeat(tf.constant(1.0, dtype=tf.float32),
                                      repeats=(self.dims,)), axis=0)
    c = lambda t: tf.logical_not(t.is_last())
    body = lambda t: [env.step(action)]

    @common.function
    def run_episode():
      time_step = env.reset()
      return tf.while_loop(cond=c, body=body, loop_vars=[time_step])

    [final_time_step_np] = self.evaluate(run_episode())
    log_abs_grads = tf.constant([[0.61834264, 0.6255897]],
                                dtype=tf.float32)
    sign_grads = tf.constant([[1.0, 1.0]],
                             dtype=tf.float32)
    final_observation = tf.concat([log_abs_grads, sign_grads], axis=-1)

    self.assertEqual(ts.StepType.LAST, final_time_step_np.step_type)
    self.assertAllClose(final_observation, final_time_step_np.observation)


if __name__ == '__main__':
  tf.test.main()
