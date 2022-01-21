"""Tests for src.environments.tf_function_env_v2."""
import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.typing import types

from optfuncs import core
from optfuncs import tensorflow_functions as tff

from src.environments.tf_function_env_v2 import TFFunctionEnvV2
from src.utils.random import functions as rand_fn


class DummyFunction(tff.TensorflowFunction):
  def __init__(self):
    super(DummyFunction, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: types.Tensor):
    return tf.reduce_sum(x)


class TFFunctionEnvV2Test(tf.test.TestCase):
  def setUp(self):
    super(TFFunctionEnvV2Test, self).setUp()

    self.seed = 100
    self.alg = tf.random.Algorithm.PHILOX

    self.dims = 50
    self.duration = 100
    self.n_fns = 9
    self.function: tff.TensorflowFunction = DummyFunction()
    self.functions = [self.function] + \
                     rand_fn.random_shifted_functions(
                       src_fn=self.function,
                       hshift_bounds=(-10.0, 10.0),
                       vshift_bounds=(-10.0, 10.0),
                       n=self.n_fns,
                       seed=self.seed)

  def test_first_reset(self):
    env = TFFunctionEnvV2(functions=self.functions,
                          dims=self.dims,
                          seed=self.seed,
                          alg=self.alg,
                          duration=self.duration)
    time_step = env.current_time_step()
    expected = tf.constant(
      [[33.23932, 49.381165, 75.386505, 69.548325, -54.580425,
        -1.4862518, 53.653366, 60.19606, 55.697586, 44.985992,
        79.50868, 59.27188, 24.243118, -95.93835, 26.414062,
        12.058708, 27.26529, -22.819427, -10.332039, -48.80364,
        49.507904, 8.303406, 14.995102, 24.299286, 66.766174,
        85.62308, 76.10106, 83.33211, -60.555695, 32.780502,
        -62.55419, -33.849716, -87.77144, 39.871567, 40.36557,
        91.0177, -61.283684, 25.074791, -80.623795, -61.81326,
        99.03938, 21.42575, -65.36798, 95.82617, 29.573914,
        74.07785, -60.790108, -31.174042, -66.323135, 57.77748]],
      dtype=tf.float32)
    self.evaluate(time_step)
    self.assertAllClose(expected, time_step.observation)

  def test_validate_next_state(self):
    env = TFFunctionEnvV2(functions=self.functions,
                          dims=self.dims,
                          seed=self.seed,
                          alg=self.alg,
                          duration=self.duration)
    action = -tf.constant(
      [[33.23932, 49.381165, 75.386505, 69.548325, -54.580425,
        -1.4862518, 53.653366, 60.19606, 55.697586, 44.985992,
        79.50868, 59.27188, 24.243118, -95.93835, 26.414062,
        12.058708, 27.26529, -22.819427, -10.332039,
        -48.80364,
        49.507904, 8.303406, 14.995102, 24.299286, 66.766174,
        85.62308, 76.10106, 83.33211, -60.555695, 32.780502,
        -62.55419, -33.849716, -87.77144, 39.871567, 40.36557,
        91.0177, -61.283684, 25.074791, -80.623795, -61.81326,
        99.03938, 21.42575, -65.36798, 95.82617, 29.573914,
        74.07785, -60.790108, -31.174042, -66.323135,
        57.77748]], dtype=tf.float32)
    env.step(action)
    time_step = env.current_time_step()
    self.evaluate(time_step)

    new_state = tf.zeros(shape=(1, self.dims,))
    fn_index = self.evaluate(env.fn_index)
    reward = -tf.expand_dims(env.functions[fn_index](new_state), axis=0)

    self.assertEqual(ts.StepType.MID, time_step.step_type)
    self.assertEqual(1.0, time_step.discount)
    self.assertAllEqual(new_state, time_step.observation)
    self.assertEqual(reward, time_step.reward)

  def test_validate_state_clip(self):
    env = TFFunctionEnvV2(functions=self.functions,
                          dims=self.dims,
                          seed=self.seed,
                          alg=self.alg,
                          duration=self.duration)
    action = tf.expand_dims(tf.repeat(tf.constant([500.0], dtype=tf.float32),
                                      repeats=(self.dims,)), axis=0)
    env.step(action)
    time_step = env.current_time_step()
    self.evaluate(time_step)
    self.assertAllEqual(
      tf.expand_dims(tf.repeat(tf.constant([100.0], dtype=tf.float32),
                               repeats=(self.dims,)), axis=0),
      time_step.observation)

  def testRunEpisode(self):
    env = TFFunctionEnvV2(functions=self.functions,
                          dims=self.dims,
                          seed=self.seed,
                          alg=self.alg,
                          duration=self.duration)
    action = tf.expand_dims(tf.repeat(tf.constant([1.0], dtype=tf.float32),
                                      repeats=(self.dims,)), axis=0)
    c = lambda t: tf.logical_not(t.is_last())
    body = lambda t: [env.step(action)]

    @common.function
    def run_episode():
      time_step = env.reset()
      return tf.while_loop(cond=c, body=body, loop_vars=[time_step])

    [final_time_step_np] = self.evaluate(run_episode())
    initial_state = tf.constant(
      [[38.69687, - 9.153366, 68.126175, - 77.681755, 94.52925,
        63.070038, 12.1708145, 11.7862015, - 40.68613, - 3.2457352,
        36.02977, - 63.711525, 22.908043, - 41.315987, 75.17453,
        - 71.203255, - 20.189499, 0.17462158, 60.847565, 49.406403,
        - 18.182495, - 46.37158, 87.18805, 4.4888, 68.32158,
        - 75.7519, 43.21167, - 15.483788, - 76.91519, - 78.767204,
        - 53.06172, - 18.491432, - 0.28393555, - 68.8405, 48.194,
        80.235916, 32.953766, - 76.265076, - 36.023666, 75.106186,
        - 81.95667, 2.1315613, - 21.04721, - 28.40245, 4.5762024,
        73.19539, - 56.4003, 0.22244263, - 62.759636, - 51.149796]],
      dtype=tf.float32)
    final_state = tf.clip_by_value(initial_state + self.duration * action,
                                   clip_value_min=self.function.domain.min,
                                   clip_value_max=self.function.domain.max)
    self.assertEqual(ts.StepType.LAST, final_time_step_np.step_type)
    self.assertAllEqual(final_state, final_time_step_np.observation)


if __name__ == '__main__':
  tf.test.main()
