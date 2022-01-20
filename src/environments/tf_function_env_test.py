"""Tests for src.environments.py_function_environment."""
import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.typing import types

from optfuncs import core
from optfuncs import tensorflow_functions as tff

from src.environments.tf_function_env import TFFunctionEnv


class DummyFunction(tff.TensorflowFunction):
  def __init__(self):
    super(DummyFunction, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: types.Tensor):
    return tf.reduce_sum(x)


class TFFunctionEnvTest(tf.test.TestCase):
  def setUp(self):
    super(TFFunctionEnvTest, self).setUp()

    self.seed = 100
    self.alg = tf.random.Algorithm.PHILOX

    self.function = DummyFunction()
    self.function.enable_tf_function()
    self.dims = 50
    self.duration = 100

  def test_first_reset(self):
    env = TFFunctionEnv(function=self.function,
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
    env = TFFunctionEnv(function=self.function,
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
    new_state = tf.zeros(shape=(1, self.dims,))
    reward = self.function(new_state)
    env.step(action)
    time_step = env.current_time_step()
    self.evaluate(time_step)
    self.evaluate(reward)
    self.assertEqual(ts.StepType.MID, time_step.step_type)
    self.assertEqual(1.0, time_step.discount)
    self.assertEqual(reward, time_step.reward)
    self.assertAllEqual(new_state, time_step.observation)

  def test_validate_state_clip(self):
    env = TFFunctionEnv(function=self.function,
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
    env = TFFunctionEnv(function=self.function,
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
      [[52.41487, -91.37347, -6.9405823, -49.872803, -86.09543, -69.27092,
        35.470764, 42.229507, 33.856323, -84.4501, 57.46176, 68.576294,
        40.721725, -39.526485, 92.992096, - 1.1740723, 99.2688, -38.001396,
        -56.528687, -28.987595, 94.98648, 55.953674, 86.02985, -6.362274,
        -9.186363, 31.337708, -15.002129, -86.84091, -59.088158, -92.581535,
        -5.079315, 81.50101, 19.932983, -55.993484, -2.8683624, -42.346786,
        82.11188, 95.5443, -81.82397, -98.573875, -38.047432, -33.21199,
        -71.86837, 9.218529, 26.496216, 95.000534, 58.148575, 4.2685013,
        -47.305916, -53.625034]],
      dtype=tf.float32)
    final_state = tf.clip_by_value(initial_state + self.duration * action,
                                   clip_value_min=self.function.domain.min,
                                   clip_value_max=self.function.domain.max)
    self.assertEqual(ts.StepType.LAST, final_time_step_np.step_type)
    self.assertAllEqual(final_state, final_time_step_np.observation)


if __name__ == '__main__':
  tf.test.main()
