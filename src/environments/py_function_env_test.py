"""Tests for src.environments.py_function_environment."""

import numpy as np
from optfuncs import core

from tf_agents.environments import utils as env_utils
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import test_utils

from src.environments.py_function_env import PyFunctionEnv


class DummyFunction(core.Function):
  def __init__(self):
    super(DummyFunction, self).__init__(core.Domain(-100.0, 100.0))

  def __call__(self, x: np.ndarray):
    return np.sum(x)


class PyFunctionEnvTest(test_utils.TestCase):

  def setUp(self):
    super(PyFunctionEnvTest, self).setUp()

    self.seed = 100

    self.function = DummyFunction()
    self.dims = 50

    PyFunctionEnv.MAX_STEPS = 100
    self.env = PyFunctionEnv(function=self.function,
                             dims=self.dims,
                             seed=self.seed)

    ts = self.env.reset()
    state = np.asarray(
      [19.429054, -70.22521, 12.009726, 12.633504,
       16.197638, -63.0102, 41.923782, -85.99587,
       -35.02701, -78.62891, 49.60574, 0.31480148,
       -6.7383795, -85.26132, -11.580723, -55.297783,
       -39.404854, -21.130398, 41.32613, 57.27263,
       -28.63278, 34.02597, 2.044624, 34.229053,
       -33.933586, 79.683914, 32.558403, -61.64053,
       15.031705, 49.880577, 55.985065, -12.453688,
       -44.337463, 23.204817, 45.980267, 1.4470294,
       -63.07843, 22.85883, 52.070415, 83.00289,
       98.88976, -26.34415, 61.119022, 37.90722,
       94.587296, 1.7535747, 72.1283, 0.80926585,
       -42.34945, 92.07652], dtype=np.float32)

    np.testing.assert_array_equal(state, ts.observation)

  def test_validate_specs(self):
    self.env = PyFunctionEnv(function=self.function,
                             dims=self.dims,
                             seed=self.seed)
    env_utils.validate_py_environment(self.env, episodes=10)

  def test_validate_rewards(self):
    states = [np.zeros((self.dims,)).astype(np.float32),
              np.repeat(-100.0, (self.dims,)).astype(np.float32),
              np.repeat(100.0, (self.dims,)).astype(np.float32)]

    actions = [np.zeros((self.dims,)).astype(np.float32),
               np.ones((self.dims,)).astype(np.float32)]

    for action in actions:
      for state in states:
        self.env.set_state((state - action, 0, False))
        ts = self.env.step(action)
        self.assertEqual(-self.function(state),
                         ts.reward)
        self.assertEqual(StepType.MID, ts.step_type)

  def test_validate_state_clip(self):
    state = np.repeat(99.0, (self.dims,)).astype(np.float32)
    action = np.repeat(2.0, (self.dims,)).astype(np.float32)
    next_state = np.repeat(100.0, (self.dims,)).astype(np.float32)

    for m in [1.0, -1.0]:
      self.env.set_state((m * state, 0, False))
      self.env.step(m * action)
      self.assertAllEqual(m * next_state,
                          self.env.current_time_step().observation)

  def test_validate_episode_end(self):
    self.env = PyFunctionEnv(function=self.function,
                             dims=self.dims,
                             seed=self.seed)
    ts = self.env.reset()
    steps = 0
    action = np.ones((self.dims,)).astype(np.float32)

    while not ts.is_last():
      ts = self.env.step(action)
      steps += 1

    self.assertEqual(PyFunctionEnv.MAX_STEPS,
                     steps)


if __name__ == '__main__':
  test_utils.main()
