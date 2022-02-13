"""TFEnvironment for function optimization with RL using POMDP and
  function distribution."""

import typing

import tensorflow as tf
from tensorflow.python.autograph.impl import api as autograph
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common, nest_utils

from sarlopt.utils.functions import distributions as fn_distributions

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


class TFFunctionEnvV3(tf_environment.TFEnvironment):
  """Single-agent function environment as a POMDP for Learning global
  optimization.
    Observations: log(|grad_t|), sng(grad_t), (x_t - x_0)/t.
    Actions: deltaX.
    States: x_t,
            fx_t,
            {(x_0, fx_0, grads_0), ..., (x_{t-1}, fx_{t-1}, grads_{t-1})},
            other unknown hidden states.
    Reward: R(s_t, a_t, s_{t+1}) = -sgn(y)*log_10(1 + |y*ln(10)|),
            where y = fx_{t+1}
    """

  def __init__(self,
               fn_dist: fn_distributions.UniformFunctionDistribution,
               dims,
               seed,
               duration: int = 50000,
               action_bounds: typing.Tuple[float, float] = None,
               alg=tf.random.Algorithm.PHILOX):
    self._fn_dist = fn_dist
    self._fn_dist.enable_tf_function()

    self._domain = self._fn_dist.current_domain
    self._dims = dims

    if action_bounds is not None:
      a_min, a_max = action_bounds
      action_spec = specs.BoundedTensorSpec(shape=tf.TensorShape([self._dims]),
                                            dtype=tf.float32,
                                            minimum=a_min,
                                            maximum=a_max,
                                            name='action')
    else:
      action_spec = specs.TensorSpec(shape=tf.TensorShape([self._dims]),
                                     dtype=tf.float32,
                                     name='action')
    # 3 * dims: [grad_mag_dims] + [grad_sign_dims] + [velocity_dim]
    observation_spec = specs.TensorSpec(shape=tf.TensorShape([3 * self._dims]),
                                        dtype=tf.float32,
                                        name='observation')

    time_step_spec = ts.time_step_spec(observation_spec)
    super().__init__(time_step_spec, action_spec)

    self._seed = seed
    self._alg = alg

    self._rng = tf.random.Generator.from_seed(self._seed, self._alg)

    self._episode_ended = common.create_variable(name='episode_ended',
                                                 initial_value=False,
                                                 dtype=tf.bool)
    self._steps_taken = common.create_variable(name='steps_taken',
                                               initial_value=0,
                                               dtype=tf.int32)
    self._duration = tf.constant(value=duration,
                                 dtype=tf.int32,
                                 name='duration')

    # States
    self._x0 = common.create_variable(name='x0',
                                      shape=tf.TensorShape([self._dims]),
                                      initial_value=0,
                                      dtype=tf.float32)
    self._x = common.create_variable(name='x',
                                     shape=tf.TensorShape([self._dims]),
                                     initial_value=0,
                                     dtype=tf.float32)
    self._fx = common.create_variable(name='fx',
                                      shape=(),
                                      initial_value=0,
                                      dtype=tf.float32)

    # Observations
    self._grads_at_x = common.create_variable(
      name='grads_at_x',
      shape=tf.TensorShape([self._dims]),
      initial_value=0,
      dtype=tf.float32)

    self._avg_velocity = common.create_variable(
      name='avg_velocity',
      shape=tf.TensorShape([self._dims]),
      initial_value=0,
      dtype=tf.float32)

    # Observations utilities
    self._p = tf.constant(10, dtype=tf.float32)

    def log_processing(g: tf.Tensor):
      abs_g = tf.abs(g)
      return tf.cond(pred=tf.greater_equal(abs_g,
                                           tf.math.exp(-self._p)),
                     true_fn=lambda: tf.math.divide(tf.math.log(abs_g),
                                                    self._p),
                     false_fn=lambda: tf.constant(-1.0, dtype=tf.float32))

    def sign_processing(g: tf.Tensor):
      return tf.cond(pred=tf.greater_equal(tf.abs(g),
                                           tf.math.exp(-self._p)),
                     true_fn=lambda: tf.sign(g),
                     false_fn=lambda: tf.multiply(g,
                                                  tf.math.exp(-self._p)))

    self._log_processing_fn = log_processing
    self._sign_processing_fn = sign_processing

    # Reward function utilities
    self._b = tf.constant(10, dtype=tf.float32)
    self._c = tf.constant(tf.math.reciprocal(tf.math.log(self._b)),
                          dtype=tf.float32)

    def bi_symmetrical_log(fx):
      log_10 = tf.divide(
        tf.math.log(tf.math.add(tf.abs(fx / self._c), 1)),
        tf.math.log(self._b))
      return tf.math.multiply(tf.sign(fx), log_10)

    self._reward_transformation_fn = bi_symmetrical_log

  def _current_time_step(self) -> ts.TimeStep:
    grads = self._grads_at_x.value()
    fx = self._fx.value()
    x0 = self._x0.value()
    x = self._x.value()
    t = tf.cast(self._steps_taken.value(), dtype=tf.float32)

    with tf.control_dependencies([grads]):
      log_grad = tf.map_fn(self._log_processing_fn, grads)
      sign_grad = tf.map_fn(self._sign_processing_fn, grads)

    with tf.control_dependencies([x0, x, t]):
      avg_velocity = tf.math.divide_no_nan(x - x0, t)

    with tf.control_dependencies([log_grad, sign_grad, avg_velocity]):
      observation = tf.concat([log_grad, sign_grad, avg_velocity], axis=-1,
                              name='observation_concat')

    def first():
      return (tf.constant(FIRST, dtype=tf.int32),
              tf.constant(0.0, dtype=tf.float32))

    def mid():
      return (tf.constant(MID, dtype=tf.int32),
              tf.reshape(-self._reward_transformation_fn(fx),
                         shape=()))

    def last():
      return (tf.constant(LAST, dtype=tf.int32),
              tf.reshape(-self._reward_transformation_fn(fx),
                         shape=()))

    with tf.control_dependencies([fx]):
      discount = tf.constant(1.0, dtype=tf.float32)
      step_type, reward = tf.case(
        [(tf.math.less_equal(self._steps_taken, 0), first),
         (tf.math.reduce_any(self._episode_ended), last)],
        default=mid,
        exclusive=True, strict=True)

      with tf.control_dependencies([observation]):
        return nest_utils.batch_nested_tensors(
          ts.TimeStep(step_type=step_type,
                      reward=reward,
                      discount=discount,
                      observation=observation),
          self.time_step_spec())

  def _reset(self) -> ts.TimeStep:
    reset_ended = self._episode_ended.assign(value=False)
    reset_steps = self._steps_taken.assign(value=0)
    self._fn_dist.sample()
    current_domain = self._fn_dist.current_domain

    with tf.control_dependencies([reset_ended,
                                  reset_steps,
                                  current_domain]):
      rng_x0 = self._rng.uniform(
        shape=tf.TensorShape([self._dims]),
        minval=current_domain[0],
        maxval=current_domain[1],
        dtype=tf.float32)

      with tf.control_dependencies([rng_x0]):
        x0_reset = self._x0.assign(value=rng_x0)
        x_reset = self._x.assign(value=rng_x0)

      with tf.control_dependencies([x_reset, x0_reset]):
        grad, fx = self._fn_dist.grads_at(x0_reset.value())

        with tf.control_dependencies([fx, grad]):
          fx_reset = self._fx.assign(value=fx)
          grads_x_reset = self._grads_at_x.assign(value=grad)

          with tf.control_dependencies([fx_reset,
                                        grads_x_reset]):
            time_step = self.current_time_step()
            return time_step

  def _step(self, action):
    action = tf.squeeze(tf.convert_to_tensor(value=action))

    def take_step():
      steps_update = self._steps_taken.assign_add(1)
      episode_finished = tf.cond(
        pred=tf.math.greater_equal(self._steps_taken, self._duration),
        true_fn=lambda: self._episode_ended.assign(True),
        false_fn=self._episode_ended.value)

      with tf.control_dependencies([action, self._domain]):
        new_x = tf.clip_by_value(self._x + action,
                                 clip_value_min=self._domain[0],
                                 clip_value_max=self._domain[1])

      with tf.control_dependencies([new_x]):
        x_update = self._x.assign(new_x)
        grads, fx = self._fn_dist.grads_at(new_x)

        with tf.control_dependencies([fx, grads]):
          fx_update = self._fx.assign(fx)
          grads_update = self._grads_at_x.assign(grads)

      with tf.control_dependencies([x_update,
                                    fx_update,
                                    grads_update,
                                    steps_update,
                                    episode_finished]):
        return self.current_time_step()

    def reset_env():
      return self.reset()

    return tf.cond(pred=tf.math.reduce_any(self._episode_ended),
                   true_fn=reset_env,
                   false_fn=take_step)

  @autograph.do_not_convert()
  def get_info(self, to_numpy=False):
    raise NotImplementedError("No info available for this environment.")

  def render(self):
    raise ValueError('Environment does not support render yet.')
