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


def min_max_scaling(value: tf.Tensor,
                    min_val: tf.Tensor,
                    max_val: tf.Tensor) -> tf.Tensor:
  return tf.divide(value - min_val, max_val - min_val)


class TFFunctionEnvV3(tf_environment.TFEnvironment):
  """Single-agent function environment as a POMDP for smooth convex functions.
    Observations are log(|grad(x)|), sng(grad(x)).
    Actions are dX.
    States (s) are x, grad_at(x), f(x), expected_min, expected_max plus
      other hidden (unknown) states.
    Reward function is
      R(s_t, a_t, s_{t+1}) = clipped(f(x_t) - f(x_{t+1}))
        + min_max_scaling(f(x_{t+1}))  + bonus
        where bonus = 100, if d < 0
                        d, 0 <= d < 0.5
                        0, otherwise
            let
              d = f(x_{t+1}) - expected_min

    bonus should encourage getting closer (and better than expected min)
    Generalization of hidden states would be keep running statistics of the
      function for the iterates.
    """

  def __init__(self,
               fn_dist: fn_distributions.UniformFunctionDistribution,
               dims,
               seed,
               duration: int = 50000,
               action_bounds: typing.Tuple[float, float] = None,
               num_x_samples_reset: int = 100,
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

    observation_spec = specs.TensorSpec(shape=tf.TensorShape([2*self._dims]),
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
    self._num_x_samples_reset = tf.constant(value=num_x_samples_reset,
                                            dtype=tf.int32,
                                            name="num_x_samples_reset")

    # Hidden state
    self._x = common.create_variable(name='x',
                                     shape=tf.TensorShape([self._dims]),
                                     initial_value=0,
                                     dtype=tf.float32)
    self._fx_t = common.create_variable(name='fx_t',
                                        shape=(),
                                        initial_value=0,
                                        dtype=tf.float32)
    self._fx_t1 = common.create_variable(name='fx_t1',
                                         shape=(),
                                         initial_value=0,
                                         dtype=tf.float32)
    self._known_min = common.create_variable(name='known_min',
                                             shape=(),
                                             initial_value=0,
                                             dtype=tf.float32)
    self._known_max = common.create_variable(name='known_max',
                                             shape=(),
                                             initial_value=0,
                                             dtype=tf.float32)

    def update_reset_min_max():
      rng_xs = self._rng.uniform(
        shape=tf.TensorShape(
          self._num_x_samples_reset).concatenate([self._dims]),
        minval=self._domain[0],
        maxval=self._domain[1],
        dtype=tf.float32)
      fxs = tf.map_fn(lambda x: self._fn_dist(x), rng_xs)
      self._known_max.assign(tf.reduce_max(fxs))
      self._known_min.assign(tf.reduce_min(fxs))

    self._update_reset_min_max = update_reset_min_max

    def update_min_max(fx: tf.Tensor):
      self._known_max.assign(
        tf.cond(pred=tf.math.greater(fx, self._known_max),
                true_fn=lambda: fx,
                false_fn=self._known_max.value))

      self._known_min.assign(
        tf.cond(pred=tf.math.less(fx, self._known_min),
                true_fn=lambda: fx,
                false_fn=self._known_min.value))

    self._update_min_max = update_min_max

    # Observable state
    self._grads_at_x = common.create_variable(
      name='grads_at_x',
      shape=tf.TensorShape([self._dims]),
      initial_value=0,
      dtype=tf.float32)

  def _current_time_step(self) -> ts.TimeStep:
    grads = self._grads_at_x.value()
    fx_t = self._fx_t.value()
    fx_t1 = self._fx_t1.value()
    known_min = self._known_min.value()
    known_max = self._known_max.value()

    with tf.control_dependencies([grads]):
      def log_processing(g: tf.Tensor):
        abs_g = tf.abs(g)
        p = tf.constant(10, dtype=tf.float32)
        return tf.cond(pred=tf.greater_equal(abs_g, tf.math.exp(-p)),
                       true_fn=lambda: tf.math.divide(tf.math.log(abs_g), p),
                       false_fn=lambda: tf.constant(-1.0, dtype=tf.float32))

      def sign_processing(g: tf.Tensor):
        p = tf.constant(10, dtype=tf.float32)
        return tf.cond(pred=tf.greater_equal(tf.abs(g), tf.math.exp(-p)),
                       true_fn=lambda: tf.sign(g),
                       false_fn=lambda: tf.multiply(g, tf.math.exp(-p)))

      log_grad = tf.map_fn(log_processing, grads)
      sign_grad = tf.map_fn(sign_processing, grads)

      with tf.control_dependencies([log_grad, sign_grad]):
        observation = tf.concat([log_grad, sign_grad], axis=-1,
                                name='log_sign_grad_concat')

    with tf.control_dependencies([fx_t, fx_t1, known_min, known_max]):
      clipped_delta = tf.clip_by_value(fx_t - fx_t1,
                                       clip_value_min=-1e2,
                                       clip_value_max=1e2)
      scaled_fx_t1 = min_max_scaling(fx_t1, known_min, known_max)
      distance_to_known_min = fx_t1 - known_min

      with tf.control_dependencies([distance_to_known_min]):
        bonus = tf.case(
          [(tf.math.less(
            distance_to_known_min, 0),
            lambda: tf.constant(10, dtype=tf.float32)),
            (tf.math.logical_and(tf.math.greater(distance_to_known_min, 0),
                                 tf.math.less(distance_to_known_min, 0.5)),
             lambda: 10*distance_to_known_min)],
          default=lambda: tf.constant(0, dtype=tf.float32),
          exclusive=True, strict=True)

    def first():
      return (tf.constant(FIRST, dtype=tf.int32),
              tf.constant(0.0, dtype=tf.float32))

    def mid():
      return (tf.constant(MID, dtype=tf.int32),
              tf.reshape(clipped_delta + scaled_fx_t1 + bonus,
                         shape=()))

    def last():
      return (tf.constant(LAST, dtype=tf.int32),
              tf.reshape(clipped_delta + scaled_fx_t1 + bonus,
                         shape=()))

    with tf.control_dependencies([clipped_delta, scaled_fx_t1, bonus]):
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
      x_reset = self._x.assign(value=self._rng.uniform(
        shape=tf.TensorShape([self._dims]),
        minval=current_domain[0],
        maxval=current_domain[1],
        dtype=tf.float32))

      with tf.control_dependencies([x_reset]):
        self._update_reset_min_max()
        grad, fx = self._fn_dist.grads_at(x_reset.value())

        with tf.control_dependencies([fx, grad]):
          fx_t_reset = self._fx_t.assign(value=fx)
          fx_t1_reset = self._fx_t1.assign(value=fx)
          grads_x_reset = self._grads_at_x.assign(value=grad)

          with tf.control_dependencies([fx_t_reset,
                                        fx_t1_reset,
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
        fx_t1_value = self._fx_t1.value()

        with tf.control_dependencies([fx, grads, fx_t1_value]):
          fx_t_update = self._fx_t.assign(fx_t1_value)
          fx_t1_update = self._fx_t1.assign(fx)
          grads_update = self._grads_at_x.assign(grads)
          self._update_min_max(fx)

      with tf.control_dependencies([x_update,
                                    fx_t_update,
                                    fx_t1_update,
                                    grads_update,
                                    steps_update,
                                    episode_finished]):
        return self.current_time_step()

    def reset_env():
      return self.reset()

    return tf.cond(pred=tf.math.reduce_any(self._episode_ended),
                   true_fn=reset_env,
                   false_fn=take_step)

  @property
  @autograph.do_not_convert()
  def functions(self):
    return self._functions

  @property
  @autograph.do_not_convert()
  def fn_index(self):
    return self._fn_index

  @autograph.do_not_convert()
  def get_info(self, to_numpy=False):
    raise NotImplementedError("No info available for this environment.")

  def render(self):
    raise ValueError('Environment does not support render yet.')
