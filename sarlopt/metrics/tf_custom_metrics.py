"""Custom TF Metrics."""

import tensorflow as tf

from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common

from optfuncs import core


class AverageBestObjectiveValueMetric(tf_metric.TFStepMetric):
  def __init__(self,
               function: core.Function,
               name='AverageBestValue',
               prefix='Metrics',
               dtype=tf.float32,
               buffer_size=10):
    super(AverageBestObjectiveValueMetric, self).__init__(name=name,
                                                          prefix=prefix)
    self._function = common.function(function)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._best_value = common.create_variable(initial_value=0, dtype=dtype,
                                              shape=(), name='BestValue')
    self._current_value = common.create_variable(initial_value=0, dtype=dtype,
                                                 shape=(), name='CurrentValue')

  @common.function(autograph=True)
  def call(self, trajectory):
    self._current_value.assign(self._function(trajectory.observation[0]))

    if trajectory.is_first():
      self._best_value.assign(self._current_value)
    else:
      if tf.reduce_all(tf.math.less(self._current_value,
                                    self._best_value)):
        self._best_value.assign(self._current_value)

      # Every Trajectory has information of 2 time_step's,
      # when a boundary trajectory is found, the best value found in the
      # episode is added to the buffer.
      if trajectory.is_boundary():
        self._buffer.add(self._best_value)

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._best_value.assign(0)


class ConvergenceMultiMetric(tf_metric.TFMultiMetricStepMetric):
  def __init__(self,
               trajectory_size,
               function: core.Function,
               name='ConvergenceMultiMetric',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    self._function = common.function(function)
    self._batch_size = batch_size
    self._buffer = tf_metrics.TFDeque(buffer_size,
                                      dtype,
                                      shape=(trajectory_size,))
    self._buffer_single_trajectory = tf_metrics.TFDeque(trajectory_size,
                                                        dtype,
                                                        shape=(1,))
    self._metric_names = ('AverageTrajectory', 'MinTrajectory', 'MaxTrajectory')
    self._dtype = dtype

    self._best_value = common.create_variable(initial_value=0, dtype=dtype,
                                              shape=(), name='BestValue')
    self._current_value = common.create_variable(initial_value=0, dtype=dtype,
                                                 shape=(), name='CurrentValue')

    super(ConvergenceMultiMetric, self).__init__(
      name=name,
      prefix=prefix,
      metric_names=self._metric_names)

  @common.function(autograph=True)
  def call(self, trajectory):
    self._current_value.assign(self._function(trajectory.observation[0]))

    if trajectory.is_first():
      self._best_value.assign(self._current_value)
    else:
      if tf.reduce_all(tf.math.less(self._current_value,
                                    self._best_value)):
        self._best_value.assign(self._current_value)

    self._buffer_single_trajectory.add(self._best_value)

    # Every Trajectory has information of 2 time_step's,
    # when a boundary trajectory is found, the "convergence" of the
    # episode is added to the buffer.
    if trajectory.is_boundary():
      self._buffer.add(tf.squeeze(self._buffer_single_trajectory.data))
      self._buffer_single_trajectory.clear()

    return trajectory

  def result(self):
    return [self._buffer.mean(),
            self._buffer.min(),
            self._buffer.max()]

  @common.function
  def reset(self):
    self._buffer.clear()
    self._buffer_single_trajectory.clear()
    self._best_value.assign(0)
    self._current_value.assign(0)
