"""LSTM-TD3 Actor Policy. """

from typing import Optional, Text

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.replay_buffers import table

from sarlopt.networks.lstm_td3_actor_network import (LSTMTD3ActorNetwork,
                                                     LSTMTD3InputActor)


class HistoryBuffer(object):
  def __init__(self,
               history_length,
               observation_spec,
               action_spec,
               name='HistoryBuffer'):
    self._hist_len = tf.convert_to_tensor(history_length, dtype=tf.int32)
    self._obs_spec = tensor_spec.from_spec(observation_spec)
    self._act_spec = tensor_spec.from_spec(action_spec)

    self._obs_buffer = table.Table(self._obs_spec, capacity=history_length)
    self._act_buffer = table.Table(self._act_spec, capacity=history_length)

    self._write_head = common.create_variable(initial_value=0,
                                              dtype=tf.int32,
                                              shape=(),
                                              name=name + 'WriteHead')

    self._written_size = common.create_variable(initial_value=0,
                                                dtype=tf.int32,
                                                shape=(),
                                                name=name + 'WrittenSize')

    self._0 = tf.constant(0, tf.int32)
    self._1 = tf.constant(1, tf.int32)

    self._lambda_read_full = lambda: tf.concat(
      [tf.range(self._write_head, self._written_size,
                dtype=tf.int32),
       tf.range(self._0, self._write_head,
                dtype=tf.int32)],
      axis=-1)
    self._lambda_read_otherwise = lambda: tf.range(self._0,
                                                   self._written_size,
                                                   dtype=tf.int32)
    self._hist_indices_fn = lambda: tf.cond(
      tf.math.less(self._written_size, self._hist_len),
      self._lambda_read_otherwise,
      self._lambda_read_full)

  def add(self, observation, action):
    self._obs_buffer.write(self._write_head, observation)
    self._act_buffer.write(self._write_head, action)

    self._written_size.assign_add(
      tf.cond(tf.less(self._written_size, self._hist_len),
              lambda: self._1, lambda: self._0))
    self._write_head.assign(tf.math.floormod(self._write_head + 1,
                                             self._hist_len))

  def read_history(self):
    indices = self._hist_indices_fn()
    observations = self._obs_buffer.read(indices)  # [L, obs_shape]
    actions = self._act_buffer.read(indices)  # [L, act_shape]

    # Padding if needed.
    paddings = tf.cast(tf.reshape(tf.concat(
      [tf.stack([self._hist_len - self._written_size, self._0], axis=0),
       tf.stack([self._0, self._0], axis=0)], axis=0), [2, 2]), dtype=tf.int32)
    observations = tf.pad(observations, paddings)
    actions = tf.pad(actions, paddings)
    # Result shapes:  [T, obs_shape] and [T, act_shape]

    # Adding batch dimension (expected by networks).
    observations = tf.expand_dims(observations, axis=0)  # [1, T, obs_shape]
    actions = tf.expand_dims(actions, axis=0)  # [1, T, act_shape]

    return tf.concat([observations, actions], axis=-1)

  def clear(self):
    self._written_size.assign(0)
    self._write_head.assign(0)


class LSTMTD3ActorPolicy(tf_policy.TFPolicy):
  """LSTM-TD3 Actor Policy."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               actor_network: LSTMTD3ActorNetwork,
               history_length: int,
               info_spec: types.NestedTensorSpec = (),
               clip: bool = True,
               training: bool = False,
               name: Optional[Text] = None):
    time_step_spec = tensor_spec.from_spec(time_step_spec)
    action_spec = tensor_spec.from_spec(action_spec)

    if not isinstance(actor_network, LSTMTD3ActorNetwork):
      raise ValueError('Only LSTMTD3ActorNetwork is currently supported.')

    # Create variables regardless of if we use the output spec.
    actor_output_spec = actor_network.create_variables()

    nest_utils.assert_same_structure(actor_output_spec, action_spec)

    self._actor_network = actor_network
    self._training = training

    policy_state_spec = actor_network.state_spec
    self._memory = HistoryBuffer(history_length=history_length,
                                 observation_spec=time_step_spec.observation,
                                 action_spec=action_spec)

    super(LSTMTD3ActorPolicy, self).__init__(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      policy_state_spec=policy_state_spec,
      info_spec=info_spec,
      clip=clip,
      observation_and_action_constraint_splitter=None,
      automatic_state_reset=True,
      emit_log_probability=False,
      validate_args=True,
      name=name)

  @property
  def observation_normalizer(self):
    return None

  def _variables(self):
    return self._actor_network.variables

  def _distribution(self, time_step, policy_state):
    step_type = time_step.step_type
    observation = time_step.observation

    if time_step.is_first():
      self._memory.clear()

    history = self._memory.read_history()

    network_inputs = LSTMTD3InputActor(history=history,
                                       observations=observation)

    actions, policy_state = self._actor_network(
      network_inputs, step_type=step_type,
      network_state=policy_state)

    self._memory.add(
      tf.squeeze(observation, axis=[0]),
      tf.squeeze(actions, axis=[0]))

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions)
    return policy_step.PolicyStep(distributions, policy_state)


class LSTMTD3GaussianPolicy(tf_policy.TFPolicy):
  """LSTM-TD3 Gaussian Policy for exploration."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               actor_network: LSTMTD3ActorNetwork,
               history_length: int,
               scale: types.Float,
               info_spec: types.NestedTensorSpec = (),
               clip: bool = True,
               training: bool = False,
               name: Optional[Text] = None):
    time_step_spec = tensor_spec.from_spec(time_step_spec)
    action_spec = tensor_spec.from_spec(action_spec)

    if not isinstance(actor_network, LSTMTD3ActorNetwork):
      raise ValueError('Only LSTMTD3ActorNetwork is currently supported.')

    # Create variables regardless of if we use the output spec.
    actor_output_spec = actor_network.create_variables()

    nest_utils.assert_same_structure(actor_output_spec, action_spec)

    self._actor_network = actor_network
    self._training = training
    self._scale = scale

    policy_state_spec = actor_network.state_spec
    self._memory = HistoryBuffer(history_length=history_length,
                                 observation_spec=time_step_spec.observation,
                                 action_spec=action_spec)

    # noinspection PyUnresolvedReferences
    self._noise_distribution = tfp.distributions.Normal(
      loc=tf.zeros(action_spec.shape, dtype=action_spec.dtype),
      scale=tf.ones(action_spec.shape, dtype=action_spec.dtype) * scale)

    super(LSTMTD3GaussianPolicy, self).__init__(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      policy_state_spec=policy_state_spec,
      info_spec=info_spec,
      clip=clip,
      observation_and_action_constraint_splitter=None,
      automatic_state_reset=True,
      emit_log_probability=False,
      validate_args=True,
      name=name)

  @property
  def observation_normalizer(self):
    return None

  def _variables(self):
    return self._actor_network.variables

  @tf.function(autograph=True)
  def _action(self, time_step, policy_state, seed=None):
    step_type = time_step.step_type
    observation = time_step.observation

    if time_step.is_first():
      self._memory.clear()

    history = self._memory.read_history()

    network_inputs = LSTMTD3InputActor(history=history,
                                       observations=observation)

    actions, policy_state = self._actor_network(
      network_inputs, step_type=step_type,
      network_state=policy_state)

    # noinspection PyUnresolvedReferences
    seed_stream = tfp.util.SeedStream(seed=seed, salt='gaussian_noise')

    def _add_noise(action, distribution):
      noisy_action = action + distribution.sample(seed=seed_stream())

      if self._clip and isinstance(self.action_spec,
                                   tensor_spec.BoundedTensorSpec):
        return common.clip_to_spec(action, self.action_spec)

      return noisy_action

    actions = tf.nest.map_structure(_add_noise, actions,
                                    self._noise_distribution)

    self._memory.add(
      tf.squeeze(observation, axis=[0]),
      tf.squeeze(actions, axis=[0]))

    return policy_step.PolicyStep(actions, policy_state)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('Distributions are not implemented yet.')
