import typing
from typing import Optional, Text

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.specs import tensor_spec

from sarlopt.networks.lstm_td3_actor_network import (LSTMTD3ActorNetwork,
                                                     LSTMTD3InputActor)
from sarlopt.networks.lstm_td3_critic_network import (LSTMTD3CriticNetwork,
                                                      LSTMTD3InputCritic)
from sarlopt.policies.lstm_td3_policies import (LSTMTD3ActorPolicy,
                                                LSTMTD3GaussianPolicy)


class LSTMTD3Info(typing.NamedTuple):
  actor_loss: types.Tensor
  critic_loss: types.Tensor


class LSTMTD3Agent(tf_agent.TFAgent):
  """A LSTM-TD3 Agent."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensor,
               observation_spec: types.NestedTensor,
               actor_optimizer: types.Optimizer,
               critic_optimizer: types.Optimizer,
               history_length: int,
               actor_memory_fc_before_lstm: typing.List[int] = (128,),
               actor_memory_lstm_hidden: typing.List[int] = (128,),
               actor_memory_fc_after_lstm: typing.List[int] = (),
               actor_fc_current_feature: typing.List[int] = (128,),
               actor_fc_after_concat: typing.List[int] = (128,),
               critic_memory_fc_before_lstm: typing.List[int] = (128,),
               critic_memory_lstm_hidden: typing.List[int] = (128,),
               critic_memory_fc_after_lstm: typing.List[int] = (),
               critic_fc_current_feature: typing.List[int] = (128,),
               critic_fc_after_concat: typing.List[int] = (128,),
               exploration_noise_std: types.Float = 0.1,
               target_update_tau: types.Float = 1.0,
               target_update_period: types.Int = 1,
               actor_update_period: types.Int = 1,
               td_errors_loss_fn: Optional[types.LossFn] = None,
               gamma: types.Float = 1.0,
               reward_scale_factor: types.Float = 1.0,
               target_policy_noise: types.Float = 0.2,
               target_policy_noise_clip: types.Float = 0.5,
               gradient_clipping: Optional[types.Float] = None,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               train_step_counter: Optional[tf.Variable] = None,
               name: Optional[Text] = None):
    tf.Module.__init__(self, name=name)
    flat_action_spec = tf.nest.flatten(action_spec)
    flat_observation_spec = tf.nest.flatten(observation_spec)
    self._hist_len = tf.convert_to_tensor(history_length, dtype=tf.int32)

    if len(flat_action_spec) > 1:
      raise ValueError(
        'Only a single observation is supported by this network')

    if len(flat_observation_spec) > 1:
      raise ValueError(
        'Only a single observation is supported by this network.')

    n_elements_actions = flat_action_spec[0].shape.num_elements()
    n_elements_observations = flat_observation_spec[0].shape.num_elements()
    n_elements_concat = n_elements_actions + n_elements_observations

    actor_input_spec = LSTMTD3InputActor(
      history=tensor_spec.TensorSpec(
        [history_length, n_elements_concat], dtype=tf.float32),
      observations=tensor_spec.TensorSpec(
        [n_elements_observations], dtype=tf.float32))

    critic_input_spec = LSTMTD3InputCritic(
      history=tensor_spec.TensorSpec(
        [history_length, n_elements_concat], dtype=tf.float32),
      observations=tensor_spec.TensorSpec(
        [n_elements_observations], dtype=tf.float32),
      actions=tensor_spec.TensorSpec(
        [n_elements_actions], dtype=tf.float32))

    self._actor_network = LSTMTD3ActorNetwork(
      actor_input_spec,
      flat_action_spec[0],
      memory_fc_before_lstm=actor_memory_fc_before_lstm,
      memory_lstm_hidden=actor_memory_lstm_hidden,
      memory_fc_after_lstm=actor_memory_fc_after_lstm,
      fc_current_feature=actor_fc_current_feature,
      fc_after_concat=actor_fc_after_concat)
    self._actor_network.create_variables()
    self._target_actor_network = common.maybe_copy_target_network_with_checks(
      self._actor_network, None, 'TargetActorNetwork')

    self._critic_network_1 = LSTMTD3CriticNetwork(
      critic_input_spec,
      memory_fc_before_lstm=critic_memory_fc_before_lstm,
      memory_lstm_hidden=critic_memory_lstm_hidden,
      memory_fc_after_lstm=critic_memory_fc_after_lstm,
      fc_current_feature=critic_fc_current_feature,
      fc_after_concat=critic_fc_after_concat)
    self._critic_network_1.create_variables()
    self._target_critic_network_1 = (
      common.maybe_copy_target_network_with_checks(self._critic_network_1,
                                                   None,
                                                   'TargetCriticNetwork1'))

    self._critic_network_2 = self._critic_network_1.copy(name='CriticNetwork2')
    self._critic_network_2.create_variables()
    self._target_critic_network_2 = (
      common.maybe_copy_target_network_with_checks(self._critic_network_2,
                                                   None,
                                                   'TargetCriticNetwork2'))

    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer

    self._exploration_noise_std = exploration_noise_std
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_update_period = actor_update_period
    self._td_errors_loss_fn = (
          td_errors_loss_fn or common.element_wise_huber_loss)
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_policy_noise = target_policy_noise
    self._target_policy_noise_clip = target_policy_noise_clip
    self._gradient_clipping = gradient_clipping

    self._update_target = self._get_target_updater(
      target_update_tau, target_update_period)

    # Create Policy with memory (keep history)
    policy = LSTMTD3ActorPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=self._actor_network,
      clip=True,
      history_length=self._hist_len)
    collect_policy = LSTMTD3GaussianPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=self._actor_network,
      clip=False,
      scale=self._exploration_noise_std,
      history_length=self._hist_len)

    train_sequence_length = None
    super(LSTMTD3Agent, self).__init__(
      time_step_spec,
      action_spec,
      policy,
      collect_policy,
      train_sequence_length=train_sequence_length,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step_counter,
    )

  def _initialize(self):
    common.soft_variables_update(
      self._critic_network_1.variables,
      self._target_critic_network_1.variables,
      tau=1.0)
    common.soft_variables_update(
      self._critic_network_2.variables,
      self._target_critic_network_2.variables,
      tau=1.0)
    common.soft_variables_update(
      self._actor_network.variables,
      self._target_actor_network.variables,
      tau=1.0)

  def _get_target_updater(self, tau=1.0, period=1):
    with tf.name_scope('update_targets'):
      def update():  # pylint: disable=missing-docstring
        critic_update_1 = common.soft_variables_update(
          self._critic_network_1.variables,
          self._target_critic_network_1.variables,
          tau,
          tau_non_trainable=1.0)

        critic_2_update_vars = common.deduped_network_variables(
          self._critic_network_2, self._critic_network_1)
        target_critic_2_update_vars = common.deduped_network_variables(
          self._target_critic_network_2, self._target_critic_network_1)

        critic_update_2 = common.soft_variables_update(
          critic_2_update_vars,
          target_critic_2_update_vars,
          tau,
          tau_non_trainable=1.0)

        actor_update_vars = common.deduped_network_variables(
          self._actor_network, self._critic_network_1, self._critic_network_2)
        target_actor_update_vars = common.deduped_network_variables(
          self._target_actor_network, self._target_critic_network_1,
          self._target_critic_network_2)

        actor_update = common.soft_variables_update(
          actor_update_vars,
          target_actor_update_vars,
          tau,
          tau_non_trainable=1.0)
        return tf.group(critic_update_1, critic_update_2, actor_update)

      return common.Periodically(update, period, 'update_targets')

  def _train(self, experience, weights=None):
    # Experiencie is a Trajectory with following specs
    #   - step_type:        [B, H + 2]
    #   - observation:      [B, H + 2, ...]
    #   - action:           [B, H + 2, ...]
    #   - reward:           [B, H + 2]
    #   - discount:         [B, H + 2]
    #   - next_step_type:   [B, H + 2]
    #   - policy_info:      ()
    # We need to extract:
    #   Reminder:. indices (axis=1) goes from 0 to H + 1 (Length = H + 2).
    #   1. o_t:   [B, ...] (gather from index H)
    #   2. o_t1:  [B, ...] (gather from index H + 1)
    #   3. a_t:   [B, ...] (gather from index H)
    #   4. hist_o_t:  [B, H, ....] (gather from index 0 to H - 1)
    #   5. hist_a_t:  [B, H, ....] (gather from index 0 to H - 1)
    #   6. hist_o_t1: [B, H, ....] (gather from index 1 to H)
    #   7. r_t1: [B, ...] (gather from index H, Trajectory keeps reward after
    #     executing action)
    #   8. d_t: [B, ...] (gather from index H)

    observations = experience.observation
    actions = experience.action
    rewards = experience.reward
    discounts = experience.discount

    o_t = tf.gather(observations, self._hist_len, axis=1)
    a_t = tf.gather(actions, self._hist_len, axis=1)
    r_t1 = tf.gather(rewards, self._hist_len, axis=1)
    d_t1 = tf.gather(discounts, self._hist_len + 1, axis=1)
    o_t1 = tf.gather(observations, self._hist_len + 1, axis=1)

    hist_indices_t = tf.range(start=0, limit=self._hist_len)
    hist_indices_t1 = hist_indices_t + 1

    hist_o_t = tf.gather(observations, hist_indices_t, axis=1)
    hist_a_t = tf.gather(actions, hist_indices_t, axis=1)
    hist_o_t1 = tf.gather(observations, hist_indices_t1, axis=1)
    hist_a_t1 = tf.gather(actions, hist_indices_t1, axis=1)

    trainable_critic_variables = list(object_identity.ObjectIdentitySet(
      self._critic_network_1.trainable_variables +
      self._critic_network_2.trainable_variables))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self.critic_loss(o_t, o_t1,
                                     a_t,
                                     d_t1, r_t1,
                                     hist_o_t, hist_a_t,
                                     hist_o_t1, hist_a_t1,
                                     weights=weights,
                                     training=True)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self.actor_loss(o_t,
                                   hist_o_t, hist_a_t,
                                   weights=weights,
                                   training=True)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

    # We only optimize the actor every actor_update_period training steps.
    def optimize_actor():
      actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
      return self._apply_gradients(actor_grads, trainable_actor_variables,
                                   self._actor_optimizer)

    remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)
    tf.cond(
      pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = actor_loss + critic_loss

    return tf_agent.LossInfo(total_loss,
                             LSTMTD3Info(actor_loss, critic_loss))

  def _apply_gradients(self, gradients, variables, optimizer):
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(
        grads_and_vars, self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    return optimizer.apply_gradients(grads_and_vars)

  def critic_loss(self,
                  o_t: types.Tensor,
                  o_t1: types.Tensor,
                  a_t: types.Tensor,
                  d_t1: types.Tensor,
                  r_t1: types.Tensor,
                  hist_o_t: types.Tensor,
                  hist_a_t: types.Tensor,
                  hist_o_t1: types.Tensor,
                  hist_a_t1: types.Tensor,
                  weights: Optional[types.Tensor] = None,
                  training: bool = False) -> types.Tensor:
    with tf.name_scope('critic_loss'):
      hist_t = tf.concat([hist_o_t, hist_a_t], axis=-1)
      hist_t1 = tf.concat([hist_o_t1, hist_a_t1], axis=-1)

      target_actions, _ = self._target_actor_network(
        LSTMTD3InputActor(history=hist_t1,
                          observations=o_t1),
        None, training=training)

      # Add gaussian noise to each action before computing target q values
      def add_noise_to_action(action):  # pylint: disable=missing-docstring
        # noinspection PyUnresolvedReferences
        dist = tfp.distributions.Normal(
          loc=tf.zeros_like(action),
          scale=self._target_policy_noise * tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                 self._target_policy_noise_clip)
        return action + noise

      noisy_target_actions = tf.nest.map_structure(add_noise_to_action,
                                                   target_actions)

      # Target q-values are the min of the two networks
      target_q_values_1, _ = self._target_critic_network_1(
        LSTMTD3InputCritic(history=hist_t1,
                           actions=noisy_target_actions,
                           observations=o_t1),
        None,
        training=False)
      target_q_values_2, _ = self._target_critic_network_2(
        LSTMTD3InputCritic(history=hist_t1,
                           actions=noisy_target_actions,
                           observations=o_t1),
        None,
        training=False)
      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      td_targets = tf.stop_gradient(
        self._reward_scale_factor * r_t1 +
        self._gamma * d_t1 * target_q_values)

      pred_td_targets_1, _ = self._critic_network_1(
        LSTMTD3InputCritic(history=hist_t,
                           observations=o_t,
                           actions=a_t),
        None,
        training=training)

      pred_td_targets_2, _ = self._critic_network_2(
        LSTMTD3InputCritic(history=hist_t,
                           observations=o_t,
                           actions=a_t),
        None,
        training=training)

      pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
          name='td_targets', data=td_targets, step=self.train_step_counter)
        with tf.name_scope('td_targets'):
          tf.compat.v2.summary.scalar(
            name='mean',
            data=tf.reduce_mean(input_tensor=td_targets),
            step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
            name='max',
            data=tf.reduce_max(input_tensor=td_targets),
            step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
            name='min',
            data=tf.reduce_min(input_tensor=td_targets),
            step=self.train_step_counter)

        for td_target_idx in range(2):
          pred_td_targets = pred_td_targets_all[td_target_idx]
          td_errors = td_targets - pred_td_targets
          with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
            tf.compat.v2.summary.histogram(
              name='td_errors', data=td_errors, step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
              name='pred_td_targets',
              data=pred_td_targets,
              step=self.train_step_counter)
            with tf.name_scope('td_errors'):
              tf.compat.v2.summary.scalar(
                name='mean',
                data=tf.reduce_mean(input_tensor=td_errors),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='mean_abs',
                data=tf.reduce_mean(input_tensor=tf.abs(td_errors)),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='max',
                data=tf.reduce_max(input_tensor=td_errors),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='min',
                data=tf.reduce_min(input_tensor=td_errors),
                step=self.train_step_counter)
            with tf.name_scope('pred_td_targets'):
              tf.compat.v2.summary.scalar(
                name='mean',
                data=tf.reduce_mean(input_tensor=pred_td_targets),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='max',
                data=tf.reduce_max(input_tensor=pred_td_targets),
                step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                name='min',
                data=tf.reduce_min(input_tensor=pred_td_targets),
                step=self.train_step_counter)

      critic_loss = (self._td_errors_loss_fn(td_targets, pred_td_targets_1)
                     + self._td_errors_loss_fn(td_targets, pred_td_targets_2))

      if weights is not None:
        critic_loss *= weights

      return tf.reduce_mean(input_tensor=critic_loss)

  def actor_loss(self,
                 o_t: types.Tensor,
                 hist_o_t: types.Tensor,
                 hist_a_t: types.Tensor,
                 weights: Optional[types.Tensor] = None,
                 training: bool = False) -> types.Tensor:
    with tf.name_scope('actor_loss'):
      hist_t = tf.concat([hist_o_t, hist_a_t], axis=-1)
      actions, _ = self._actor_network(
        LSTMTD3InputActor(history=hist_t,
                          observations=o_t),
        None,
        training=training)

      q_values, _ = self._critic_network_1(
        LSTMTD3InputCritic(history=hist_t,
                           actions=actions,
                           observations=o_t),
        None,
        training=False)
      actor_loss = -q_values
      # Sum over the time dimension.
      if actor_loss.shape.rank > 1:
        actor_loss = tf.reduce_sum(
          actor_loss, axis=range(1, actor_loss.shape.rank))
      actor_loss = common.aggregate_losses(
        per_example_loss=actor_loss, sample_weight=weights).total_loss

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)

    return actor_loss

  def history_length(self) -> types.Tensor:
    return self._hist_len
