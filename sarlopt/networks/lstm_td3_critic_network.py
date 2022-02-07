import typing
import functools

import tensorflow as tf

from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.typing import types


class LSTMTD3InputCritic(typing.NamedTuple):
  history: types.SpecTensorOrArray
  observations: types.SpecTensorOrArray
  actions: types.SpecTensorOrArray


class LSTMTD3CriticNetwork(network.Network):
  """Creates a LSTM-TD3 critic network."""

  def __init__(self,
               input_tensor_spec,
               memory_fc_before_lstm=(128,),
               memory_lstm_hidden=(128,),
               memory_fc_after_lstm=(),
               fc_current_feature=(128,),
               fc_after_concat=(128,),
               name='LSTMTD3CriticNetwork'):
    if not isinstance(input_tensor_spec, LSTMTD3InputCritic):
      raise ValueError("Input spec not supported.")

    # Memory
    # Before LSTM
    memory_input_layers = utils.mlp_layers(
      fc_layer_params=memory_fc_before_lstm,
      activation_fn=tf.keras.activations.relu,
      kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
      name='memory_input')

    # LSTM
    if len(memory_lstm_hidden) == 1:
      cell = tf.keras.layers.LSTMCell(memory_lstm_hidden[0])
    else:
      cell = tf.keras.layers.StackedRNNCells(
        [tf.keras.layers.LSTMCell(size) for size in memory_lstm_hidden])

    memory_state_spec = tf.nest.map_structure(
      functools.partial(
        tensor_spec.TensorSpec, dtype=tf.float32,
        name='network_memory_state_spec'), list(cell.state_size))

    # After LSTM
    memory_output_layers = utils.mlp_layers(
      fc_layer_params=memory_fc_after_lstm,
      activation_fn=tf.keras.activations.relu,
      name='memory_output')

    # Current feature extraction
    feature_input_layers = utils.mlp_layers(
      fc_layer_params=fc_current_feature,
      activation_fn=tf.keras.activations.relu,
      kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
      name='feature_input')

    # After concat
    after_concat_layers = utils.mlp_layers(
      fc_layer_params=fc_after_concat,
      activation_fn=tf.keras.activations.relu,
      kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
      name='concat_input')

    # Q-value ouput
    output_q_layer = tf.keras.layers.Dense(
      1,
      activation=tf.keras.activations.linear,
      kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.003, maxval=0.003),
      name='action')

    super(LSTMTD3CriticNetwork, self).__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=memory_state_spec,
      name=name)

    self._memory_fc_before_lstm = memory_fc_before_lstm
    self._memory_lstm_hidden = memory_lstm_hidden
    self._memory_fc_after_lstm = memory_fc_after_lstm
    self._fc_current_feature = fc_current_feature
    self._fc_after_concat = fc_after_concat

    self._feature_input_layers = feature_input_layers
    self._memory_input_layers = memory_input_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._memory_output_layers = memory_output_layers
    self._after_concat_layers = after_concat_layers
    self._output_q_layer = output_q_layer

  def call(self, inputs, step_type, network_state=(), training=False):
    del step_type  # unused.
    actions = tf.cast(inputs.actions, tf.float32)
    observations = tf.cast(inputs.observations, tf.float32)
    history = tf.cast(inputs.history, tf.float32)

    rank = tf.rank(history)
    tf.debugging.assert_equal(rank, 3, message='Expected tensor with rank 3.')

    # Memory
    extracted_memory = history  # has shape: [B, T, ...]

    for layer in self._memory_input_layers:
      extracted_memory = layer(extracted_memory, training=training)

    # noinspection PyCallingNonCallable
    extracted_memory, network_state = self._dynamic_unroll(
      extracted_memory,
      initial_state=network_state,
      training=training)

    for layer in self._memory_output_layers:
      extracted_memory = layer(extracted_memory, training=training)

    # Current feature extractions
    x = tf.concat([observations, actions], axis=-1)
    for layer in self._feature_input_layers:
      x = layer(x, training=training)

    # Concat
    x = tf.concat([extracted_memory, x], axis=-1)
    for layer in self._after_concat_layers:
      x = layer(x, training=training)

    # Q-value output
    x = self._output_q_layer(x, training=training)
    q_value = tf.squeeze(x, axis=-1)  # Ensure that q_value.shape == [B]
    return q_value, network_state
