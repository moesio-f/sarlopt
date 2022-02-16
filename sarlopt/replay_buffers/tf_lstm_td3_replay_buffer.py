import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import table


class TFLSTMTD3ReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
  """LSTM-TD3 replay buffer."""

  def __init__(self,
               data_spec,
               batch_size,
               episode_length=-1,
               max_length=1000,
               scope='LSTMTD3ReplayBuffer',
               device='cpu:*',
               table_fn=table.Table,
               dataset_drop_remainder=False,
               dataset_window_shift=None,
               stateful_dataset=False):
    assert episode_length > 0
    assert batch_size == 1
    self._episode_length = tf.convert_to_tensor(episode_length, dtype=tf.int64)
    super(TFLSTMTD3ReplayBuffer, self).__init__(
      data_spec=data_spec,
      batch_size=batch_size,
      max_length=max_length,
      scope=scope,
      device=device,
      table_fn=table_fn,
      dataset_drop_remainder=dataset_drop_remainder,
      dataset_window_shift=dataset_window_shift,
      stateful_dataset=stateful_dataset)

  def _num_episodes(self):
    return tf.cast(tf.math.floordiv(self._get_last_id() + 1,
                                    self._episode_length), tf.int64)

  def _num_partial_episodes(self):
    return tf.cond(tf.not_equal(tf.math.mod(self._get_last_id() + 1,
                                            self._episode_length), 0),
                   lambda: tf.constant(1, dtype=tf.int64),
                   lambda: tf.constant(0, dtype=tf.int64))

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('get_next'):
        min_val, max_val = _valid_range_ids(
          self._get_last_id(), self._max_length, num_steps)
        rows_shape = () if sample_batch_size is None else (sample_batch_size,)
        assert_nonempty = tf.compat.v1.assert_greater(
          max_val,
          min_val,
          message='Buffer is empty. Make sure to add items '
                  'before sampling the buffer.')
        assert_time_stacked = tf.debugging.Assert(tf.equal(time_stacked, True),
                                                  [time_stacked])

        with tf.control_dependencies([assert_nonempty, assert_time_stacked]):
          if num_steps is None:
            num_ids = max_val - min_val
            probability = tf.cond(
              pred=tf.equal(num_ids, 0),
              true_fn=lambda: 0.,
              false_fn=lambda: 1. / tf.cast(num_ids * self._batch_size,
                                            # pylint: disable=g-long-lambda
                                            tf.float32))
            ids = tf.random.uniform(
              rows_shape, minval=min_val, maxval=max_val, dtype=tf.int64)

            rows_to_get = tf.math.mod(ids, self._max_length)
            data = self._data_table.read(rows_to_get)
            data_ids = self._id_table.read(rows_to_get)
          else:
            # If num_steps is not None, we must resample the ids
            #   to make sure that it's possible to construct a history
            #   from the sampled ids.
            num_steps = tf.convert_to_tensor(num_steps, dtype=tf.int64)
            assert_valid_steps = tf.compat.v1.assert_less_equal(
              num_steps,
              self._episode_length,
              message='Number of steps can\'t be greater than episode length.')

            with tf.control_dependencies([assert_valid_steps]):
              # The valid ids are made of:
              #   1. All indices of complete episodes;
              #   2. All indices of partial episodes that have length
              #      equal or greater to `num_steps`.
              last_id = self._get_last_id()
              n_complete_episodes = self._num_episodes()
              id_after_complete_episodes = tf.multiply(n_complete_episodes,
                                                       self._episode_length)
              n_partial_episodes = self._num_partial_episodes()
              size_partial_ep = last_id - id_after_complete_episodes + 1

              assert_exists_ids = tf.debugging.Assert(
                tf.math.logical_or(
                  tf.math.logical_and(
                    tf.greater_equal(self._episode_length, num_steps),
                    tf.greater(n_complete_episodes, 0)),
                  tf.greater_equal(size_partial_ep, num_steps)),
                [n_complete_episodes, n_partial_episodes, size_partial_ep])

              with tf.control_dependencies([assert_exists_ids]):
                max_val = tf.cond(tf.greater_equal(size_partial_ep, num_steps),
                                  lambda: max_val,
                                  lambda: id_after_complete_episodes)

                # Either [1] or [B]
                ids = tf.random.uniform(rows_shape,
                                        minval=min_val,
                                        maxval=max_val,
                                        dtype=tf.int64)
                num_ids = min_val - max_val
                probability = tf.cond(
                  pred=tf.equal(num_ids, 0),
                  true_fn=lambda: 0.,
                  false_fn=lambda: 1. / tf.cast(num_ids * self._batch_size,
                                                tf.float32))

                def history_offset(id_):
                  k = tf.math.floordiv(id_, self._episode_length)

                  limits = tf.cond(tf.greater_equal(
                    id_,
                    id_after_complete_episodes),
                    true_fn=lambda: tf.stack([id_after_complete_episodes,
                                              last_id + 1],
                                             axis=0),
                    false_fn=lambda: tf.stack(
                      [tf.multiply(k, self._episode_length),
                       tf.multiply(k + 1, self._episode_length)],
                      axis=0))
                  id_1 = id_ + 1

                  offset_0_start = tf.math.maximum(limits[0],
                                                   id_1 - num_steps)
                  offset_0 = tf.math.subtract(
                    tf.range(start=offset_0_start,
                             limit=id_1,
                             dtype=tf.int64),
                    id_)

                  remaining_steps = num_steps - (id_1 - offset_0_start)
                  offset_1_limit = tf.math.minimum(limits[1],
                                                   id_1 + remaining_steps)

                  offset_1 = tf.cond(
                    tf.greater(remaining_steps, 0),
                    true_fn=lambda: tf.math.subtract(
                      tf.range(start=id_1,
                               limit=offset_1_limit,
                               dtype=tf.int64),
                      id_),
                    false_fn=lambda: tf.constant([], dtype=tf.int64))

                  return tf.concat([offset_0, offset_1], axis=0)

                # Either [num_steps] or [B, num_steps]
                step_range = tf.map_fn(history_offset, ids)

                if sample_batch_size:
                  # [B, num_steps]
                  ids = tf.tile(tf.expand_dims(ids, -1), [1, num_steps])

                rows_to_get = tf.math.mod(step_range + ids, self._max_length)
                data = self._data_table.read(rows_to_get)
                data_ids = self._id_table.read(rows_to_get)

        probabilities = tf.fill(rows_shape, probability)
        buffer_info = tf_uniform_replay_buffer.BufferInfo(
          ids=data_ids,
          probabilities=probabilities)
    return data, buffer_info


def _valid_range_ids(last_id, max_length, num_steps=None):
  if num_steps is None:
    num_steps = tf.constant(1, tf.int64)

  min_id_not_full = tf.constant(0, dtype=tf.int64)
  max_id_not_full = tf.maximum(last_id + 1 - num_steps + 1, 0)

  min_id_full = last_id + 1 - max_length
  max_id_full = last_id + 1 - num_steps + 1

  return (tf.where(last_id < max_length, min_id_not_full, min_id_full),
          tf.where(last_id < max_length, max_id_not_full, max_id_full))
