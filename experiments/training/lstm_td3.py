"""LSTM-TD3 for L2O."""

import time
import typing

import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from optfuncs import tensorflow_functions as tff

from sarlopt.environments import tf_function_env_v3 as tf_fun_env
from sarlopt.agents import lstm_td3_agent
from sarlopt.networks.lstm_td3_actor_network import LSTMTD3InputActor
from sarlopt.utils.functions import distributions as fn_distributions
from sarlopt.replay_buffers import tf_lstm_td3_replay_buffer

from experiments.training import utils as training_utils


class LayersLSTMTD3(typing.NamedTuple):
  memory_fc_before_lstm: typing.Optional[typing.List[int]]
  memory_lstm_hidden: typing.Optional[typing.List[int]]
  memory_fc_after_lstm: typing.Optional[typing.List[int]]
  fc_current_feature: typing.Optional[typing.List[int]]
  fc_after_concat: typing.Optional[typing.List[int]]


class DistributionBounds(typing.NamedTuple):
  vshift_bounds: fn_distributions.ParamBound
  hshift_bounds: fn_distributions.ParamBound
  scale_bounds: fn_distributions.ParamBound
  dims_params: int


def evaluate_agent(agent: lstm_td3_agent.LSTMTD3Agent,
                   function: tff.TensorflowFunction,
                   dims: int,
                   steps: int,
                   episodes: int,
                   seed=10000):
  history_length = agent.history_length()
  # noinspection PyProtectedMember
  network = common.maybe_copy_target_network_with_checks(agent._actor_network,
                                                         name='ActorPolicy')

  rng = tf.random.Generator.from_seed(seed, tf.random.Algorithm.PHILOX)
  history_obs = []
  history_act = []
  best_values = []

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

  for _ in range(history_length):
    history_obs.append(tf.zeros(shape=agent.time_step_spec.observation.shape,
                                dtype=tf.float32))
    history_act.append(tf.zeros(shape=agent.action_spec.shape,
                                dtype=tf.float32))

  print('-------- Evaluation --------')
  print('Function: {0}'.format(function.name))
  print('Domain: {0}'.format(function.domain))
  start_eval = time.time()
  for ep in range(episodes):
    domain = function.domain
    x = rng.uniform(shape=(dims,),
                    minval=domain[0],
                    maxval=domain[1],
                    dtype=tf.float32)
    best_fx = function(x)
    for step in range(steps):
      grads, fx = function.grads_at(x)

      if fx < best_fx:
        best_fx = fx

      log_grad = tf.map_fn(log_processing, grads)
      sign_grad = tf.map_fn(sign_processing, grads)

      observation = tf.concat([log_grad, sign_grad], axis=-1)
      h_obs = tf.convert_to_tensor(history_obs, dtype=tf.float32)
      h_act = tf.convert_to_tensor(history_act, dtype=tf.float32)
      history = tf.expand_dims(tf.concat([h_obs, h_act], axis=-1), axis=0)

      action, _ = network(
        LSTMTD3InputActor(history=history,
                          observations=tf.expand_dims(observation,
                                                      axis=0)),
        None, training=False)
      action = tf.squeeze(action)

      x = tf.clip_by_value(x + action,
                           clip_value_min=domain[0],
                           clip_value_max=domain[1])

      history_obs[step % history_length] = observation
      history_act[step % history_length] = action
    best_values.append(best_fx)

  avg_best = tf.reduce_mean(tf.convert_to_tensor(best_values, dtype=tf.float32))
  print('Average best value: {0}'.format(avg_best))
  print('Eval delta time: {0:.2f}'.format(time.time() - start_eval))
  print('---------------------------')

  return avg_best.numpy()


def train_lstm_td3(functions: typing.List[fn_distributions.FunctionList],
                   dist_bounds: DistributionBounds,
                   dims: int,
                   actions_bounds: typing.Tuple[float, float],
                   seed=1000,
                   training_episodes: int = 2000,
                   stop_threshold: float = None,
                   env_steps: int = 50,
                   eval_steps: int = 500,
                   eval_interval: int = 100,
                   eval_episodes: int = 10,
                   initial_collect_episodes: int = 20,
                   collect_steps_per_iteration: int = 1,
                   history_length: int = 3,  # Change to curriculum strategy.
                   buffer_size: int = 1000000,
                   batch_size: int = 64,
                   actor_lr: float = 3e-4,
                   critic_lr: float = 1e-3,
                   tau: float = 1e-3,
                   actor_update_period: int = 2,
                   target_update_period: int = 2,
                   discount: float = 0.99,
                   exploration_noise_std: float = 0.15, # + analytic optimizers.
                   target_policy_noise: float = 0.2,
                   target_policy_noise_clip: float = 0.5,
                   actor_layers: LayersLSTMTD3 = None,
                   critic_layers: LayersLSTMTD3 = None,
                   summary_flush_secs: int = 10,
                   debug_summaries: bool = False,
                   summarize_grads_and_vars: bool = False):
  algorithm_name = 'LSTM-TD3'

  # Creating the training/agent directory
  agent_dir = training_utils.create_agent_dir_str(algorithm_name,
                                                  'distribution',
                                                  dims)

  # Creating distribution
  fn_dist = fn_distributions.UniformFunctionDistribution(
    functions=functions,
    rng_seed=seed,
    vshift_bounds=dist_bounds.vshift_bounds,
    hshift_bounds=dist_bounds.hshift_bounds,
    scale_bounds=dist_bounds.scale_bounds,
    dims_params=dist_bounds.dims_params)

  # Creating the environment.
  tf_env_training = tf_fun_env.TFFunctionEnvV3(fn_dist=fn_dist,
                                               dims=dims,
                                               seed=seed,
                                               duration=env_steps,
                                               action_bounds=actions_bounds)

  # Instantiating the SummaryWriter's
  print('Creating logs directories.')
  log_dir, log_eval_dir, log_train_dir = training_utils.create_logs_dir(
    agent_dir)

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
    log_train_dir, flush_millis=summary_flush_secs * 1000)
  train_summary_writer.set_as_default()

  # Instantiating the metrics.
  train_metrics = [tf_metrics.AverageReturnMetric(buffer_size=100),
                   tf_metrics.MaxReturnMetric(buffer_size=100)]

  # Agent, Neural Networks and Optimizers.
  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  if actor_layers is None:
    actor_layers = LayersLSTMTD3(memory_fc_before_lstm=[128],
                                 memory_lstm_hidden=[128],
                                 memory_fc_after_lstm=[],
                                 fc_current_feature=[128],
                                 fc_after_concat=[128])

  if critic_layers is None:
    critic_layers = LayersLSTMTD3(memory_fc_before_lstm=[128],
                                  memory_lstm_hidden=[128],
                                  memory_fc_after_lstm=[],
                                  fc_current_feature=[128],
                                  fc_after_concat=[128])

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  # Instantiation of lstm_td3_agent
  agent = lstm_td3_agent.LSTMTD3Agent(
    time_spec,
    act_spec,
    obs_spec,
    history_length=history_length,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    actor_memory_fc_before_lstm=actor_layers.memory_fc_before_lstm,
    actor_memory_lstm_hidden=actor_layers.memory_lstm_hidden,
    actor_memory_fc_after_lstm=actor_layers.memory_fc_after_lstm,
    actor_fc_current_feature=actor_layers.fc_current_feature,
    actor_fc_after_concat=actor_layers.fc_after_concat,
    critic_memory_fc_before_lstm=critic_layers.memory_fc_before_lstm,
    critic_memory_lstm_hidden=critic_layers.memory_lstm_hidden,
    critic_memory_fc_after_lstm=critic_layers.memory_fc_after_lstm,
    critic_fc_current_feature=critic_layers.fc_current_feature,
    critic_fc_after_concat=critic_layers.fc_after_concat,
    exploration_noise_std=exploration_noise_std,
    target_update_tau=tau,
    target_update_period=target_update_period,
    actor_update_period=actor_update_period,
    gamma=discount,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=train_step)
  agent.initialize()

  # Creating the Replay Buffer and drivers.
  replay_buffer = tf_lstm_td3_replay_buffer.TFLSTMTD3ReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    episode_length=env_steps + 1,
    max_length=buffer_size)

  observers_train = [replay_buffer.add_batch] + train_metrics
  driver = dy_sd.DynamicStepDriver(env=tf_env_training,
                                   policy=agent.collect_policy,
                                   observers=observers_train,
                                   num_steps=collect_steps_per_iteration)

  initial_collect_driver = dy_ed.DynamicEpisodeDriver(
    env=tf_env_training,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=initial_collect_episodes)

  # Converting to tf.Function's
  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  agent.train = common.function(agent.train)

  print('Initializing replay buffer by collecting experience for {0} '
        'episodes with a collect policy.'.format(initial_collect_episodes))
  initial_collect_driver.run()

  replay_buffer.get_next(sample_batch_size=batch_size,
                         num_steps=history_length + 2)

  # Creating the dataset
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.AUTOTUNE,
    sample_batch_size=batch_size,
    num_steps=history_length + 2).prefetch(tf.data.AUTOTUNE)

  iterator = iter(dataset)
  agent.train_step_counter.assign(0)

  @tf.function
  def train_phase():
    print('tracing')
    driver.run()
    experience, _ = next(iterator)
    agent.train(experience)

  # Dictionary containing the hyperparameters
  hp_dict = {
    "seed": seed,
    "discount": discount,
    "exploration_noise_std": exploration_noise_std,
    "tau": tau,
    "target_update_period": target_update_period,
    "actor_update_period": actor_update_period,
    "training_episodes": training_episodes,
    "buffer_size": buffer_size,
    "batch_size": batch_size,
    "stop_threshold": stop_threshold,
    "train_env": {
      "steps": env_steps,
      "functions": [f.name for flist in functions for f in flist],
      "dims": dims,
      "domains": [f.domain for flist in functions for f in flist]
    },
    "algorithm": type(agent).__name__,
    "optimizers": {
      "actor_optimizer": type(actor_optimizer).__name__,
      "actor_lr": actor_lr,
      "critic_optimizer": type(critic_optimizer).__name__,
      "critic_lr": critic_lr
    }
  }

  training_utils.save_specs(agent_dir, hp_dict)
  tf.summary.text("Hyperparameters",
                  training_utils.json_pretty_string(hp_dict),
                  step=0)

  # Training phase
  for ep in range(training_episodes):
    start_time = time.time()

    if ep % eval_interval == 0:
      result = evaluate_agent(agent=agent,
                              function=tff.Sphere(),
                              dims=dims,
                              steps=eval_steps,
                              episodes=eval_episodes)
      if stop_threshold is not None and result < stop_threshold:
        break

    for _ in range(env_steps):
      train_phase()

      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=agent.train_step_counter)

    delta_time = time.time() - start_time
    print('Finished episode {0}. '
          'Delta time since last episode: {1:.2f}'.format(ep, delta_time))

  # Saving the learned policy.
  training_utils.save_policy(agent_dir, agent.policy)


if __name__ == '__main__':
  # tf.config.run_functions_eagerly(True)
  # tf.data.experimental.enable_debug_mode()
  train_lstm_td3(functions=[[tff.SchumerSteiglitz(),
                             tff.PowellSum(),
                             tff.SumSquares()]],
                 actions_bounds=(-100.0, 100.0),
                 dist_bounds=DistributionBounds(
                   vshift_bounds=(-100.0, 100.0),
                   hshift_bounds=(-15.0, 15.0),
                   scale_bounds=(-4.0, 4.0),
                   dims_params=2),
                 dims=2,
                 stop_threshold=1e-3,
                 seed=0)
