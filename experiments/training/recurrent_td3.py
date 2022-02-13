"""Recurrent TD3 for l2o."""

import time
import typing
import numpy as np

import tensorflow as tf

from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.ddpg import actor_rnn_network as actor_net
from tf_agents.agents.ddpg import critic_rnn_network as critic_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.policies import tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils

from optfuncs import tensorflow_functions as tff

from sarlopt.environments import tf_function_env_v3 as tf_fun_env
from sarlopt.typing.types import LayerParam
from sarlopt.utils.functions import distributions as fn_distributions

from experiments.evaluation import utils as eval_utils
from experiments.training import utils as training_utils

ts = trajectory.ts


class DistributionBounds(typing.NamedTuple):
  vshift_bounds: fn_distributions.ParamBound
  hshift_bounds: fn_distributions.ParamBound
  scale_bounds: fn_distributions.ParamBound
  dims_params: int


def evaluate_recurrent_policy(policy: tf_policy.TFPolicy,
                              function: tff.TensorflowFunction,
                              dims: int,
                              steps: int,
                              episodes: int):
  rng = tf.random.Generator.from_non_deterministic_state()
  x = common.create_variable(name='x',
                             shape=tf.TensorShape([dims]),
                             initial_value=0,
                             dtype=tf.float32)
  best_fx = common.create_variable(name='best_fx',
                                   shape=(),
                                   initial_value=tf.float32.max,
                                   dtype=tf.float32)
  best_x = common.create_variable(name='best_x',
                                  shape=tf.TensorShape([dims]),
                                  initial_value=0,
                                  dtype=tf.float32)
  t = common.create_variable(name='t',
                             shape=(),
                             initial_value=0,
                             dtype=tf.int32)
  steps = tf.convert_to_tensor(steps, dtype=tf.int32)

  p = tf.constant(10, dtype=tf.float32)

  def log_processing(g: tf.Tensor):
    abs_g = tf.abs(g)
    return tf.cond(pred=tf.greater_equal(abs_g, tf.math.exp(-p)),
                   true_fn=lambda: tf.math.divide(tf.math.log(abs_g), p),
                   false_fn=lambda: tf.constant(-1.0, dtype=tf.float32))

  def sign_processing(g: tf.Tensor):
    return tf.cond(pred=tf.greater_equal(tf.abs(g), tf.math.exp(-p)),
                   true_fn=lambda: tf.sign(g),
                   false_fn=lambda: tf.multiply(g, tf.math.exp(-p)))

  def observation_to_time_step(observation) -> ts.TimeStep:
    return nest_utils.batch_nested_tensors(
          ts.TimeStep(step_type=ts.StepType.MID,
                      reward=0.0,
                      discount=tf.constant(1.0, dtype=tf.float32),
                      observation=observation))

  @tf.function(autograph=True)
  def run_episode():
    x0 = rng.uniform(
      shape=tf.TensorShape([dims]),
      minval=function.domain[0],
      maxval=function.domain[1],
      dtype=tf.float32)
    x.assign(value=x0)
    t.assign(value=0)
    best_fx.assign(value=tf.float32.max)
    best_x.assign(value=x0)

    policy_state = policy.get_initial_state(batch_size=1)
    fx = None

    while tf.reduce_all(tf.less(t, steps)):
      grads, fx = function.grads_at(x)

      if tf.math.reduce_all(tf.math.less(fx, best_fx)):
        best_fx.assign(fx)
        best_x.assign(x)

      log_grad = tf.map_fn(log_processing, grads)
      sign_grad = tf.map_fn(sign_processing, grads)
      avg_velocity = tf.math.divide_no_nan(x - x0, tf.cast(t, dtype=tf.float32))

      obs = tf.concat([log_grad, sign_grad, avg_velocity], axis=-1,
                      name='observation')
      time_step = observation_to_time_step(obs)
      policy_step = policy.action(time_step, policy_state)

      action = tf.squeeze(policy_step.action)
      policy_state = policy_step.state

      x.assign(tf.clip_by_value(x + action,
                                clip_value_min=function.domain[0],
                                clip_value_max=function.domain[1]))
      t.assign_add(1)

    if tf.math.reduce_all(tf.math.less(fx, best_fx)):
      best_fx.assign(fx)
      best_x.assign(x)

    return best_fx.value(), best_x.value()

  best_values = []
  best_solutions = []

  for ep in range(episodes):
    best_value, best_solution = run_episode()
    best_values.append(best_value.numpy())
    best_solutions.append(best_solution.numpy())

  avg_best_values = np.mean(best_values, axis=0)
  avg_best_solutions = np.mean(best_solutions, axis=0)
  print('Average best value: {0}'.format(avg_best_values))
  print('Average best solution: {0}'.format(avg_best_solutions))

  return avg_best_values, avg_best_solutions


def train_recurrent_td3(functions: typing.List[fn_distributions.FunctionList],
                        dist_bounds: DistributionBounds,
                        dims: int,
                        actions_bounds: typing.Tuple[float, float],
                        train_sequence_length: int,
                        seed=1000,
                        training_episodes: int = 2000,
                        stop_threshold: float = None,
                        env_steps: int = 50,
                        env_eval_steps: int = 150,
                        eval_interval: int = 20,
                        eval_episodes: int = 10,
                        initial_collect_episodes: int = 20,
                        collect_steps_per_iteration: int = 1,
                        buffer_size: int = 1000000,
                        batch_size: int = 64,
                        actor_lr: float = 3e-4,
                        critic_lr: float = 3e-4,
                        tau: float = 5e-3,
                        actor_update_period: int = 2,
                        target_update_period: int = 2,
                        discount: float = 0.99,
                        exploration_noise_std: float = 0.1,
                        target_policy_noise: float = 0.2,
                        target_policy_noise_clip: float = 0.5,
                        actor_layers: LayerParam = None,
                        critic_action_layers: LayerParam = None,
                        critic_observation_layers: LayerParam = None,
                        critic_joint_layers: LayerParam = None,
                        summary_flush_secs: int = 10,
                        debug_summaries: bool = False,
                        summarize_grads_and_vars: bool = False):
  algorithm_name = 'Recurrent-TD3'

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

  # Creating the environments.
  tf_env_training = tf_fun_env.TFFunctionEnvV3(fn_dist=fn_dist,
                                               dims=dims,
                                               seed=seed,
                                               duration=env_steps,
                                               action_bounds=actions_bounds)
  tf_env_eval = tf_fun_env.TFFunctionEnvV3(fn_dist=fn_dist,
                                           dims=dims,
                                           seed=seed,
                                           duration=env_eval_steps,
                                           action_bounds=actions_bounds)

  # Instantiating the SummaryWriter's
  print('Creating logs directories.')
  log_dir, log_eval_dir, log_train_dir = training_utils.create_logs_dir(
    agent_dir)

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
    log_train_dir, flush_millis=summary_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
    log_eval_dir, flush_millis=summary_flush_secs * 1000)

  # Instantiating the metrics.
  train_metrics = [tf_metrics.AverageReturnMetric(buffer_size=200),
                   tf_metrics.MaxReturnMetric(buffer_size=200)]

  eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=eval_episodes),
                  tf_metrics.MaxReturnMetric(buffer_size=eval_episodes)]

  # Agent, Neural Networks and Optimizers.
  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  if actor_layers is None:
    actor_layers = [256, 256]

  actor_activation_fn = tf.keras.activations.relu

  actor_network = actor_net.ActorRnnNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    input_fc_layer_params=actor_layers,
    lstm_size=(40,),
    output_fc_layer_params=(200, 100),
    activation_fn=actor_activation_fn)

  if critic_joint_layers is None:
    critic_joint_layers = [256, 256]

  critic_activation_fn = tf.keras.activations.relu

  critic_network = critic_net.CriticRnnNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=(200,),
    action_fc_layer_params=(200,),
    joint_fc_layer_params=(100,),
    lstm_size=(40,),
    output_fc_layer_params=(200, 100),
    activation_fn=critic_activation_fn)

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  agent = td3_agent.Td3Agent(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exploration_noise_std=exploration_noise_std,
    target_policy_noise=target_policy_noise,
    target_policy_noise_clip=target_policy_noise_clip,
    actor_update_period=actor_update_period,
    target_update_period=target_update_period,
    train_step_counter=train_step,
    gamma=discount,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars)

  agent.initialize()

  # Creating the Replay Buffer and drivers.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
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

  eval_driver = dy_ed.DynamicEpisodeDriver(env=tf_env_eval,
                                           policy=agent.policy,
                                           observers=eval_metrics,
                                           num_episodes=eval_episodes)

  # Converting to tf.Function's
  initial_collect_driver.run = common.function(initial_collect_driver.run)
  driver.run = common.function(driver.run)
  eval_driver.run = common.function(eval_driver.run)
  agent.train = common.function(agent.train)

  print('Initializing replay buffer by collecting experience for {0} '
        'episodes with a collect policy.'.format(initial_collect_episodes))
  initial_collect_driver.run()

  # Creating the dataset
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=tf.data.AUTOTUNE,
    sample_batch_size=batch_size,
    num_steps=train_sequence_length).prefetch(tf.data.AUTOTUNE)

  iterator = iter(dataset)

  # Metric evaluation function
  def compute_eval_metrics():
    return eval_utils.eager_compute(eval_metrics,
                                    eval_driver,
                                    train_step=agent.train_step_counter,
                                    summary_writer=eval_summary_writer,
                                    summary_prefix='Metrics')

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
    "networks": {
      "actor_net": {
        "class": type(actor_network).__name__,
        "activation_fn": actor_activation_fn.__name__,
        "actor_layers": actor_layers
      },
      "critic_net": {
        "class": type(critic_network).__name__,
        "activation_fn": critic_activation_fn.__name__,
        "critic_action_fc_layers": critic_action_layers,
        "critic_obs_fc_layers": critic_observation_layers,
        "critic_joint_layers": critic_joint_layers
      }
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
    # Evaluation
    if ep % eval_interval == 0:
      print('-------- Evaluation --------')
      start_eval = time.time()
      results = compute_eval_metrics()
      avg_return = results.get(eval_metrics[0].name)
      max_return = results.get(eval_metrics[1].name)
      print('Average return: {0}'.format(avg_return))
      print('Max return: {0}'.format(max_return))
      print('Eval delta time: {0:.2f}'.format(time.time() - start_eval))
      print('---------------------------')

    # Training
    start_time = time.time()
    for _ in range(env_steps):
      train_phase()

      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=agent.train_step_counter)

    delta_time = time.time() - start_time
    print('Finished episode {0}. '
          'Delta time since last episode: {1:.2f}'.format(ep, delta_time))

  # Computing metrics after training.
  compute_eval_metrics()

  # Saving the learned policy.
  # Output directory: output/TD3-{dims}D-{function.name}-{num}/policy
  training_utils.save_policy(agent_dir, agent.policy)


if __name__ == '__main__':
  train_recurrent_td3(functions=[[tff.SchumerSteiglitz(),
                                  tff.SumSquares(),
                                  tff.PowellSum()],
                                 [tff.ChungReynolds(),
                                  tff.Schwefel()],
                                 [tff.Brown(),
                                  tff.DixonPrice(),
                                  tff.Schwefel12(),
                                  tff.Schwefel222(),
                                  tff.Schwefel223(),
                                  tff.StrechedVSineWave()],
                                 [tff.Alpine2(),
                                  tff.Csendes(),
                                  tff.Deb1(),
                                  tff.Deb3(),
                                  tff.Qing(),
                                  tff.Schwefel226(),
                                  tff.WWavy(),
                                  tff.Weierstrass()],
                                 [tff.Exponential(),
                                  tff.Mishra2(),
                                  tff.Salomon(),
                                  tff.Sargan(),
                                  tff.Trigonometric2(),
                                  tff.Whitley(),
                                  tff.Zakharov()]
                                 ],
                      actions_bounds=(-1.0, 1.0),
                      dist_bounds=DistributionBounds(
                        vshift_bounds=(-5.0, 5.0),
                        hshift_bounds=(-1.5, 1.5),
                        scale_bounds=(0.5, 1.5),
                        dims_params=2),
                      dims=2,
                      train_sequence_length=10,
                      seed=0,
                      training_episodes=500)
