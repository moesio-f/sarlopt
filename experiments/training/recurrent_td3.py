"""Recurrent TD3 for L2O."""

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

from optfuncs import tensorflow_functions as tff

from sarlopt.environments import tf_function_env_v3 as tf_fun_env
from sarlopt.utils.functions import distributions as fn_distributions

from experiments.evaluation import utils as eval_utils
from experiments.evaluation import tf_env3_evaluation
from experiments.training import utils as training_utils

ts = trajectory.ts


class DistributionBounds(typing.NamedTuple):
  vshift_bounds: fn_distributions.ParamBound
  hshift_bounds: fn_distributions.ParamBound
  scale_bounds: fn_distributions.ParamBound
  dims_params: int


class ActorNetworkParams(typing.NamedTuple):
  input_fc_layer_params: typing.Optional[typing.List[int]]
  lstm_size: typing.Optional[typing.List[int]]
  output_fc_layer_params: typing.Optional[typing.List[int]]
  activation_fn: typing.Optional[typing.Any]


class CriticNetworkParams(typing.NamedTuple):
  observation_fc_layer_params: typing.Optional[typing.List[int]]
  action_fc_layer_params: typing.Optional[typing.List[int]]
  joint_fc_layer_params: typing.Optional[typing.List[int]]
  lstm_size: typing.Optional[typing.List[int]]
  output_fc_layer_params: typing.Optional[typing.List[int]]
  activation_fn: typing.Optional[typing.Any]


def train_recurrent_td3(functions: typing.List[fn_distributions.FunctionList],
                        dist_bounds: DistributionBounds,
                        dims: int,
                        actions_bounds: typing.Tuple[float, float],
                        train_sequence_length: int,
                        curriculum_strategy: typing.List[
                          typing.Tuple[int, int]] = None,
                        seed=1000,
                        training_episodes: int = 2000,
                        stop_threshold: float = None,
                        env_steps: int = 50,
                        env_eval_steps: int = 150,
                        eval_interval: int = 20,
                        eval_episodes: int = 10,
                        initial_collect_episodes: int = 20,
                        collect_steps_per_iteration: int = 1,
                        buffer_size: int = 10000000,
                        batch_size: int = 64,
                        actor_lr: float = 3e-4,
                        critic_lr: float = 3e-4,
                        tau: float = 5e-3,
                        actor_update_period: int = 2,
                        target_update_period: int = 2,
                        discount: float = 0.99,
                        gradient_clip_norm: typing.Optional[float] = None,
                        exploration_noise_std: float = 0.1,
                        target_policy_noise: float = 0.2,
                        target_policy_noise_clip: float = 0.5,
                        actor_net_params: ActorNetworkParams = None,
                        critic_net_params: CriticNetworkParams = None,
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
  tf_env_training = tf_fun_env.TFFunctionEnvV3(
    fn_dist=fn_dist,
    dims=dims,
    seed=seed,
    duration=env_steps,
    action_bounds=actions_bounds,
    curriculum_strategy=curriculum_strategy)

  # Evaluation should sample from all possible optimizees.
  tf_env_eval = tf_fun_env.TFFunctionEnvV3(
    fn_dist=fn_dist,
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

  if actor_net_params is None:
    actor_net_params = ActorNetworkParams(
      input_fc_layer_params=[256, 256],
      lstm_size=[128],
      output_fc_layer_params=[256, 256],
      activation_fn=tf.keras.activations.relu)

  actor_network = actor_net.ActorRnnNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    input_fc_layer_params=actor_net_params.input_fc_layer_params,
    lstm_size=actor_net_params.lstm_size,
    output_fc_layer_params=actor_net_params.output_fc_layer_params,
    activation_fn=actor_net_params.activation_fn)

  if critic_net_params is None:
    critic_net_params = CriticNetworkParams(
      observation_fc_layer_params=[256],
      action_fc_layer_params=[256],
      joint_fc_layer_params=[128],
      lstm_size=[128],
      output_fc_layer_params=[256, 128],
      activation_fn=tf.keras.activations.relu)

  critic_network = critic_net.CriticRnnNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=critic_net_params.observation_fc_layer_params,
    action_fc_layer_params=critic_net_params.action_fc_layer_params,
    joint_fc_layer_params=critic_net_params.joint_fc_layer_params,
    lstm_size=critic_net_params.lstm_size,
    output_fc_layer_params=critic_net_params.output_fc_layer_params,
    activation_fn=critic_net_params.activation_fn)

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
    gradient_clipping=gradient_clip_norm,
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
        "activation_fn": actor_net_params.activation_fn.__name__,
        "input_fc_layer_params": actor_net_params.input_fc_layer_params,
        "lstm_size": actor_net_params.lstm_size,
        "output_fc_layer_params": actor_net_params.output_fc_layer_params
      },
      "critic_net": {
        "class": type(critic_network).__name__,
        "activation_fn": critic_net_params.activation_fn.__name__,
        "action_fc_layers": critic_net_params.action_fc_layer_params,
        "obs_fc_layers": critic_net_params.observation_fc_layer_params,
        "lstm_size": critic_net_params.lstm_size,
        "joint_layers": critic_net_params.joint_fc_layer_params,
        "output_fc_layer_params": critic_net_params.output_fc_layer_params
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
  # Output directory: output/Recurrent-TD3-{dims}D-distribution-{num}/policy
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
                        vshift_bounds=(0.0, 0.0),
                        hshift_bounds=(0.0, 0.0),
                        scale_bounds=(1.0, 1.0),
                        dims_params=2),
                      dims=2,
                      gradient_clip_norm=1.0,
                      env_steps=50,
                      env_eval_steps=200,
                      train_sequence_length=10,
                      seed=0,
                      training_episodes=100)
  # Problems with functions:
  #   Rosenbrock; (Fixed, removed batch)
  #   Dixon Price; (Fixed, removed batch)
  #   Griewank; (Fixed, removed batch)
  #   Levy; (Fixed, removed batch)
  #   RotatedHyperEllipsoid; (Not fixed, need to review non-batched formulation)
  # Temporary fix: remove batched inputs. It seems related to the map_fn(...),
  #   however it's hard to find the real culprit. All functions that used
  #   atleast_2d(...) had this problem.
