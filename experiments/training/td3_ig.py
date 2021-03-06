"""TD3-IG for learning an optimization algorithm."""

import time

import tensorflow as tf
from tf_agents.agents.ddpg import critic_network as critic_net
from tf_agents.drivers import dynamic_step_driver as dy_sd
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from optfuncs import tensorflow_functions as tff

from sarlopt.agents import td3_inverting_gradients as td3_ig
from sarlopt.environments import tf_function_env as tf_fun_env
from sarlopt.networks import linear_actor_network as linear_actor_net
from sarlopt.typing.types import LayerParam
from sarlopt.metrics import tf_custom_metrics

from experiments.evaluation import utils as eval_utils
from experiments.training import utils as training_utils


def td3_ig_train(function: tff.TensorflowFunction,
                 dims: int,
                 seed=1000,
                 training_episodes: int = 2000,
                 stop_threshold: float = None,
                 env_steps: int = 250,
                 env_eval_steps: int = 500,
                 eval_interval: int = 100,
                 eval_episodes: int = 10,
                 initial_collect_episodes: int = 20,
                 collect_steps_per_iteration: int = 1,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 tau: float = 5e-3,
                 actor_update_period: int = 2,
                 target_update_period: int = 2,
                 discount: float = 0.99,
                 exploration_noise_std: float = 0.5,
                 exploration_noise_std_end: float = 0.1,
                 exploration_noise_num_episodes: int = 1850,
                 target_policy_noise: float = 0.2,
                 target_policy_noise_clip: float = 0.5,
                 actor_layers: LayerParam = None,
                 critic_action_layers: LayerParam = None,
                 critic_observation_layers: LayerParam = None,
                 critic_joint_layers: LayerParam = None,
                 summary_flush_secs: int = 10,
                 debug_summaries: bool = False,
                 summarize_grads_and_vars: bool = False):
  algorithm_name = 'TD3-IG'

  # Creating the training/agent directory
  agent_dir = training_utils.create_agent_dir(algorithm_name,
                                              function,
                                              dims)

  # Calculating the number of exploration steps
  exploration_noise_num_steps = round(exploration_noise_num_episodes *
                                      env_steps)

  # Creating the environments.
  tf_env_training = tf_fun_env.TFFunctionEnv(function=function,
                                             dims=dims,
                                             seed=seed,
                                             duration=env_steps,
                                             bounded_actions_spec=False)
  tf_env_eval = tf_fun_env.TFFunctionEnv(function=function,
                                         dims=dims,
                                         seed=seed,
                                         duration=env_eval_steps,
                                         bounded_actions_spec=False)

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
  train_metrics = [tf_metrics.AverageReturnMetric(),
                   tf_metrics.MaxReturnMetric()]

  eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=eval_episodes),
                  tf_custom_metrics.AverageBestObjectiveValueMetric(
                    function=function, buffer_size=eval_episodes)]

  # Agent, Neural Networks and Optimizers.
  obs_spec = tf_env_training.observation_spec()
  act_spec = tf_env_training.action_spec()
  time_spec = tf_env_training.time_step_spec()

  if actor_layers is None:
    actor_layers = [256, 256]

  actor_activation_fn = tf.keras.activations.relu

  actor_network = linear_actor_net.LinearActorNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layers,
    activation_fn=actor_activation_fn)

  if critic_joint_layers is None:
    critic_joint_layers = [256, 256]

  critic_activation_fn = tf.keras.activations.relu
  critic_output_activation_fn = tf.keras.activations.linear

  critic_network = critic_net.CriticNetwork(
    input_tensor_spec=(obs_spec, act_spec),
    observation_fc_layer_params=critic_observation_layers,
    action_fc_layer_params=critic_action_layers,
    joint_fc_layer_params=critic_joint_layers,
    activation_fn=critic_activation_fn,
    output_activation_fn=critic_output_activation_fn)

  actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
  critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

  train_step = train_utils.create_train_step()

  agent = td3_ig.Td3AgentInvertingGradients(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=tau,
    exp_noise_std=exploration_noise_std,
    exp_noise_std_end=exploration_noise_std_end,
    exp_noise_steps=exploration_noise_num_steps,
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
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

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
    "exploration_noise_std_end": exploration_noise_std_end,
    "exploration_noise_num_episodes": exploration_noise_num_episodes,
    "exploration_noise_num_steps": exploration_noise_num_steps,
    "tau": tau,
    "target_update_period": target_update_period,
    "actor_update_period": actor_update_period,
    "training_episodes": training_episodes,
    "buffer_size": buffer_size,
    "batch_size": batch_size,
    "stop_threshold": stop_threshold,
    "train_env": {
      "steps": env_steps,
      "function": function.name,
      "dims": dims,
      "domain": function.domain
    },
    "eval_env": {
      "steps": env_eval_steps,
      "function": function.name,
      "dims": dims,
      "domain": function.domain
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
        "output_activation_fn": critic_output_activation_fn.__name__,
        "critic_action_fc_layers": critic_action_layers,
        "critic_obs_fc_layers": critic_observation_layers,
        "critic_joint_layers": critic_joint_layers
      }
    },
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
    for _ in range(env_steps):
      train_phase()

      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=agent.train_step_counter)

    if ep % eval_interval == 0:
      print('-------- Evaluation --------')
      start_eval = time.time()
      results = compute_eval_metrics()
      avg_return = results.get(eval_metrics[0].name)
      avg_best_value = results.get(eval_metrics[1].name)
      print('Average return: {0}'.format(avg_return))
      print('Average best value: {0}'.format(avg_best_value))
      print('Eval delta time: {0:.2f}'.format(time.time() - start_eval))
      print('---------------------------')
      if stop_threshold is not None and avg_best_value < stop_threshold:
        break

    delta_time = time.time() - start_time
    print('Finished episode {0}. '
          'Delta time since last episode: {1:.2f}'.format(ep, delta_time))

  # Computing metrics after training.
  compute_eval_metrics()

  # Policy evaluation for 100 episodes.
  # Outputs a convergence plot for the policy in the function.
  eval_utils.evaluate_agent(tf_env_eval,
                            agent.policy,
                            function,
                            dims,
                            env_eval_steps,
                            algorithm_name=algorithm_name,
                            save_to_file=True,
                            episodes=100,
                            save_dir=agent_dir)

  # Saving the learned policy.
  # Output directory: output/DDPG-{dims}D-{function.name}-{num}/policy
  training_utils.save_policy(agent_dir, agent.policy)


if __name__ == '__main__':
  td3_ig_train(tff.Sphere(), 2,
               env_steps=50,
               eval_interval=10,
               env_eval_steps=100,
               stop_threshold=1e-2,
               training_episodes=100)
