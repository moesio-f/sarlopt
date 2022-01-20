"""REINFORCE for learning an optimization algorithm."""

import time

import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.networks import actor_distribution_network as actor_net
from tf_agents.networks import value_network as value_net
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from optfuncs import tensorflow_functions as tff

from src.environments import tf_function_environment as tf_fun_env
from src.typing.types import LayerParam
from src.metrics import tf_custom_metrics

from experiments.evaluation import utils as eval_utils
from experiments.training import utils as training_utils


def reinforce_train(function: tff.TensorflowFunction,
                    dims: int,
                    seed=10000,
                    training_episodes: int = 2000,
                    stop_threshold: float = None,
                    env_steps: int = 250,
                    env_eval_steps: int = 500,
                    eval_interval: int = 100,
                    eval_episodes: int = 10,
                    lr: float = 3e-4,
                    discount: float = 0.99,
                    actor_layers: LayerParam = None,
                    value_layers: LayerParam = None,
                    summary_flush_secs: int = 10,
                    debug_summaries: bool = False,
                    summarize_grads_and_vars: bool = False):
  algorithm_name = 'REINFORCE'

  # Creating the training/agent directory
  agent_dir = training_utils.create_agent_dir(algorithm_name,
                                              function,
                                              dims)

  # Conversion to TFPyEnvironment's.
  tf_env_training = tf_fun_env.TFFunctionEnv(function=function,
                                             dims=dims,
                                             seed=seed,
                                             duration=env_steps)
  tf_env_eval = tf_fun_env.TFFunctionEnv(function=function,
                                         dims=dims,
                                         seed=seed,
                                         duration=env_eval_steps)

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

  actor_network = actor_net.ActorDistributionNetwork(
    input_tensor_spec=obs_spec,
    output_tensor_spec=act_spec,
    fc_layer_params=actor_layers)

  if value_layers is None:
    value_layers = [256, 256]

  value_network = value_net.ValueNetwork(input_tensor_spec=obs_spec,
                                         fc_layer_params=value_layers)

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

  train_step = train_utils.create_train_step()

  agent = reinforce_agent.ReinforceAgent(
    time_step_spec=time_spec,
    action_spec=act_spec,
    actor_network=actor_network,
    value_network=value_network,
    optimizer=optimizer,
    gamma=discount,
    normalize_returns=False,
    train_step_counter=train_step,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars)

  agent.initialize()

  # Creating the Replay Buffer and drivers.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_training.batch_size,
    max_length=env_steps + 5)

  observers_train = [replay_buffer.add_batch] + train_metrics
  driver = dy_ed.DynamicEpisodeDriver(env=tf_env_training,
                                      policy=agent.collect_policy,
                                      observers=observers_train,
                                      num_episodes=1)

  eval_driver = dy_ed.DynamicEpisodeDriver(env=tf_env_eval,
                                           policy=agent.policy,
                                           observers=eval_metrics,
                                           num_episodes=eval_episodes)

  # Converting to tf.Function's
  driver.run = common.function(driver.run)
  eval_driver.run = common.function(eval_driver.run)
  agent.train = common.function(agent.train)

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
    experience = replay_buffer.gather_all()
    agent.train(experience)
    replay_buffer.clear()

    # Dictionary containing the hyperparameters
    hp_dict = {
      "seed": seed,
      "discount": discount,
      "training_episodes": training_episodes,
      "stop_threshold": stop_threshold,
      "learning_rate": lr,
      "optimizer": type(optimizer).__name__,
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
          "actor_layers": actor_layers
        },
        "value_net": {
          "class": type(value_network).__name__,
          "value_layers": value_layers
        }
      }
    }

    training_utils.save_specs(agent_dir, hp_dict)
    tf.summary.text("Hyperparameters",
                    training_utils.json_pretty_string(hp_dict),
                    step=0)

  # Training phase
  for ep in range(training_episodes):
    start_time = time.time()
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
  # Output directory: output/REINFORCE-{dims}D-{function.name}-{num}/policy
  training_utils.save_policy(agent_dir, agent.policy)


if __name__ == '__main__':
  reinforce_train(tff.Sphere(), 2,
                  stop_threshold=1e-2,
                  eval_interval=10,
                  training_episodes=500,
                  debug_summaries=False)
