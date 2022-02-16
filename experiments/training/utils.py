"""Training utilities for experiments."""

import os
import json
import typing
import h5py

import tensorflow as tf
from tensorflow import io as tf_io

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from optfuncs import core

from sarlopt import config
from sarlopt.policies import lstm_td3_policies
from sarlopt.networks import lstm_td3_actor_network

from experiments.training.lstm_td3 import LayersLSTMTD3

ROOT_DIR = config.ROOT_DIR
OUTPUT_DIR = config.OUTPUT_DIR


def create_logs_dir(agent_dir: str):
  log_dir = os.path.join(agent_dir, 'logs')
  log_eval_dir = os.path.join(log_dir, 'eval')
  log_train_dir = os.path.join(log_dir, 'train')

  tf_io.gfile.makedirs(log_eval_dir)
  tf_io.gfile.makedirs(log_train_dir)

  return log_dir, log_eval_dir, log_train_dir


def load_lstm_td3_policy(policy_dir: str) -> \
      lstm_td3_policies.LSTMTD3ActorPolicy:
  # Loading specs.
  input_spec = tensor_spec.from_pbtxt_file(os.path.join(policy_dir,
                                                        'input_spec.pbtxt'))
  input_spec = lstm_td3_actor_network.LSTMTD3InputActor(*input_spec)
  output_spec = tensor_spec.from_pbtxt_file(os.path.join(policy_dir,
                                                         'output_spec.pbtxt'))
  action_spec = tensor_spec.from_pbtxt_file(os.path.join(policy_dir,
                                                         'action_spec.pbtxt'))
  ts_spec = tensor_spec.from_pbtxt_file(os.path.join(policy_dir,
                                                     'time_step_spec.pbtxt'))
  ts_spec = time_step.TimeStep(*ts_spec)
  with open(os.path.join(policy_dir, 'layers_specs.json'), 'r') as specs_file:
    layers_specs = LayersLSTMTD3(*json.load(specs_file).values())

  # Create network.
  network = lstm_td3_actor_network.LSTMTD3ActorNetwork(
    input_spec,
    output_spec,
    memory_fc_before_lstm=layers_specs.memory_fc_before_lstm,
    memory_lstm_hidden=layers_specs.memory_lstm_hidden,
    memory_fc_after_lstm=layers_specs.memory_fc_after_lstm,
    fc_current_feature=layers_specs.fc_current_feature,
    fc_after_concat=layers_specs.fc_after_concat)
  network.create_variables()

  # Load weights.
  w_file = h5py.File(os.path.join(policy_dir, 'weights.hdf5'), 'r')
  sorted_layers = sorted(network.layers, key=lambda x: x.name)
  sorted_keys = sorted(list(w_file.keys()))

  assert len(sorted_layers) == len(sorted_keys)

  for (layer, layer_group_str) in zip(sorted_layers, sorted_keys):
    layer_group = w_file.get(layer_group_str)
    list_weights = [data[:] for data in layer_group.values()]
    layer.set_weights(list_weights)

  w_file.close()

  # Create policy.
  return lstm_td3_policies.LSTMTD3ActorPolicy(
    action_spec=action_spec,
    time_step_spec=ts_spec,
    actor_network=network,
    history_length=input_spec.history.shape[0])


def save_policy(agent_dir: str,
                policy: tf_policy.TFPolicy,
                **kwargs):
  output_dir = os.path.join(agent_dir, 'policy')
  tf_io.gfile.makedirs(output_dir)

  if isinstance(policy, lstm_td3_policies.LSTMTD3ActorPolicy) or \
     isinstance(policy, lstm_td3_policies.LSTMTD3GaussianPolicy):
    assert 'layers_specs' in kwargs
    assert isinstance(kwargs['layers_specs'], typing.Tuple)

    print('LSTM-TD3 policies only support saving weights and spec.')
    print('Custom loading is necessary after saved.')
    net = policy.actor_network(copy=True)
    layers_specs = kwargs['layers_specs']
    input_spec = net.input_tensor_spec
    output_spec = net.output_tensor_spec

    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'input_spec.pbtxt'),
                              input_spec)
    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'output_spec.pbtxt'),
                              output_spec)
    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'action_spec.pbtxt'),
                              policy.action_spec)
    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'time_step_spec.pbtxt'),
                              policy.time_step_spec)

    f = h5py.File(os.path.join(output_dir, 'weights.hdf5'), 'w')
    for layer in sorted(net.layers, key=lambda x: x.name):
      weights = layer.get_weights()
      layer_group = f.create_group(layer.name.replace('/', '-'))
      for i, wlist in enumerate(weights):
        layer_group.create_dataset(name=f'{i}', data=wlist)
    f.close()

    with open(os.path.join(output_dir, 'layers_specs.json'), 'w') as specs_file:
      # noinspection PyProtectedMember
      json.dump(layers_specs._asdict(), specs_file, indent=True)

  else:
    tf_policy_saver = policy_saver.PolicySaver(policy)
    tf_policy_saver.save(output_dir)


def save_specs(agent_dir: str,
               specs_dict: typing.Dict):
  with open(os.path.join(agent_dir, 'specs.json'), 'w') as specs_file:
    json.dump(specs_dict, specs_file, indent=True)


def create_agent_dir_str(algorithm_name: str,
                         info_str: str,
                         dims: int) -> str:
  str_dims = str(dims)
  agent_identifier = f'{algorithm_name}-{str_dims}D-{info_str}-0'
  agent_dir = os.path.join(OUTPUT_DIR, agent_identifier)

  # TODO: Reduce time complexity.
  i = 0
  while os.path.exists(agent_dir):
    i += 1
    agent_identifier = f'{algorithm_name}-{str_dims}D-{info_str}-{i}'
    agent_dir = os.path.join(OUTPUT_DIR, agent_identifier)

  tf_io.gfile.makedirs(agent_dir)
  return agent_dir


def create_agent_dir(algorithm_name: str,
                     function: core.Function,
                     dims: int) -> str:
  return create_agent_dir_str(algorithm_name, function.name, dims)


def json_pretty_string(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))
