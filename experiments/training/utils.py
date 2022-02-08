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

from optfuncs import core

from sarlopt import config
from sarlopt.policies import lstm_td3_policies

ROOT_DIR = config.ROOT_DIR
OUTPUT_DIR = config.OUTPUT_DIR


def create_logs_dir(agent_dir: str):
  log_dir = os.path.join(agent_dir, 'logs')
  log_eval_dir = os.path.join(log_dir, 'eval')
  log_train_dir = os.path.join(log_dir, 'train')

  tf_io.gfile.makedirs(log_eval_dir)
  tf_io.gfile.makedirs(log_train_dir)

  return log_dir, log_eval_dir, log_train_dir


def save_policy(agent_dir: str,
                policy: tf_policy.TFPolicy):
  output_dir = os.path.join(agent_dir, 'policy')
  tf_io.gfile.makedirs(output_dir)

  if isinstance(policy, lstm_td3_policies.LSTMTD3ActorPolicy) or \
     isinstance(policy, lstm_td3_policies.LSTMTD3GaussianPolicy):
    print('LSTM-TD3 policies only support saving weights and spec.')
    print('Custom loading is necessary after saved.')
    net = policy.actor_network(copy=True)
    input_spec = net.input_tensor_spec
    output_spec = net.output_tensor_spec
    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'input_spec.pbtxt'),
                              input_spec)
    tensor_spec.to_pbtxt_file(os.path.join(output_dir, 'output_spec.pbtxt'),
                              output_spec)
    f = h5py.File(os.path.join(output_dir, 'weights.hdf5'), 'w')
    for layer in net.layers:
      weights = layer.get_weights()
      layer_group = f.create_group(layer.name)
      for i, wlist in enumerate(weights):
        layer_group.create_dataset(name=f'{i}', data=wlist)
    f.close()
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
