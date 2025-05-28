"""Policy evaluation tests."""

import os

import tensorflow as tf
from py_benchmark_functions.imp import tensorflow as tff

from experiments.evaluation import utils as eval_utils
from sarlopt import config
from sarlopt.environments import tf_function_env as tf_fun_env

POLICIES_DIR = config.POLICIES_DIR

if __name__ == "__main__":
    dims = 30
    function = tff.SphereTensorflow(dims)
    steps = 500
    seed = 10000

    policy_dir = os.path.join(POLICIES_DIR, "Td3Agent")
    policy_dir = os.path.join(policy_dir, f"{dims}D")
    policy_dir = os.path.join(policy_dir, function.name)

    saved_pol = tf.compat.v2.saved_model.load(policy_dir)

    tf_eval_env = tf_fun_env.TFFunctionEnv(
        function=function, dims=dims, duration=steps, seed=seed
    )

    # tf.config.run_functions_eagerly(True)
    eval_utils.evaluate_agent(
        eval_env=tf_eval_env,
        policy_eval=saved_pol,
        function=function,
        dims=dims,
        steps=steps,
        algorithm_name="TD3",
        save_to_file=False,
        episodes=10,
        save_dir=os.getcwd(),
    )
