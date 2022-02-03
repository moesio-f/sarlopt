"""Tests for sarlopt.functions.distributions."""
from math import pi

import tensorflow as tf

from optfuncs import core
from optfuncs import tensorflow_functions as tff

from sarlopt.utils.functions.distributions import UniformFunctionDistribution


class SepUnimodalF1(tff.TensorflowFunction):
  def __init__(self):
    super(SepUnimodalF1, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(x)


class SepUnimodalF2(tff.TensorflowFunction):
  def __init__(self):
    super(SepUnimodalF2, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: tf.Tensor):
    return tf.reduce_sum(tf.pow(x, 2))


class SepMultimodalF1(tff.TensorflowFunction):
  def __init__(self):
    super(SepMultimodalF1, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: tf.Tensor):
    return tf.reduce_prod(tf.multiply(tf.sqrt(x), tf.sin(x)), axis=-1)


class SepMultimodalF2(tff.TensorflowFunction):
  def __init__(self):
    super(SepMultimodalF2, self).__init__(core.Domain(-100.0, 100.0))

  def _call(self, x: tf.Tensor):
    d = x.shape[-1]
    return -tf.divide(tf.reduce_sum(tf.pow(tf.sin(tf.multiply(x, 5 * pi)), 6),
                                    axis=-1), d)


class PartUnimodalF1(tff.TensorflowFunction):
  def __init__(self):
    super(PartUnimodalF1, self).__init__(core.Domain(-50.0, 50.0))

  def _call(self, x: tf.Tensor):
    return tf.pow(tf.reduce_sum(tf.pow(x, 2), axis=-1), 2)


class UniformFunctionDistributionTest(tf.test.TestCase):
  def setUp(self):
    super(UniformFunctionDistributionTest, self).setUp()

    self.seed = 100
    self.alg = tf.random.Algorithm.PHILOX
    self.dtype = tf.float32
    self.dims = 2
    self.functions = [[SepUnimodalF1(), SepUnimodalF2()],
                      [SepMultimodalF1(), SepMultimodalF2()],
                      [PartUnimodalF1()]]
    for fn in tf.nest.flatten(self.functions):
      fn.enable_tf_function()
    self.vshift_bounds = (-2.0, 2.0)
    self.hshift_bounds = (-10.0, 10.0)
    self.scale_bounds = (0.5, 2.0)

  def test_first_sample(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()

    x = tf.constant([1.0, 1.0], dtype=self.dtype)
    expected_fx = tf.multiply(
      self.functions[1][1](x + tf.constant([3.3239317, 4.938116],
                                           dtype=self.dtype)),
      0.8840528) + 1.7433386
    self.assertEqual(expected_fx, fn_distribution(x))

  def test_multiple_samples(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()
    x = tf.constant([1.0, 1.0], dtype=self.dtype)

    fn_distribution.sample()
    expected_fx = tf.multiply(
      self.functions[0][0](x + tf.constant([3.9939976, -9.624966],
                                           dtype=self.dtype)),
      1.8881679) - 1.544632
    self.assertEqual(expected_fx, fn_distribution(x))

    fn_distribution.sample()
    expected_fx = tf.multiply(
      self.functions[2][0](x + tf.constant([-9.09575, -3.2940674],
                                           dtype=self.dtype)),
      1.4871273) - 0.48787737
    self.assertEqual(expected_fx, fn_distribution(x))

    fn_distribution.sample()
    expected_fx = tf.multiply(
      self.functions[0][0](x + tf.constant([6.7230206, -9.136286],
                                           dtype=self.dtype)),
      1.2844284) + 0.5755615
    self.assertEqual(expected_fx, fn_distribution(x))

  def test_class_samples(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()
    x = tf.constant([1.0, 1.0], dtype=self.dtype)

    fn_distribution.sample_from_class(tf.constant(1, dtype=tf.int32))
    expected_fx = tf.multiply(
      self.functions[1][0](x + tf.constant([3.9939976, -9.624966],
                                           dtype=self.dtype)),
      1.8881679) - 1.544632
    self.assertEqual(tf.math.is_nan(expected_fx),
                     tf.math.is_nan(fn_distribution(x)))

    for _ in range(6):
      fn_distribution.sample_from_class(tf.constant(1, dtype=tf.int32))

    expected_fx = tf.multiply(
      self.functions[1][1](x + tf.constant([-3.3616877, -8.070612],
                                           dtype=self.dtype)),
      0.883859) + 0.805696
    self.assertEqual(expected_fx, fn_distribution(x))

  def test_function_samples(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()
    x = tf.constant([1.0, 1.0], dtype=self.dtype)

    fn_distribution.sample_from_function(tf.constant(1, dtype=tf.int32),
                                         tf.constant(1, dtype=tf.int32))
    expected_fx = tf.multiply(
      self.functions[1][1](x + tf.constant([3.9939976, -9.624966],
                                           dtype=self.dtype)),
      1.8881679) - 1.544632
    self.assertEqual(expected_fx, fn_distribution(x))

    fn_distribution.sample_from_function(tf.constant(1, dtype=tf.int32),
                                         tf.constant(1, dtype=tf.int32))
    expected_fx = tf.multiply(
      self.functions[1][1](x + tf.constant([4.6392727, 5.350893],
                                           dtype=self.dtype)),
      1.1110494) - 1.81915
    self.assertEqual(expected_fx, fn_distribution(x))

  def test_grads(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()
    x = tf.constant([1.0, 1.0], dtype=self.dtype)

    with tf.GradientTape() as tape:
      tape.watch(x)
      expected_fx = tf.multiply(
        self.functions[1][1](x + tf.constant([3.3239317, 4.938116],
                                             dtype=self.dtype)),
        0.8840528) + 1.7433386

    expected_grads = tape.gradient(expected_fx, x)
    grads, fx = fn_distribution.grads_at(x)
    self.assertEqual(expected_fx, fx)
    self.assertAllEqual(expected_grads, grads)

  def test_domain(self):
    fn_distribution = UniformFunctionDistribution(
      functions=self.functions,
      dims_params=self.dims,
      rng_seed=self.seed,
      rng_alg=self.alg,
      vshift_bounds=self.vshift_bounds,
      hshift_bounds=self.hshift_bounds,
      scale_bounds=self.scale_bounds,
      dtype=self.dtype)
    fn_distribution.enable_tf_function()

    fn_distribution.sample()
    domain = tf.constant(self.functions[0][0].domain, dtype=self.dtype)
    self.assertAllEqual(domain, fn_distribution.current_domain)

    fn_distribution.sample()
    domain = tf.constant(self.functions[2][0].domain, dtype=self.dtype)
    self.assertAllEqual(domain, fn_distribution.current_domain)

