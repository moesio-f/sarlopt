"""Distributions of TensorflowFunctions."""
import abc
import typing

import tensorflow as tf

from optfuncs import tensorflow_functions as tff

FunctionList = typing.List[tff.TensorflowFunction]
ParamBound = typing.Tuple[float, float]


class FunctionDistribution(abc.ABC):
  @abc.abstractmethod
  def sample(self) -> None:
    pass

  @abc.abstractmethod
  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    pass


class UniformFunctionDistribution(FunctionDistribution):
  def __init__(self,
               functions: typing.List[FunctionList],
               vshift_bounds: ParamBound,
               hshift_bounds: ParamBound,
               scale_bounds: ParamBound,
               rng_seed,
               rng_alg=tf.random.Algorithm.PHILOX,
               dims_params: int = 1,
               dtype: tf.DType = tf.float32):
    self._seed = rng_seed
    self._alg = rng_alg
    self._rng = tf.random.Generator.from_seed(self._seed, self._alg)
    self._dtype = dtype

    self._dims_params = tf.constant(dims_params, dtype=tf.int32)

    vshift_min, vshift_max = vshift_bounds
    self._vshift_min = tf.constant(vshift_min, dtype=dtype)
    self._vshift_max = tf.constant(vshift_max, dtype=dtype)

    hshift_min, hshift_max = hshift_bounds
    self._hshift_min = tf.constant(hshift_min, dtype=dtype)
    self._hshift_max = tf.constant(hshift_max, dtype=dtype)

    scale_min, scale_max = scale_bounds
    self._scale_min = tf.constant(scale_min, dtype=dtype)
    self._scale_max = tf.constant(scale_max, dtype=dtype)

    self._fns = tf.nest.flatten(functions)

    self._fn_evaluator = lambda x: tf.nest.map_structure(
      lambda f: lambda: f(x),
      self._fns)

    self._domains = tf.constant([(f.domain.min, f.domain.max) for f in
                                 self._fns],
                                dtype=dtype)
    self._len_classes = tf.constant([len(fn_l) for fn_l in functions],
                                    dtype=tf.int32)
    self._cum_sum_len_classes = tf.cumsum(self._len_classes)
    self._n_classes = tf.constant(self._len_classes.shape[-1], dtype=tf.int32)

    def mapper(cls_index, fn_index):
      start_flat_index = tf.cond(tf.greater(cls_index, 0),
                                 lambda: tf.gather(self._cum_sum_len_classes,
                                                   cls_index - 1),
                                 lambda: 0)
      return fn_index + start_flat_index

    self._flat_indice_mapper = mapper

    self._class_index = tf.Variable(name='class_index',
                                    initial_value=0,
                                    trainable=False,
                                    dtype=tf.int32)
    self._fn_index = tf.Variable(name='fn_index',
                                 initial_value=0,
                                 trainable=False,
                                 dtype=tf.int32)

    self._hshift = tf.Variable(name='hshift',
                               initial_value=tf.zeros(
                                 shape=(self._dims_params,),
                                 dtype=dtype),
                               trainable=False,
                               dtype=dtype)
    self._vshift = tf.Variable(name='vshift',
                               initial_value=0,
                               trainable=False,
                               dtype=dtype)
    self._scale = tf.Variable(name='scale',
                              initial_value=0,
                              trainable=False,
                              dtype=dtype)

    self._fn = self._call
    self.sample()

  def _call(self, x: tf.Tensor) -> tf.Tensor:
    hshift = self._hshift.value()
    vshift = self._vshift.value()
    scale = self._scale.value()
    cls_index = self._class_index.value()
    fn_index = self._fn_index.value()

    with tf.control_dependencies([cls_index, fn_index]):
      index = self._flat_indice_mapper(cls_index, fn_index)
      with tf.control_dependencies([index, hshift]):
        branches = self._fn_evaluator(x + hshift)

    with tf.control_dependencies([branches]):
      fx = tf.switch_case(index, branches)

    with tf.control_dependencies([fx, vshift, scale]):
      return tf.multiply(fx, scale) + vshift

  def _sample_update(self,
                     cls_index: tf.Tensor,
                     fn_index: tf.Tensor,
                     hshift: tf.Tensor,
                     vshift: tf.Tensor,
                     scale: tf.Tensor):
    self._hshift.assign(value=hshift)
    self._vshift.assign(value=vshift)
    self._scale.assign(value=scale)
    self._class_index.assign(value=cls_index)
    self._fn_index.assign(fn_index)

  def sample(self) -> None:
    random_hshift = self._rng.uniform(
      shape=(self._dims_params,),
      minval=self._hshift_min,
      maxval=self._hshift_max,
      dtype=self._dtype)
    random_vshift = self._rng.uniform(
      shape=(),
      minval=self._vshift_min,
      maxval=self._vshift_max,
      dtype=self._dtype)
    random_scale = self._rng.uniform(
      shape=(),
      minval=self._scale_min,
      maxval=self._scale_max,
      dtype=self._dtype)
    random_cls_index = self._rng.uniform(
      shape=(),
      minval=0,
      maxval=self._n_classes,
      dtype=tf.int32)

    with tf.control_dependencies([random_cls_index]):
      random_fn_index = self._rng.uniform(
        shape=(),
        minval=0,
        maxval=tf.gather(self._len_classes, random_cls_index),
        dtype=tf.int32)

    with tf.control_dependencies([random_hshift,
                                  random_vshift,
                                  random_scale,
                                  random_cls_index,
                                  random_fn_index]):
      self._sample_update(hshift=random_hshift,
                          vshift=random_vshift,
                          scale=random_scale,
                          cls_index=random_cls_index,
                          fn_index=random_fn_index)

  def sample_from_class(self, cls_index: tf.Tensor) -> None:
    random_hshift = self._rng.uniform(
      shape=(self._dims_params,),
      minval=self._hshift_min,
      maxval=self._hshift_max,
      dtype=self._dtype)
    random_vshift = self._rng.uniform(
      shape=(),
      minval=self._vshift_min,
      maxval=self._vshift_max,
      dtype=self._dtype)
    random_scale = self._rng.uniform(
      shape=(),
      minval=self._scale_min,
      maxval=self._scale_max,
      dtype=self._dtype)
    random_fn_index = self._rng.uniform(
      shape=(),
      minval=0,
      maxval=tf.gather(self._len_classes, cls_index),
      dtype=tf.int32)
    with tf.control_dependencies([random_hshift,
                                  random_vshift,
                                  random_scale,
                                  random_fn_index]):
      self._sample_update(hshift=random_hshift,
                          vshift=random_vshift,
                          scale=random_scale,
                          cls_index=cls_index,
                          fn_index=random_fn_index)

  def sample_from_function(self,
                           cls_index: tf.Tensor,
                           fn_index: tf.Tensor) -> None:
    random_hshift = self._rng.uniform(
      shape=(self._dims_params,),
      minval=self._hshift_min,
      maxval=self._hshift_max,
      dtype=self._dtype)
    random_vshift = self._rng.uniform(
      shape=(),
      minval=self._vshift_min,
      maxval=self._vshift_max,
      dtype=self._dtype)
    random_scale = self._rng.uniform(
      shape=(),
      minval=self._scale_min,
      maxval=self._scale_max,
      dtype=self._dtype)
    with tf.control_dependencies([random_hshift,
                                  random_vshift,
                                  random_scale]):
      self._sample_update(hshift=random_hshift,
                          vshift=random_vshift,
                          scale=random_scale,
                          cls_index=cls_index,
                          fn_index=fn_index)

  @tf.function
  def grads(self, x: tf.Tensor) -> tf.Tensor:
    grads, _ = self.grads_at(x)
    return grads

  @tf.function
  def grads_at(self, x: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = self(x)

    return tape.gradient(y, x), y

  def enable_tf_function(self):
    self._fn = tf.function(self._fn)

  def disable_tf_function(self):
    self._fn = self._call

  @property
  def current_domain(self) -> tf.Tensor:
    cls_index = self._class_index.value()
    fn_index = self._fn_index.value()

    with tf.control_dependencies([cls_index, fn_index]):
      index = self._flat_indice_mapper(cls_index, fn_index)
      with tf.control_dependencies([index]):
        return tf.gather(self._domains, index)

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    return self._fn(x)
