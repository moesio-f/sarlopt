"""Random function transformation module."""
import typing

import numpy as np

from optfuncs import tensorflow_functions as tff
from optfuncs import transformations_tensorflow as t_tff


def random_shifted_functions(src_fn: tff.TensorflowFunction,
                             n: int,
                             vshift_bounds: typing.Tuple[float, float],
                             hshift_bounds: typing.Tuple[float, float]) -> \
      typing.List[tff.TensorflowFunction]:
  rng = np.random.default_rng()
  vshifts = rng.uniform(vshift_bounds[0],
                        vshift_bounds[1], n).astype(np.float32)
  hshifts = rng.uniform(hshift_bounds[0],
                        hshift_bounds[1], n).astype(np.float32)

  def transformed_fn(v: float, h: float):
    nonlocal src_fn
    return t_tff.VerticalShift(t_tff.HorizontalShift(src_fn, h), v)

  return [transformed_fn(v, h) for v, h in zip(vshifts, hshifts)]
