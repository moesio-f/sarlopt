"""Random function transformation module."""

import typing

import numpy as np
from py_benchmark_functions.imp import tensorflow as tff


def random_shifted_functions(
    src_fn: tff.TensorflowFunction,
    n: int,
    vshift_bounds: typing.Tuple[float, float],
    hshift_bounds: typing.Tuple[float, float],
    seed,
) -> typing.List[tff.TensorflowFunction]:
    rng = np.random.default_rng(seed=seed)
    vshifts = rng.uniform(vshift_bounds[0], vshift_bounds[1], n).astype(np.float32)
    hshifts = rng.uniform(hshift_bounds[0], hshift_bounds[1], n).astype(np.float32)

    def transformed_fn(v: float, h: float):
        return tff.TensorflowTransformation(src_fn, vshift=v, hshift=h)

    return [transformed_fn(v, h) for v, h in zip(vshifts, hshifts)]
