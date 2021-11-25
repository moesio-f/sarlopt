"""Common types used."""

import typing

from tf_agents.metrics import tf_metric
from tf_agents.typing import types as tf_types

from optfuncs import core

LayerParam = typing.Union[typing.List, typing.Tuple]
TFMetric = typing.Union[tf_metric.TFStepMetric,
                        tf_metric.TFMultiMetricStepMetric,
                        tf_metric.TFHistogramStepMetric]
FunctionOrListFunctions = typing.Union[core.Function,
                                       typing.List[core.Function]]
GradientCallable = typing.Callable[[tf_types.TensorOrArray],
                                   tf_types.TensorOrArray]
