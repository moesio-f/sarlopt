"""Ambiente utilizado para a minimização de funções matemáticas com RL."""

import abc
import typing

import numpy as np
import tensorflow as tf
from numpy.random import default_rng
from tf_agents.environments import py_environment, PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from optfuncs import core
from src.typing import types as custom_types


class AbstractPyFunctionEnv(py_environment.PyEnvironment, abc.ABC):
  @abc.abstractmethod
  def current_position(self) -> types.TensorOrArray:
    """Retorna a posição atual do agente na função. """

  @abc.abstractmethod
  def get_function(self) -> custom_types.FunctionOrListFunctions:
    """Retorna a(s) função(ões) associadas com esse ambiente. """

  @abc.abstractmethod
  def get_current_function(self) -> core.Function:
    """Retorna a função associada com o ambiente no momento da chamada. """


class PyFunctionEnv(AbstractPyFunctionEnv):
  """Ambiente para a minimização de função.
  Dada uma função f: D -> I, onde D é um subconjunto de R^d
  e I é um subconjunto de R, as especificações do ambiente são:
    as observações (s em D) são posições do domínio;
    as ações (a em R^d) são os possíveis passos;
    as recompensas são r = -f(s + a).
  """

  # Quantidade máxima de iterações entre o agente e ambiente.
  MAX_STEPS = 50000

  def get_info(self) -> types.NestedArray:
    return self._state

  def __init__(self, function: core.Function, dims,
               bounded_actions_spec: bool = True):
    super().__init__()
    self._rng = default_rng()
    self.func = function
    self._dims = dims

    self._action_spec = array_spec.BoundedArraySpec(shape=(self._dims,),
                                                    dtype=np.float32,
                                                    minimum=-1.0,
                                                    maximum=1.0,
                                                    name='action')
    if not bounded_actions_spec:
      self._action_spec = array_spec.ArraySpec.from_spec(self._action_spec)

    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(self._dims,),
      dtype=np.float32,
      minimum=function.domain.min,
      maximum=function.domain.max,
      name='observation')

    self._episode_ended = False
    self._steps_taken = 0

    self._state = self.__initial_state()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_state(self):
    state = (self._state, self._steps_taken, self._episode_ended)
    return state

  def set_state(self, state):
    _state, _steps_taken, _episode_ended = state
    self._state = _state
    self._steps_taken = _steps_taken
    self._episode_ended = _episode_ended

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    self._state = self._state + action
    domain_min, domain_max = self.func.domain
    self._state = np.clip(self._state, domain_min, domain_max)

    self._steps_taken += 1
    if self._steps_taken >= PyFunctionEnv.MAX_STEPS:
      self._episode_ended = True

    obj_value = self.func(self._state)
    reward = -obj_value

    if self._episode_ended:
      return ts.termination(self._state, reward)

    return ts.transition(self._state, reward)

  def _reset(self):
    self._state = self.__initial_state()
    self._episode_ended = False
    self._steps_taken = 0
    return ts.restart(self._state)

  def render(self, mode: str = 'human'):
    raise NotImplementedError("Not Implemented yet.")

  def __initial_state(self) -> np.ndarray:
    domain_min, domain_max = self.func.domain
    state = self._rng.uniform(size=(self._dims,),
                              low=domain_min,
                              high=domain_max)
    return state.astype(dtype=np.float32, copy=False)

  def current_position(self) -> types.TensorOrArray:
    return self._state

  def get_function(self) -> custom_types.FunctionOrListFunctions:
    return self.func

  def get_current_function(self) -> core.Function:
    return self.func


class PyFunctionEnvV1(AbstractPyFunctionEnv):
  """Ambiente utilizada para a minimização de funções.
  Essa versão (V1) considera o cenário de observação parcial, assim o agente
  não possui acesso direto aos estados Markovianos do ambiente.

  Seja f: D -> I um função, onde D é um subconjunto de R^d
  e I é um subconjunto de R, as especificações do ambiente são descritas da
  seguinte forma:
      - Conjunto de estados (S): Descrevem a função por completo (posição do
          mínimo global, convexa, monótoma, posição do agente no
          espaço de busca, gradientes em todos pontos, valores objetivos).
          O estado é formado pelo estado observável e estado escondido.
          O estado escondido é estacionário ao longo do episódio.
            (*) Estado observável (s^o): posição atual, gradientes na posição
              atual, valor objetivo na posição atual, melhor posição encontrada,
              melhor valor objetivo encontrado.
            (*) Estado escondido (s^h): Mínimo global (
              se conhecido, argmin f), gradientes em todos os pontos (se existe),
              valor de mínimo global (min f).
          Dessa forma, o estado pode ser representado pela tupla:
            s = (x, grad(f(x)), f(x), bx, bf(x), argmin f, min f, grads(f)),
            onde
              s^o = (x, grad(f(x)), f(x), bx, bf(x)) é o estado observável
              s^h = (argmin f, min f, grads(f)) é o estado escondido.
          Assim, S = D x D x R x D x R x D x R x (D x D)
      - Conjunto de observações (O): (f(x), grad(f(x)), dx, df)
          Obtidos dos estados por meio da função de emissão.
      - Conjunto de ações (A): Dx, step-vector.
      - Função de transição: Desconhecida (Model-Free).
      - Distribuição probabilística dos estados iniciais: Uniforme sobre o conj-
          unto de estados.
      - Função Recompensa:
          R(s^o_t, a_t, s^o_{t+1}, s^h) = -f(x_{t+1})
  """
  # Quantidade máxima de interações agente-ambiente.
  MAX_STEPS: int = 2000
  # String's para as chaves das observações (dictionary)
  GRADIENT_STR: str = "gradient"
  OBJECTIVE_VALUE_STR: str = "objective_value"
  OBJECTIVE_VALUE_DELTA_STR: str = "objective_value_delta"
  POSITION_DELTA_STR: str = "position_delta"

  def __init__(self,
               function: core.Function,
               dims: int,
               grad_fun: custom_types.GradientCallable,
               argmin: types.Optional[types.Array] = None,
               globalmin: types.Optional[types.Float] = None,
               action_min: types.Float = -1.0,
               action_max: types.Float = 1.0,
               float_dtype: types.TypeVar = np.float32,
               seed=None):
    super().__init__()
    # Gerador aleatório.
    self._seed = seed
    if self._seed is None:
      self._seed = np.random.SeedSequence().generate_state(1)
    self._rng = default_rng(seed=self._seed)

    # Informações da função.
    self._fn: core.Function = function
    self._dims: int = dims
    self._float_dtype: types.TypeVar = float_dtype

    # Variáveis de controle
    self._episode_ended = False
    self._steps_taken = 0

    # Estado escondido (descrição da função).
    self._fn_gradients = grad_fun
    self._argmin = argmin
    self._globalmin = globalmin

    # Função recompensa
    self._fn_reward = lambda x_: -self._fn(x_)

    # Estado observável
    self._position: np.ndarray = self._rng.uniform(
      size=(self._dims,),
      low=self._fn.domain.min,
      high=self._fn.domain.max)
    self._best_position: np.ndarray = self._position
    self._objective_value: np.ndarray = np.array([self._fn(self._position)],
                                                 dtype=self._float_dtype)
    self._best_value: np.ndarray = self._objective_value
    self._gradient: np.ndarray = self._fn_gradients(self._position)

    # Specs
    self._action_spec = array_spec.BoundedArraySpec(shape=(self._dims,),
                                                    dtype=self._float_dtype,
                                                    minimum=action_min,
                                                    maximum=action_max,
                                                    name='action')
    self._observation_spec = {
      PyFunctionEnvV1.GRADIENT_STR: array_spec.ArraySpec(
        shape=(self._dims,),
        dtype=self._float_dtype),
      PyFunctionEnvV1.OBJECTIVE_VALUE_STR: array_spec.ArraySpec(
        shape=(1,),
        dtype=self._float_dtype),
      PyFunctionEnvV1.OBJECTIVE_VALUE_DELTA_STR: array_spec.ArraySpec(
        shape=(1,),
        dtype=self._float_dtype),
      PyFunctionEnvV1.POSITION_DELTA_STR: array_spec.ArraySpec(
        shape=(self._dims,),
        dtype=self._float_dtype),
    }

  def _step(self, action):
    if self._episode_ended:
      return self.reset()

    self._steps_taken += 1

    new_pos = np.clip(self._position + action,
                      a_min=self._fn.domain.min,
                      a_max=self._fn.domain.max).astype(self._float_dtype)
    new_obj_value = np.array([self._fn(new_pos)],
                             dtype=self._float_dtype)
    new_grad = self._fn_gradients(new_pos)
    reward = self._fn_reward(new_pos)

    obs = self.build_observation(
      x=self._position,
      x_=new_pos,
      f=self._objective_value,
      f_=new_obj_value,
      grad=new_grad)

    self._position = new_pos
    self._objective_value = new_obj_value
    self._gradient = new_grad

    if np.all(np.less(self._objective_value, self._best_value)):
      self._best_value = self._objective_value
      self._best_position = self._position

    self._episode_ended = self._steps_taken >= PyFunctionEnv.MAX_STEPS
    if self._episode_ended:
      return ts.termination(obs, reward)

    return ts.transition(obs, reward)

  def _reset(self):
    print('Best value: {0} | Best position: {1}'.format(self._best_value,
                                                        self._best_position))
    self._position = self._rng.uniform(
      size=(self._dims,),
      low=self._fn.domain.min,
      high=self._fn.domain.max)
    self._best_position: np.ndarray = self._position
    self._objective_value: np.ndarray = np.array([self._fn(self._position)],
                                                 dtype=self._float_dtype)
    self._best_value = self._objective_value
    self._gradient = self._fn_gradients(self._position)
    self._episode_ended = False
    self._steps_taken = 0

    obs = self.build_observation(f=self._objective_value,
                                 f_=self._objective_value,
                                 x=self._position,
                                 x_=self._position,
                                 grad=self._gradient)

    return ts.restart(obs)

  def build_observation(self,
                        f: types.Array,
                        f_: types.Array,
                        x: types.Array,
                        x_: types.Array,
                        grad: types.Array) -> typing.Dict:
    dx = (x_ - x).astype(self._float_dtype)
    df = (f_ - f).astype(self._float_dtype)
    return tf.nest.pack_sequence_as(structure=self.observation_spec(),
                                    flat_sequence=[grad, f_, df, dx])

  def get_info(self) -> types.NestedArray:
    pass

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_state(self):
    pass

  def set_state(self, state):
    pass

  def render(self, mode: str = 'human'):
    raise NotImplementedError("Not Implemented yet.")

  def current_position(self) -> types.TensorOrArray:
    return self._position

  def get_function(self) -> custom_types.FunctionOrListFunctions:
    return self._fn

  def get_current_function(self) -> core.Function:
    return self._fn
