""" [IMPORT] """
import numpy as np
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union, Any, Callable
from abc import abstractmethod
from collections.abc import Hashable

from functools import lru_cache


""" [HELPER] """
NumericType = Union[int, float]
FunctionType = Union['DifferentiableFunction', NumericType]


# main function for type coercion
def as_function(func: FunctionType) -> 'DifferentiableFunction':
  if isinstance(func, DifferentiableFunction):
    return func
  return Constant(func)


""" [NEW] - remember the derivative of a function - called in `_derivative` """
@lru_cache
def _derivative(self: 'DifferentiableFunction') -> 'DifferentiableFunction':
  """
  We will never again compute a derivative twice ;-)
  """
  return self._deriv()


# inheriting from `Hashable` automatically inherits that metaclass=ABCMeta
class DifferentiableFunction(Hashable):

  def __init__(self, args: Sequence[Hashable]) -> None:
    self._args = tuple(args)
    self._hash = hash(self._args)

  def __hash__(self) -> int:
    return self._hash

  def __eq__(self, other: Any) -> bool:
    return self.__class__ == other.__class__ and hash(self) == hash(other) and self._args == other._args

  @abstractmethod
  def _deriv(self):
    pass

  @abstractmethod
  def __call__(self, x: NumericType):
    pass

  """ [NEW] - called `cached` first order derivative repeatedly """
  def derivative(self, n: int = 1) -> 'DifferentiableFunction':
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    # use the cached version.
    return _derivative(self).derivative(n=n-1)

  def plot(self, interval: Tuple[int, int] = (0, 1), npoints: int = 1001) -> None:
    """ Plot function over the interval `interval` using `npoints` function evaluations. """
    a, b = interval
    assert b > a
    x = np.linspace(*interval, 1001)
    y = [self(_x) for _x in x]
    plt.plot(x, y)
    plt.show()

  def __add__(self, other: FunctionType) -> 'Add':
    return Add(self, other)

  __radd__ = __add__

  def __mul__(self, other: FunctionType) -> 'Multiply':
    return Multiply(self, other)

  __rmul__ = __mul__

  def __sub__(self, other: FunctionType) -> 'Add':
    return self + (-1) * other

  def __rsub__(self, other: NumericType) -> 'Add':
    return other + (-1) * self


class Constant(DifferentiableFunction):

  def __init__(self, value: NumericType) -> None:
    self.value = float(value)
    super().__init__([self.value])

  def _deriv(self) -> 'Constant':
    return Constant(0)

  def __call__(self, x: NumericType) -> float:
    return self.value


class Argument(DifferentiableFunction):

  def __init__(self) -> None:
    super().__init__([])

  def _deriv(self) -> Constant:
    return Constant(1)

  def __call__(self, x: NumericType) -> float:
    return float(x)


class Add(DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0, self.f1 = sorted(map(as_function, (f0, f1)), key=lambda x: (x.__class__.__name__, hash(x)))
    super().__init__([self.f0, self.f1])

  def _deriv(self) -> 'Add':
    return self.f0.derivative() + self.f1.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) + self.f1(x)


class Multiply(DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0, self.f1 = sorted(map(as_function, (f0, f1)), key=lambda x: (x.__class__.__name__, hash(x)))
    super().__init__([self.f0, self.f1])

  def _deriv(self) -> Add:
    return self.f0.derivative() * self.f1 + self.f0 * self.f1.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) * self.f1(x)


class ChainRule(DifferentiableFunction):

  evalf: Callable
  df: Callable

  def __init__(self, argument: FunctionType) -> None:
    assert all(hasattr(self, item) for item in ('evalf', 'df')), 'Each derived class needs to implement `evalf` and `df`.'
    self.argument = as_function(argument)
    super().__init__([self.argument])

  def _deriv(self) -> DifferentiableFunction:
    return self.df(self.argument) * self.argument.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.evalf(self.argument(x))


class Exp(ChainRule):
  evalf = np.exp
  df = lambda self, argument: self


class Sin(ChainRule):
  evalf = np.sin
  df = lambda self, argument: Cos(argument)


class Cos(ChainRule):
  evalf = np.cos
  df = lambda self, argument: (-1) * Sin(argument)


""" [NEW] - We define a class that passes a Sequence of non-hashable types as `args`
            to `DifferentiableFunction.__init__(self, args)`.
            Since we annotated `args` as a `Sequence[Hashable]`, static type
            checking will complain. """
class BrokenClass(DifferentiableFunction):

  """ [FOCUS] - pass a non-hashable list [1, 2, 3] as sole argument to super().__init__ """
  def __init__(self) -> None:
    super().__init__( [[1, 2, 3]] )

  def _deriv(self) -> DifferentiableFunction:
    return Constant(0)

  def __call__(self, x: float) -> float:
    return 0.0


""" [HELPER] - solve PDE as before """
def test():
  """
    The differential equation:

    y = y(x):

    25 * y + y' + y'' = 0,

    has the general solution:

    y(x) =   c0 * exp(-x/2) * sin(3 * sqrt(11) / 2 * x)
           + c1 * exp(-x/2) * cos(3 * sqrt(11) / 2 * x)
  """

  # Total time interval
  T = 10

  # make an argument f(x) = x
  x = Argument()

  # choose some c0, c1
  c0, c1 = 2, 1

  # make the damping term
  exp = Exp(-.5 * x)

  # define the natural frequency
  w0 = 3 * np.sqrt(11) / 2

  # create y(x) using syntactic sugar
  y = c0 * exp * Sin(w0 * x) + c1 * exp * Cos(w0 * x)

  # set plot interval
  interval = (0, T)

  # plot y:
  y.plot(interval=interval)

  # If we plot the below, what should we get ?
  (25 * y + y.derivative() + y.derivative(2)).plot(interval=interval)


if __name__ == '__main__':
  test()
