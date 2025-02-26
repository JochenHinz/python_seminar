""" [IMPORT] - numpy, plotting, type hinting """
import numpy as np
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union, Any, Callable
from abc import ABCMeta, abstractmethod


""" [HELPER] - type aliases, function coercion """
NumericType = Union[int, float]
FunctionType = Union['DifferentiableFunction', NumericType]


def as_function(func: FunctionType) -> 'DifferentiableFunction':
  if isinstance(func, DifferentiableFunction):
    return func
  return Constant(func)


""" [NEW] - Mixin for finding roots. End the class name on `Mixin` """
class FindRootMixin:

  def __init__(self, *args, **kwargs):
    """ Forward all args to next base class """
    super().__init__(*args, **kwargs)

  def find_root(self, x0=0, **scipykwargs):
    from scipy.optimize import root_scalar
    root, = root_scalar(lambda x: self(x[0]), x0=[x0], **scipykwargs).root
    x = np.linspace(root - 1, root + 1, 1001)
    y = [self(_x) for _x in x]
    plt.plot(x, y)
    plt.scatter([root], [self(root)])
    plt.show()


""" [FOCUS] - By default:
    1) We assume that the function can't have roots.
    2) If it can have roots, we overwrite `find_root` using the Mixin from above. """
class DifferentiableFunction(metaclass=ABCMeta):

  def __init__(self, args: Sequence[Any]) -> None:
    self._args = tuple(args)

  @abstractmethod
  def _deriv(self):
    pass

  @abstractmethod
  def __call__(self, x: NumericType):
    pass

  """ [NEW] - overwrite this method in case the function can have roots """
  def find_root(self, *args, **kwargs):
    raise TypeError("Functions of class {} have no roots.".format(self.__class__.__name__))

  def derivative(self, n: int = 1) -> 'DifferentiableFunction':
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._deriv().derivative(n=n-1)

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


# Constant ain't got no root (FIXME: unless the value is 0...)
class Constant(DifferentiableFunction):

  def __init__(self, value: NumericType) -> None:
    self.value = float(value)
    super().__init__([self.value])

  def _deriv(self) -> 'Constant':
    return Constant(0)

  def __call__(self, x: NumericType) -> float:
    return self.value


""" [NEW] - f(x) = x has a root. To overwrite, the Mixin must be inherited from first. """
class Argument(FindRootMixin, DifferentiableFunction):

  def __init__(self) -> None:
    super().__init__([])

  def _deriv(self) -> Constant:
    return Constant(1)

  def __call__(self, x: NumericType) -> float:
    return float(x)


""" [NEW] - the sum of two functions can have roots """
class Add(FindRootMixin, DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0 = as_function(f0)
    self.f1 = as_function(f1)
    super().__init__([self.f0, self.f1])

  def _deriv(self) -> 'Add':
    return self.f0.derivative() + self.f1.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) + self.f1(x)


""" [NEW] - the product of two functions can have roots """
class Multiply(FindRootMixin, DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0 = as_function(f0)
    self.f1 = as_function(f1)
    super().__init__([self.f0, self.f1])

  def _deriv(self) -> Add:
    return self.f0.derivative() * self.f1 + self.f0 * self.f1.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) * self.f1(x)


class ChainRule(DifferentiableFunction):
  # All functions that are subject to the chain rule: d(f(g)) = df(g) * dg

  evalf: Callable
  df: Callable

  def __init__(self, argument: FunctionType) -> None:
    self.argument = as_function(argument)
    super().__init__([self.argument])

  def _deriv(self) -> DifferentiableFunction:
    return self.df(self.argument) * self.argument.derivative()

  def __call__(self, x: NumericType) -> float:
    return self.evalf(self.argument(x))


# Exp ain't got no roots
class Exp(ChainRule):
  evalf = np.exp
  df = lambda self, argument: self


""" [NEW] - sin(f(x)) may have roots """
class Sin(FindRootMixin, ChainRule):
  evalf = np.sin
  df = lambda self, argument: Cos(argument)


""" [NEW] - cos(f(x)) may have roots """
class Cos(FindRootMixin, ChainRule):
  evalf = np.cos
  df = lambda self, argument: (-1) * Sin(argument)


""" [HELPER] - find the root of y(x) = x * sin(x) - 5 """
def test():

  x = Argument()
  y = x * Sin(x) - 5

  y.find_root(x0=2)


if __name__ == '__main__':
  test()
