import numpy as np
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union, Any, Callable


NumericType = Union[int, float]
FunctionType = Union['DifferentiableFunction', NumericType]


# main function for type coercion
def as_function(func: FunctionType) -> 'DifferentiableFunction':
  """
    func is a DifferentialFunction => return func,
    func is an int or float => return Constant(func),
    func is anything else => this method fails.
  """
  if isinstance(func, DifferentiableFunction):
    return func
  return Constant(func)


class DifferentiableFunction:

  def __init__(self, args: Sequence[Any]) -> None:
    self._args = tuple(args)

  def __call__(self, x: NumericType):
    raise NotImplementedError("Each derived class needs to implement its call behaviour.")

  def _deriv(self):
    raise NotImplementedError("Each derived class needs to implement its derivative.")

  def derivative(self, n: int = 1) -> 'DifferentiableFunction':
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._deriv().derivative(n=n-1)

  def plot(self, interval: Tuple[int, int] = (0, 1), npoints: int = 1001) -> None:
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

  def _deriv(self) -> 'Constant':
    return Constant(0)

  def __init__(self, value: NumericType) -> None:
    self.value = float(value)
    super().__init__([self.value])

  def __call__(self, x: NumericType) -> float:
    return self.value


class Argument(DifferentiableFunction):

  def _deriv(self) -> Constant:
    return Constant(1)

  def __init__(self) -> None:
    super().__init__([])

  def __call__(self, x: NumericType) -> float:
    return float(x)


class Add(DifferentiableFunction):

  def _deriv(self) -> 'Add':
    return self.f0.derivative() + self.f1.derivative()

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0 = as_function(f0)
    self.f1 = as_function(f1)
    super().__init__([self.f0, self.f1])

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) + self.f1(x)


class Multiply(DifferentiableFunction):

  def _deriv(self) -> Add:
    return self.f0.derivative() * self.f1 + self.f0 * self.f1.derivative()

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0 = as_function(f0)
    self.f1 = as_function(f1)
    super().__init__([self.f0, self.f1])

  def __call__(self, x: NumericType) -> float:
    return self.f0(x) * self.f1(x)


class ChainRule(DifferentiableFunction):
  # All functions that are subject to the chain rule: d(f(g)) = df(g) * dg
  pass


class Exp(ChainRule):
  pass


class Sin(ChainRule):
  pass


class Cos(ChainRule):
  pass


def test():
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