import numpy as np
from matplotlib import pyplot as plt
from typing import Sequence, Tuple, Union, Any


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


# I am a template for other differentiable functions
class DifferentiableFunction:

  def __init__(self, args: Sequence[Any]) -> None:
    # Store all arguments that characterise the class. This will come in handy later.
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
    """ Plot function over the interval `interval` using `npoints` function evaluations. """
    a, b = interval
    assert b > a
    x = np.linspace(*interval, 1001)
    y = [self(_x) for _x in x]
    plt.plot(x, y)
    plt.show()

  def __add__(self, other: FunctionType) -> 'Add':
    "self: DifferentialFunction + other: FunctionType"
    return Add(self, other)

  # if we invoke other: NumericType + self: DifferentiableFunction,
  # python will not know how to add `self` to `other`. Then python will check if
  # `self` implements __radd__.
  # It does ! and we simply return self + other (addition is commutative).
  # In this case `other` is coerced to Constant(other) in Add.__init__.
  __radd__ = __add__

  def __mul__(self, other: FunctionType) -> 'Multiply':
    "self: DifferentialFunction * other: FunctionType"
    return Multiply(self, other)

  # idem with other: NumericType * self: DifferentiableFunction
  __rmul__ = __mul__

  def __sub__(self, other: FunctionType) -> 'Add':
    "self: DifferentialFunction - other: FunctionType"
    return self + (-1) * other

  def __rsub__(self, other: NumericType) -> 'Add':
    """
      other: NumericType - self: DifferentialFunction.
      Here, the -1 has to go in front of self.
      other - self => self.__rsub__(other).
    """
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


# FIXME: Lots of boilerplate from here on out.


class Exp(DifferentiableFunction):

  def _deriv(self) -> Multiply:
    return self * self.argument.derivative()

  def __init__(self, argument: FunctionType) -> None:
    self.argument = as_function(argument)
    super().__init__([self.argument])

  def __call__(self, x: NumericType) -> float:
    return np.exp(self.argument(x))


class Sin(DifferentiableFunction):

  def _deriv(self) -> Multiply:
    return Cos(self.argument) * self.argument.derivative()

  def __init__(self, argument: FunctionType) -> None:
    self.argument = as_function(argument)
    super().__init__([self.argument])

  def __call__(self, x: NumericType) -> float:
    return np.sin(self.argument(x))


class Cos(DifferentiableFunction):

  def _deriv(self) -> Multiply:
    return (-1) * Sin(self.argument) * self.argument.derivative()

  def __init__(self, argument: FunctionType) -> None:
    self.argument = as_function(argument)
    super().__init__([self.argument])

  def __call__(self, x: NumericType) -> float:
    return np.cos(self.argument(x))


#


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