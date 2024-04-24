import numpy as np
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
    self._hash = hash(self._args)  # will fail if an element of `args` is not hashable

  def __hash__(self) -> int:
    return self._hash

  def __eq__(self, other: Any) -> bool:
    # class not the same => False, hash not the same => instance._args not the same => False
    # If other doesn't have a hash, there's no problem because the first check will fail.
    return self.__class__ == other.__class__ and hash(self) == hash(other) and self._args == other._args

  def __add__(self, other: FunctionType) -> 'Add':
    return Add(self, other)

  def __mul__(self, other: FunctionType) -> 'Multiply':
    return Multiply(self, other)


class Constant(DifferentiableFunction):

  def __init__(self, value: NumericType) -> None:
    self.value = float(value)
    super().__init__([self.value])


class Argument(DifferentiableFunction):

  def __init__(self) -> None:
    super().__init__([])


class Add(DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0, self.f1 = sorted(map(as_function, (f0, f1)), key=lambda x: (x.__class__.__name__, hash(x)))
    super().__init__([self.f0, self.f1])


class Multiply(DifferentiableFunction):

  def __init__(self, f0: FunctionType, f1: FunctionType) -> None:
    self.f0, self.f1 = sorted(map(as_function, (f0, f1)), key=lambda x: (x.__class__.__name__, hash(x)))
    super().__init__([self.f0, self.f1])


class ChainRule(DifferentiableFunction):

  evalf: Callable
  df: Callable

  def __init__(self, argument: FunctionType) -> None:
    self.argument = as_function(argument)
    super().__init__([self.argument])


class Exp(ChainRule):
  evalf = np.exp
  df = lambda self, argument: self


class Sin(ChainRule):
  evalf = np.sin
  df = lambda self, argument: Cos(argument)


class Cos(ChainRule):
  evalf = np.cos
  df = lambda self, argument: (-1) * Sin(argument)


def test():
  """
    Test the whether the class instantiations are
    hashable and whether `a == b` acts as expected.
  """

  x = Argument()

  print(f"The hash value of x is {hash(x)}. \n")

  sinx = Sin(x)

  print(f"The hash value of sin(x) is {hash(sinx)}. \n")

  print(f"x == sin(x) is {x == sinx}. \n")
  print(f"sin(x) == sin(x) is {sinx == sinx}. \n")

  try:
    print('Trying to use sin(x) as a key in a dictionary.')
    test = {}
    test[sinx] = 5
    print('Works ! \n')
  except Exception as ex:
    print("Failed with error '{}'.".format(ex))

  apb = x + sinx
  bpa = sinx + x

  print(f"x + sin(x) == sin(x) + x is {apb == bpa} =)")


if __name__ == '__main__':
  test()