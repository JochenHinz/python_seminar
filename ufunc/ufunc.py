from typing import Sequence, Tuple, Callable, Any
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


def _vectorize_numpy_operation(op: Callable, subarrays: Sequence[np.ndarray], *args: Sequence[Any], **kwargs):
  return np.stack([ op(*subarrs, *args, **kwargs) for subarrs in zip(*subarrays, strict=True) ], axis=0)


def frozen(arr, dtype=float):
  arr = np.asarray(arr, dtype=dtype)
  arr.flags.writeable = False
  return arr


class ArrayInterpolation(NDArrayOperatorsMixin):
  "First order for now. FIXME: implement higher order."

  def __init__(self, xi_values: np.ndarray | Sequence, x_values: np.ndarray | Sequence) -> None:
    self.xi_values = frozen(xi_values)  # shape (n,)
    self.x_values = frozen(x_values)  # shape (n, n0, n1, ...)
    assert self.x_values.shape[:1] == self.xi_values.shape
    assert (np.diff(self.xi_values) > 0).all(), "Expected monotone increasing xi-values."

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.x_values.shape[1:]

  def __call__(self, xi: float|np.ndarray) -> np.ndarray:
    """ This function returns the linear data point interpolation at the xi values. """
    xi = np.clip(xi, self.xi_values[0], self.xi_values[-1])
    positions = np.clip(np.searchsorted(self.xi_values, xi), 0, len(self.xi_values) - 2)
    xi0, xi1 = self.xi_values[[positions, positions+1]]
    scale = ((xi - xi0) / (xi1 - xi0)).reshape((-1,) + (1,)*(self.x_values.ndim - 1))
    return (1 - scale) * self.x_values[positions] + scale * self.x_values[positions+1]

  def __getitem__(self, index: tuple|int|slice|np.ndarray) -> 'ArrayInterpolation':
    if not isinstance(index, tuple):
      index = (index,)
    index = (slice(None),) + index
    return ArrayInterpolation(self.xi_values, self.x_values[index])

  def reshape(self, *args, **kwargs) -> 'ArrayInterpolation':
    new_x_values = _vectorize_numpy_operation(np.reshape, [self.x_values], *args, **kwargs)
    return ArrayInterpolation(self.xi_values, new_x_values)

  def sum(self, *args, **kwargs):  # sum has to be delegated to np.add.reduce despite the Mixin
    return np.add.reduce(self, *args, **kwargs)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> 'ArrayInterpolation':
    myclass, notmyclass = [], []  # split into ArrayInterpolation and non-ArrayInterpolation
    for inp in inputs:
      (myclass if isinstance(inp, ArrayInterpolation) else notmyclass).append(inp)

    # get all unique xi values of all ArrayInterpolation's passed
    xi_values = np.unique(np.concatenate([interp.xi_values for interp in myclass]))

    # evaluate the ArrayInterpolation's in all unique xi values
    add_arrays = [x(xi_values) for x in myclass]

    # apply the ufunc to each array[i] for all array's in add_arrays. FIXME: avoid the outer for loop
    x_values = _vectorize_numpy_operation(getattr(ufunc, method), add_arrays, *notmyclass, **kwargs)

    return ArrayInterpolation(xi_values, x_values)


if __name__ == '__main__':

  # we have the entire numpy machinery at our disposal but this time
  # if arr.x_values.shape == (n, n0, n1, ...) we apply the numpy operation
  # n times to an array of shape (n0, n1, ...)

  xi0 = np.linspace(0, 1, 6)
  x0 = np.random.randn(6, 3)
  arr0 = ArrayInterpolation(xi0, x0)

  print(f"arr0.x_values has shape: {arr0.x_values.shape}.\n")
  print(f"arr0.sum(0).x_values has shape: {arr0.sum(0).x_values.shape}.\n")

  five_arr0 = 5 * arr0

  xi = np.linspace(0, 1, 3)
  print(f'arr0 evaluated in xi:\n {arr0(xi)}.\n')
  print(f'5 * arr0 evaluated in xi:\n {five_arr0(xi)}.\n')
  print(f'arr0 + 5 in xi: \n {(arr0 + 5)(xi)}.\n\n')

  xi1 = np.linspace(0, 1, 11)
  x1 = np.random.randn(11, 3)
  arr1 = ArrayInterpolation(xi1, x1)

  print(f'arr0 + arr1 in xi:\n {(arr0 + arr1)(xi)}.\n')
  print(f'arr0[:, np.newaxis] + arr1[np.newaxis, :] in xi:\n {(arr0[:, np.newaxis] + arr1[np.newaxis, :])(xi)}.\n\n')