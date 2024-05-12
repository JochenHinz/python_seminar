import numpy as np
from numba import njit, prange

import time

_ = np.newaxis


@njit(cache=True, parallel=True, fastmath=True)
def _integrate_numba(f, nelems, weights, points):
  """
    We have to pass weights and points because
    np.polynomial.legendre.leggaus is not available
    inside of a Numba function.
    We can however pass Numba jitted functions `f` as arguments.
  """
  ret = np.empty((4,), dtype=np.float64)
  batch_size = nelems // 4
  for i in prange(4):
    start = i * batch_size
    end = (i + 1) * batch_size if i != 3 else nelems

    val = 0.0
    for j in range(start, end):
      a, b = j / nelems, (j+1) / nelems

      for k in range(len(weights)):
        val += (b - a) * weights[k] * f((b - a) * points[k] + a)

    ret[i] = val

  return ret.sum()


@njit(cache=True)
def _sin(x):
 return np.sin(18 * np.pi * x)


def integrate_numba(f, nelems, order):
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  weights = weights / 2
  return _integrate_numba(f, nelems, weights, points)


print(integrate_numba(_sin, 101, 3))

# Run the numba function for several input element sizes
for power in (2, 3, 5, 6, 7, 8, 9):
  nelems = 10 ** power
  t0 = time.time()
  integrate_numba(_sin, nelems, 3)
  t1 = time.time()
  print(f"The vectorised operation with {nelems} elements took {t1 - t0} seconds.\n")