import numpy as np
from numba import njit, prange

import time

_ = np.newaxis


@njit(cache=True, parallel=True, fastmath=True)
def _integrate_numba_parallel(f, nelems, weights, points):
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


@njit(cache=True, fastmath=True)
def _integrate_numba_sequential(f, nelems, weights, points):
  """
    We have to pass weights and points because
    np.polynomial.legendre.leggaus is not available
    inside of a Numba function.
    We can however pass Numba jitted functions `f` as arguments.
  """
  val = 0.0
  for j in range(0, nelems):
    a, b = j / nelems, (j+1) / nelems

    for k in range(len(weights)):
      val += (b - a) * weights[k] * f((b - a) * points[k] + a)

  return val


@njit(cache=True)
def _sin(x):
 return np.sin(18 * np.pi * x)


def integrate_numba(f, nelems, order, parallel=True):
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  weights = weights / 2
  return {True: _integrate_numba_parallel,
          False: _integrate_numba_sequential}[parallel](f, nelems, weights, points)


integrate_numba(_sin, 10, 3, parallel=False)

# Run the numba function for several input element sizes sequentially
for power in (2, 3, 5, 6, 7, 8):
  nelems = 10 ** power
  t0 = time.time()
  integrate_numba(_sin, nelems, 3, parallel=False)
  t1 = time.time()
  print(f"The sequential Numba implementation with {nelems} elements took {t1 - t0} seconds.\n")


integrate_numba(_sin, 10, 3, parallel=True)

# Run the numba function for several input element sizes in parallel
for power in (2, 3, 5, 6, 7, 8):
  nelems = 10 ** power
  t0 = time.time()
  integrate_numba(_sin, nelems, 3, parallel=True)
  t1 = time.time()
  print(f"The parallel Numba implementation with {nelems} elements took {t1 - t0} seconds.\n")