""" [IMPORT] - njit and prange (parallel range) """
import numpy as np
from numba import njit, prange

import time

_ = np.newaxis


""" [SETUP] - number of processes to use """
NPROCS = 6


""" [FOCUS] - Parallel and sequential integration of a function over a mesh """
@njit(cache=True, parallel=True, fastmath=True)
def _integrate_numba_parallel(f, nelems, weights, points):
  """
    We have to pass weights and points because
    np.polynomial.legendre.leggaus is not available
    inside of a Numba function.
    We can however pass Numba jitted functions `f` as arguments.
  """
  ret = np.empty((NPROCS,), dtype=np.float64)
  batch_size = nelems // NPROCS

  # run in parallel
  for i in prange(NPROCS):
    start = i * batch_size
    end = (i + 1) * batch_size if i != (NPROCS-1) else nelems

    val = 0.0
    for j in range(start, end):
      a, b = j / nelems, (j+1) / nelems

      for k in range(len(weights)):
        val += (b - a) * weights[k] * f((b - a) * points[k] + a)

    ret[i] = val

  return ret.sum()  # sum sequentially


@njit(cache=True, fastmath=True)
def _integrate_numba_sequential(f, nelems, weights, points):
  """
    Same as above but sequential
  """
  val = 0.0
  for j in range(0, nelems):
    a, b = j / nelems, (j+1) / nelems

    for k in range(len(weights)):
      val += (b - a) * weights[k] * f((b - a) * points[k] + a)

  return val


# function we gonna integrate
@njit(cache=True)
def _sin(x):
 return np.sin(18 * np.pi * x)

 """ [/FOCUS] """


""" [HELPER] - Python function that calls sequential or parallel Numba integration """
def integrate_numba(f, nelems, order, parallel=True):
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  weights = weights / 2
  return {True: _integrate_numba_parallel,
          False: _integrate_numba_sequential}[parallel](f, nelems, weights, points)


print(f"Sequential and parallel outcome is the same ? {np.allclose(integrate_numba(_sin, 10, 3, parallel=False), integrate_numba(_sin, 10, 3, parallel=True))}\n\n\n")


""" [HELPER] - time sequential and parallel implementations for various mesh sizes """

# sequential
for power in (2, 3, 5, 6, 7, 8):
  nelems = 10 ** power
  t0 = time.time()
  integrate_numba(_sin, nelems, 3, parallel=False)
  t1 = time.time()
  print(f"The sequential Numba implementation with {nelems} elements took {t1 - t0} seconds.\n")


print("\n\n-----------------------------------------------------------------------------\n\n\n")


# parallel
for power in (2, 3, 5, 6, 7, 8):
  nelems = 10 ** power
  t0 = time.time()
  integrate_numba(_sin, nelems, 3, parallel=True)
  t1 = time.time()
  print(f"The parallel Numba implementation with {nelems} elements took {t1 - t0} seconds.\n")
