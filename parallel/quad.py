""" [IMPORT] """
import numpy as np
from concurrent import futures
from typing import Callable
import time


""" [HELPER] - THIS PART IS NOT SO IMPORTANT """

def integrate_sequential(f, nelems, order):
  x = np.linspace(0, 1, nelems + 1)
  grid = np.stack([x[:-1], x[1:]], axis=1)
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  weights = weights / 2

  result = 0.0
  for a, b in grid:
    result += (b - a) * (weights * f((b - a) * points + a)).sum()
  return result


# Sequential function that is used for parallel execution
def integrate_subgrid(args):
  f, grid, order, fargs = args  # unpack all arguments. Note that fargs is passed to f
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  weights = weights / 2

  result = 0.0
  for a, b in grid:
    result += (b - a) * (weights * f((b - a) * points + a, *fargs)).sum()
  return result

""" [/HELPER] """


""" [FOCUS] ON THIS PART """

# Parallel integration using ProcessPoolExecutor
def integrate_parallel(f: Callable, fargs, nelems: int, nprocs=4, order=3):
  """
    Parallel version of the sequential implementation from above.

    Parameters
    ----------
    f: positional argument only function to integrate, of the form f(x, *fargs)
    fargs: additional positional arguments passed to f, i.e., f(x, *fargs)
    nelems: number of elements to divide [0, 1] into
    nprocs: maximum number of parallel processes
    order: gaussian integration order
  """
  x = np.linspace(0, 1, nelems + 1)
  grid = np.stack([x[:-1], x[1:]], axis=1)  # [[a0, a1], [a1, a2], ...]

  # Split grid into roughly equal-sized subgrids for parallel processing
  subgrids = np.array_split(grid, nprocs, axis=0)

  args = [(f, subgrid, order, fargs) for subgrid in subgrids]

  # compute partial integrals in parallel
  with futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
    results = list(executor.map(integrate_subgrid, args))

  return sum(results)  # sum sequentially

""" [/FOCUS] """


""" [SETUP] - function to integrate """
def f(x, w):
  return np.sin(w * x)


""" [HELPER] - validate """
def main(nelems=100000, order=3, nprocs=4):
  w = 18 * np.pi

  # keep track of computational times
  t0 = time.time()
  result_seq = integrate_sequential(lambda x: f(x, w=w), nelems, order)
  t1 = time.time()
  result_par = integrate_parallel(f, (w,), nelems, nprocs=nprocs, order=order)
  t2 = time.time()

  # Check if sequential and parallel results are equivalent
  print(f"Parallel and sequential integral is the same? {np.allclose(result_seq, result_par)} \n\n")

  print(f"nprocs = {nprocs}, so we expect a maximum speedup by {nprocs}.\n\n")
  print(f"The sequential implementation took {t1 - t0} seconds.\n\n")
  print(f"The parallel implementation took {t2 - t1} seconds.\n\n")
  print("Parallelisation gives a speedup by a factor of \033[4m{}\033[0m. \n\n".format((t1 - t0) / (t2 - t1)))
