import numpy as np
from normalize_cy import normalize, normalize_loop
import time

from numba import njit, prange


def normalize_py(arr):
  for i in range(len(arr)):
    arr[i] /= np.linalg.norm(arr[i], ord=2)


def normalize_numpy(arr):
  arr /= np.linalg.norm(arr, keepdims=True, axis=1)


@njit(cache=True, fastmath=True, parallel=True)
def normalize_numba(arr):

  shape0 = arr.shape[0]
  batch = shape0 // 4
  ranges = np.array([0, batch, 2 * batch, 3 * batch, shape0])

  for i in prange(4):
    myarr = arr[ranges[i]: ranges[i+1]]
    myarr /= np.sqrt((myarr ** 2.0).sum(1)).reshape((myarr.shape[0], 1))


if __name__ == '__main__':
  A = np.random.randn(500000, 2)

  B = A.copy()
  t0 = time.time()
  normalize(B)
  t1 = time.time()

  print(f"Cython took {t1 - t0} seconds.")

  C = A.copy()
  t0 = time.time()
  normalize_numba(C)
  t1 = time.time()

  print(f"Numba took {t1 - t0} seconds.")

  D = A.copy()
  t0 = time.time()
  normalize_py(D)
  t1 = time.time()

  print(f"Python took {t1 - t0} seconds.")

  E = A.copy()
  t0 = time.time()
  normalize_numpy(E)
  t1 = time.time()

  print(f"Numpy took {t1 - t0} seconds.")

  F = A.copy()
  t0 = time.time()
  normalize_loop(F)
  t1 = time.time()

  print(f"Cython loop took {t1 - t0} seconds.")

  print(np.allclose(B, C) and np.allclose(C, D) and np.allclose(D, E) and np.allclose(E, F))

  import ipdb
  ipdb.set_trace()