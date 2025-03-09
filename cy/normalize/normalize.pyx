""" [IMPORT] """
cimport cython
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport numpy as cnp
import numpy as np


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:, ::1] _normalize_cy_parallel(double[:, ::1] arr):
  cdef:
    int i, j, shape0, shape1
    double norm
    double[:, ::1] ret

  ret = np.empty((arr.shape[0], arr.shape[1]), dtype=np.double)
  shape0, shape1 = arr.shape[0], arr.shape[1]

  """ [FOCUS] - release the GIL """
  with nogil:

    # embarassingly parallelise the outer loop
    for i in prange(shape0):
      norm = 0.0
      for j in range(shape1):
        norm += arr[i, j] ** 2
      norm = sqrt(norm)

      for j in range(shape1):
        ret[i, j] = arr[i, j] / norm

  return ret


def normalize_cy_parallel(double[:, ::1] arr):
  return np.asarray(_normalize_cy_parallel(arr))
