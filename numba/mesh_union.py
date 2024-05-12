import numpy as np
from itertools import count
from typing import Tuple
from matplotlib import pyplot as plt

import time


# First some utility functions


def plot_meshes(list_of_elements, list_of_points):

    fig, ax = plt.subplots()
    for elems, points in zip(list_of_elements, list_of_points):
        ax.triplot(*points.T, elems, alpha=0.5)

    plt.show()


def create_mesh(nx: int, ny: int, translate=None) -> Tuple[np.ndarray, np.ndarray]:
  """
    Create a regular triangular mesh over [0, 1] x [0, 1].
    Can optionally be translated by a vector `translate`.
    
    Parameters
    ----------
    nx: number of mesh vertices in x-direction
    ny: number of mesh vertices in y-direction
    translate: (optional) translation vector.
  """

  points = np.stack(list(map(np.ravel, np.meshgrid(np.linspace(0, 1, nx),
                                                   np.linspace(0, 1, ny) ))), axis=1)    

  if translate is not None:
    points += np.asarray(translate)[None]

  points = np.round(points, 8)

  indices = (np.arange(nx * ny).reshape(ny, nx)[:-1, :-1]).ravel()
  quads = np.stack([indices, indices+1, indices+ny+1, indices+ny], axis=1)
  elements = quads[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
  return elements, points

from numba import njit, prange


@njit(cache=True)
def make_numba_indexmap(points):
  # The function is compiled for the input type the first time it is called with that input
  """
    Create a hashmap that maps each point in `points` to a unique running index.
    Assumes the points in `points` to already be unique.
    
    This initialisation of the hashmap is not pythonic but we have to do it in
    this way because numba does not support the ingredients for the pythonic
    implementation.
  """
  map_coord_index = {}
  i = 0
  for point in points:
    # We cannot convert array to tuple directly because the length
    # of the tuple has to be known at compile time.
    map_coord_index[ (point[0], point[1]) ] = i
    i += 1

  return map_coord_index


@njit(cache=True, parallel=True)
def renumber_elements_from_indexmap(elements, points, map_coord_index):
  """
    Given a 2D array whose rows represent element,
    a 2D array of points and a map that maps each point in 
    `points` to a unique index, renumber the elements using the
    indexmap.

    Parameters
    ----------
    elements: element array to renumber
    points: 2D element vertex coordinates
    map_cord_index: hashmap mapping a tuple of coordinates to a
                    unique index.
  """
  newelems = np.empty(elements.shape, dtype=np.int64)
  
  for i in prange(len(elements)):  # run the assignment in parallel
    
    myvertices = elements[i]
    # for j, point in enumerate(points[myvertices]):
    for j in range(len(myvertices)):
      point = points[myvertices[j]]
      newelems[i, j] = map_coord_index[ (point[0], point[1]) ]
      
  return newelems


def take_mesh_union_numba(elems0, points0, elems1, points1):
  """
    Take the union of two meshes using Numpy and Numba functionality.
  """
  new_points = np.unique(np.concatenate([points0, points1]), axis=0)
  
  
  ### Now we use the new Numba implementation

  # map each unique point to an index in Numba
  map_point_index = make_numba_indexmap(new_points)

  mapped_elems = \
    np.concatenate([ 
                      renumber_elements_from_indexmap(myelems, mypoints, map_point_index)
                      for myelems, mypoints in zip([elems0, elems1], [points0, points1]) 
                   ])
  
  ### The rest is the same
  _, unique_indices = np.unique(np.sort(mapped_elems, axis=1), return_index=True, axis=0)
  new_elems = mapped_elems[unique_indices]

  return new_elems, new_points


elems0, points0 = create_mesh(2001, 2001)
elems1, points1 = create_mesh(2001, 2001, translate=[0.5, 0])

t0 = time.time()
elems_nb, points_nb = take_mesh_union_numba(elems0, points0, elems1, points1)
t1 = time.time()

print(f"Numba operation took {t1 - t0} seconds.")