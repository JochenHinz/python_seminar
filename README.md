The GitHub repository for the `Advanced Scientific Programming in Python` course, taught by Jochen Hinz.

The course is divided into three topics:

1. Idiomatic Python
   - What makes code readable ?
   - `*args`, `**kwargs`, star syntax variable unpacking
   - Advanced uses of Python dictionaries
   - Python truthiness
   - Advanced iteration and the use of `itertools`
   - The use of `functools` and decorators
   - The concept of hashing

2. Object-oriented Programming in Python
   - A wrap-up of the basics of OOP
   - Favouring out of place operations for safety and catching errors early in `__init__`
   - Inheritance and abstract base classes
   - (Admissible use cases of) Multiple inheritance
   - Writing hashable classes and the use of `collections.abc`
   - `classmethod`, `cached_property` and the use of `self.__class__`
   - (Direct and indirect) Inheriting from `numpy.ndarray`. Compatibility with `numpy.ufunc`'s and broadcasting
   - Creative uses of Context Managers

3. The best of two worlds: fast and readable Python
   - Advanced `Numpy` concepts
   - Writing a parallelised Python code
   - Using `Numba` for JIT compilation
   - Pimping your code with `Cython`

We additionally discuss static type checking using Mypy.


There are two supplementary notebooks:
`preparation.ipynb` and `numpy_vectorization.ipynb`.

These notebooks cover fundamental and intermediate Python concepts and serve as a refresher for the course.
Please note that this is not a beginner’s course; a certain level of Python proficiency is expected.
The topics in these supplementary notebooks are considered prerequisite knowledge for the course.
