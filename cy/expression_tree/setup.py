from distutils.core import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy
import os

Cython.Compiler.Options.annotate = True
# os.environ['CC'] = '/usr/local/Cellar/gcc/13.2.0/bin/gcc-13'

ext_modules = [
      Extension('tree', ['tree.pyx'], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
]

# setup(ext_modules=cythonize("*.pyx", annotate=True),
#       include_dirs=[numpy.get_include()])

setup(ext_modules=cythonize(ext_modules, annotate=True), include_dirs=[numpy.get_include()])
