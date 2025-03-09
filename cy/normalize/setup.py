from distutils.core import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy
import os

Cython.Compiler.Options.annotate = True

# Force Cython to use GCC instead of Clang
os.environ['CC'] = '/opt/homebrew/bin/gcc-14'
os.environ['CXX'] = '/opt/homebrew/bin/g++-14'

ext_modules = [
    Extension(
        'normalize',
        ['normalize.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True),
    include_dirs=[numpy.get_include()]
)

