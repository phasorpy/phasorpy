"""PhasorPy package Setuptools script."""

# project metadata are defined in pyproject.toml
import sys

import numpy
from setuptools import Extension, setup

if sys.platform == 'win32':
    extra_compile_args = ['/openmp']
    extra_link_args: list[str] = []
elif sys.platform == 'darwin':
    # OpenMP not available in Xcode
    # https://mac.r-project.org/openmp/
    extra_compile_args = []  # ['-Xclang', '-fopenmp']
    extra_link_args = []  # ['-lomp']
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        'phasorpy._phasor',
        ['src/phasorpy/_phasor.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            # ('CYTHON_TRACE_NOGIL', '1'),
            # ('CYTHON_LIMITED_API', '1'),
            # ('Py_LIMITED_API', '1'),
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
]


setup(ext_modules=ext_modules)
