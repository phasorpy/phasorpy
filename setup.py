"""PhasorPy package Setuptools script."""

# project metadata are defined in pyproject.toml

import numpy
from setuptools import Extension, setup

ext_modules = [
    Extension(
        'phasorpy._phasor',
        ['src/phasorpy/_phasor.pyx'],
        include_dirs=[numpy.get_include()],
        define_macros=[
            # ('CYTHON_TRACE_NOGIL', '1'),
            # ('CYTHON_LIMITED_API', '1'),
            # ('Py_LIMITED_API', '1'),
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
]


setup(ext_modules=ext_modules)
