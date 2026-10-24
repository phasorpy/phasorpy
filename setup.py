# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""PhasorPy package Setuptools script."""

# project metadata are defined in pyproject.toml
import os
import sys
import sysconfig

import numpy
from setuptools import Extension, setup

DEBUG = os.environ.get('PHASORPY_DEBUG', '').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
LIMITED_API = not sysconfig.get_config_var('Py_GIL_DISABLED')

print()
print(f'Building with numpy-{numpy.__version__}')
print()

extra_compile_args: list[str] = []
extra_link_args: list[str] = []

if 'PYODIDE' in os.environ or 'PYODIDE_ROOT' in os.environ:
    # OpenMP/threads not available in Pyodide
    pass
elif sys.platform == 'darwin':
    # OpenMP not available in Xcode
    # https://mac.r-project.org/openmp/
    # extra_compile_args = ['-Xclang', '-fopenmp']
    # extra_link_args = ['-lomp']
    pass
elif sys.platform == 'win32':
    extra_compile_args = ['/openmp']
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

if DEBUG:
    if sys.platform == 'win32':
        extra_compile_args += [
            '/Zi',
            '/Od',
            '/DCYTHON_TRACE=1',
            # '/DCYTHON_TRACE_NOGIL=1',  # too slow
        ]
        extra_link_args += ['-debug:full']
    else:
        extra_compile_args += ['-g', '-O0', '-DCYTHON_TRACE=1']
        extra_link_args += ['-g']

ext_modules = [
    Extension(
        'phasorpy._phasorpy',
        ['src/phasorpy/_phasorpy.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            # ('CYTHON_TRACE_NOGIL', '1'),
            ('NPY_NO_DEPRECATED_API', 'NPY_2_1_API_VERSION'),
        ]
        + (
            [('Py_LIMITED_API', 0x030C0000), ('CYTHON_LIMITED_API', '1')]
            if LIMITED_API
            else []
        ),
        py_limited_api=LIMITED_API,
    )
]

if DEBUG:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        ext_modules, compiler_directives={'linetrace': True}
    )  # type: ignore[no-untyped-call]


setup(
    ext_modules=ext_modules,
    options=(
        {'bdist_wheel': {'py_limited_api': 'cp312'}} if LIMITED_API else {}
    ),
)
