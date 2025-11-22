"""PhasorPy package Setuptools script."""

# project metadata are defined in pyproject.toml
import os
import sys
import sysconfig

import numpy
from setuptools import Extension, setup

DEBUG = bool(os.environ.get('PHASORPY_DEBUG', False))
LIMITED_API = not sysconfig.get_config_var('Py_GIL_DISABLED')

print()
print(f'Building with numpy-{numpy.__version__}')
print()

if sys.platform == 'win32':
    extra_compile_args = ['/openmp']
    extra_link_args: list[str] = []
    if DEBUG:
        extra_compile_args += [
            '/Zi',
            '/Od',
            '/DCYTHON_TRACE=1',
            # '/DCYTHON_TRACE_NOGIL=1',  # too slow
        ]
        extra_link_args += ['-debug:full']
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
        'phasorpy._phasorpy',
        ['src/phasorpy/_phasorpy.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            # ('CYTHON_TRACE_NOGIL', '1'),
            ('NPY_NO_DEPRECATED_API', 'NPY_2_0_API_VERSION'),
        ]
        + (
            [('Py_LIMITED_API', 0x030C0000), ('CYTHON_LIMITED_API', '1')]
            if LIMITED_API
            else []
        ),
        py_limited_api=LIMITED_API,
    )
]


setup(
    ext_modules=ext_modules,
    options=(
        {'bdist_wheel': {'py_limited_api': 'cp312'}} if LIMITED_API else {}
    ),
)
