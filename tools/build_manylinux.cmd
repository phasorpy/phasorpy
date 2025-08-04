:: Build PhasorPy manylinux wheels on Windows using Docker

setlocal
set PATH=C:\Windows;C:\Windows\System32;C:\Program Files\Docker\Docker\resources\bin
set CIBW_ARCHS_LINUX=auto
set CIBW_SKIP=cp38* cp39* cp310* *musllinux*
set CIBW_TEST_SKIP=cp314*
set CIBW_TEST_COMMAND=pytest {project}/tests
set CIBW_BUILD_VERBOSITY=3

docker version
py -m cibuildwheel --platform linux
endlocal
