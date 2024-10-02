:: Build PhasorPy manylinux2014 wheels on Windows using Docker

setlocal
set PATH=C:\Windows;C:\Windows\System32;C:\Program Files\Docker\Docker\resources\bin
set CIBW_ARCHS_LINUX=auto
set CIBW_SKIP=pp* cp37* cp38* cp39* *musllinux*
:: set CIBW_TEST_SKIP=*
set CIBW_TEST_COMMAND=python -m pytest {project}/tests
set CIBW_BUILD_VERBOSITY=3

docker version
py -m cibuildwheel --platform linux
endlocal
