# Build PhasorPy manylinux wheels on Linux or macOS using Docker

export CIBW_ARCHS_LINUX=auto
export CIBW_SKIP="cp38* cp39* cp310* *musllinux*"
export CIBW_TEST_SKIP=cp314*
export CIBW_TEST_COMMAND="pytest {project}/tests"
export CIBW_BUILD_VERBOSITY=3

docker version
python3 -m cibuildwheel --platform linux
