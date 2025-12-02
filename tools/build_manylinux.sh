# Build PhasorPy manylinux ABI3 wheels on Linux or macOS using Docker

export CIBW_ARCHS_LINUX=auto
export CIBW_SKIP="cp38* cp39* cp310* cp311* cp313* cp314-* *musllinux*"
export CIBW_TEST_SKIP=cp314t*
export CIBW_TEST_COMMAND="pytest {project}/tests"
export CIBW_BUILD_VERBOSITY=3

docker version
python3 -m cibuildwheel --platform linux
