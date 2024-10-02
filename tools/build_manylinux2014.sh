# Build PhasorPy manylinux2014 wheels on Linux or macOS using Docker

export CIBW_ARCHS_LINUX=auto
export CIBW_SKIP="pp* cp37* cp38* cp39* *musllinux*"
# export CIBW_TEST_SKIP="*"
export CIBW_TEST_COMMAND="pytest {project}/tests"
export CIBW_BUILD_VERBOSITY=3

docker version
python3 -m cibuildwheel --platform linux
