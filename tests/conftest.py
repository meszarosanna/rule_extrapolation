import pytest


@pytest.fixture
def max_length():
    return 32


@pytest.fixture
def n():
    return 20


@pytest.fixture
def n_train():
    return 128


@pytest.fixture
def n_val():
    return 64


@pytest.fixture
def n_test():
    return 32
