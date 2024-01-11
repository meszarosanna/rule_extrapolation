import pytest
import torch


@pytest.fixture
def max_length():
    return 32


@pytest.fixture
def num_samples():
    return 20


@pytest.fixture
def num_train():
    return 128


@pytest.fixture
def num_val():
    return 64


@pytest.fixture
def num_test():
    return 32


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
