import pytest
import torch


@pytest.fixture
def max_length():
    return 16


@pytest.fixture
def num_samples():
    return 8


@pytest.fixture
def num_train():
    return 32


@pytest.fixture
def num_val():
    return 16


@pytest.fixture
def num_test():
    return 8


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
