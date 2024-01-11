from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_aNbM_grammar_data,
    pad,
    EOS_token,
)

import numpy as np

import torch

from llm_non_identifiability.data import check_as_before_bs, check_same_number_as_bs


def test_aNbN_grammar_equal_as_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        num_as = np.sum(sequence == 0)
        num_bs = np.sum(sequence == 1)
        assert num_as == num_bs


def test_aNbN_grammar_as_before_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        # find the first b
        first_b = np.where(sequence == 1)[0][0]
        # find the last a
        last_a = np.where(sequence == 0)[0][-1]
        assert first_b > last_a


def test_abN_equal_as_bs(num_samples, max_length):
    data = generate_abN_grammar_data(num_samples, max_length)
    for sequence in data:
        num_a = np.sum(sequence == 0)
        num_b = np.sum(sequence == 1)
        assert num_a == num_b


def test_aNbM_grammar_as_before_bs(num_samples, max_length):
    sequences = generate_aNbM_grammar_data(num_samples, max_length)
    for sequence in sequences:
        # check only if there is an a
        if np.sum(sequence == 0) > 0:
            # find the first b
            first_b = np.where(sequence == 1)[0][0]
            # find the last a
            last_a = np.where(sequence == 0)[0][-1]
            assert first_b > last_a


def test_pad_varying_sequence_lengths():
    data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    expected_result = [
        [1, 2, 3, EOS_token.item()],
        [4, 5, EOS_token.item(), EOS_token.item()],
        [6, 7, 8, 9],
    ]
    result = pad(data)

    # check that the result is as expected with .all()
    assert (np.array(result) == np.array(expected_result)).all()


def test_check_as_before_bs():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_as_before_bs(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1])
    assert check_as_before_bs(sequence) == True


def test_check_same_number_as_bs():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_same_number_as_bs(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([0, 1, 1, 0])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([1, 1, 0, 0])
    assert check_same_number_as_bs(sequence) == True
