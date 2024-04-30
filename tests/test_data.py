import numpy as np
import pytest
import torch

from llm_non_identifiability.data import check_as_before_bs, check_same_number_as_bs
from llm_non_identifiability.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_aNbM_grammar_data,
    generate_aNbNaN_grammar_data,
    generate_aNbNcN_grammar_data,
    pad,
    PAD_token,
    check_sequence_finished,
    check_twice_many_as_than_bs,
    check_bs_in_the_middle,
    check_bs_together,
    check_more_as_before_bs,
    check_as_before_cs,
    check_bs_before_cs,
    check_more_bs_than_cs,
    check_same_number_as_bs_cs,
    check_as_before_bs_before_cs,
    check_in_dist_anbncn,
    EOS_token,
    generate_test_prompts,
    grammar_rules,
)


def test_aNbN_grammar_equal_as_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length, all_sequences=False)
    for sequence in sequences:
        assert check_same_number_as_bs(sequence)


def test_aNbN_grammar_as_before_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length, all_sequences=False)
    for sequence in sequences:
        assert check_as_before_bs(sequence)


def test_aNbNcN_grammar_equal_as_bs_cs(num_samples, max_length):
    sequences = generate_aNbNcN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_same_number_as_bs_cs(sequence)


def test_aNbNcN_grammar_as_before_bs_before_cs(num_samples, max_length):
    sequences = generate_aNbNcN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_as_before_bs_before_cs(sequence)


def test_aNbNcN_grammar_as_before_cs(num_samples, max_length):
    sequences = generate_aNbNcN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_as_before_cs(sequence)


def test_aNbNcN_grammar_bs_before_cs(num_samples, max_length):
    sequences = generate_aNbNcN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_bs_before_cs(sequence)


def test_aNbN_grammar_all_sequences(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length, all_sequences=True)
    lengths = sorted([len(sequence) - 2 for sequence in sequences])
    assert lengths == list(range(2, max_length + 1, 2))


def test_aNbN_grammar_only_even(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(
        num_samples, max_length, only_even=True, all_sequences=False
    )
    lengths = sorted([len(sequence) - 2 for sequence in sequences])
    assert lengths == list(range(4, max_length + 1, 4))


def test_aNbN_grammar_only_odd(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(
        num_samples, max_length, only_odd=True, all_sequences=False
    )
    lengths = sorted([len(sequence) - 2 for sequence in sequences])
    assert lengths == list(range(2, max_length + 1, 4))


def test_aNbNaN_grammar_twice_many_as_than_bs(num_samples, max_length):
    sequences = generate_aNbNaN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_twice_many_as_than_bs(sequence)


def test_aNbNaN_grammar_bs_together(num_samples, max_length):
    sequences = generate_aNbNaN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_bs_together(sequence)


def test_aNbNaN_grammar_bs_in_the_middle(num_samples, max_length):
    sequences = generate_aNbNaN_grammar_data(
        num_samples, max_length, all_sequences=False
    )
    for sequence in sequences:
        assert check_bs_in_the_middle(sequence)


def test_aNbNaN_grammar_all_sequences(num_samples, max_length):
    sequences = generate_aNbNaN_grammar_data(
        num_samples, max_length, all_sequences=True
    )
    lengths = sorted([len(sequence) - 2 for sequence in sequences])
    assert lengths == list(range(3, max_length + 1, 3))


def test_abN_equal_as_bs(num_samples, max_length):
    data = generate_abN_grammar_data(num_samples, max_length)
    for sequence in data:
        num_a = np.sum(sequence == 0)
        num_b = np.sum(sequence == 1)
        assert num_a == num_b


def test_aNbM_grammar_as_before_bs(num_samples, max_length):
    sequences = generate_aNbM_grammar_data(num_samples, max_length)
    for sequence in sequences:
        assert check_as_before_bs(sequence)


def test_pad_varying_sequence_lengths():
    data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

    expected_result = [
        [1, 2, 3, PAD_token.item()],
        [4, 5, PAD_token.item(), PAD_token.item()],
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

    sequence = torch.tensor([1, 1])
    assert check_as_before_bs(sequence) == True

    sequence = torch.tensor([0, 0])
    assert check_as_before_bs(sequence) == True


def test_check_more_as_before_bs():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_more_as_before_bs(sequence) == True

    sequence = torch.tensor([0, 1, 0, 1, 0, 0])
    assert check_more_as_before_bs(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1])
    assert check_more_as_before_bs(sequence) == True

    sequence = torch.tensor([1, 1])
    assert check_more_as_before_bs(sequence) == False

    sequence = torch.tensor([0, 0])
    assert check_more_as_before_bs(sequence) == True


def test_check_more_bs_than_cs():
    sequence = torch.tensor([1, 1, 2, 2])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([2, 1, 1])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([0, 0])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([0, 2, 0, 2, 0, 1])
    assert check_more_bs_than_cs(sequence) == False

    sequence = torch.tensor([0, 0, 2])
    assert check_more_bs_than_cs(sequence) == False


def test_check_bs_together():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_bs_together(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1, 0, 0])
    assert check_bs_together(sequence) == True

    sequence = torch.tensor([1, 1])
    assert check_bs_together(sequence) == True

    sequence = torch.tensor([0, 0])
    assert check_bs_together(sequence) == False


def test_check_bs_in_the_middle():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_bs_in_the_middle(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1, 0, 0])
    assert check_bs_in_the_middle(sequence) == True

    sequence = torch.tensor([1, 1])
    assert check_bs_in_the_middle(sequence) == True

    sequence = torch.tensor([0, 0])
    assert check_bs_in_the_middle(sequence) == False


def test_check_same_number_as_bs():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_same_number_as_bs(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([0, 1, 1, 0])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([1, 1, 0, 0])
    assert check_same_number_as_bs(sequence) == True


def test_check_twice_many_as_than_bs():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_twice_many_as_than_bs(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1, 0, 0])
    assert check_twice_many_as_than_bs(sequence) == True

    sequence = torch.tensor([1, 0, 0, 0, 0, 1])
    assert check_twice_many_as_than_bs(sequence) == True


def test_check_in_distr_anbncn():
    sequence = torch.tensor([0, 0, 1, 0, 1])
    assert check_in_dist_anbncn(sequence) == False

    sequence = torch.tensor([0, 0, 1, 1])
    assert check_in_dist_anbncn(sequence) == True

    sequence = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    assert check_in_dist_anbncn(sequence) == True

    sequence = torch.tensor([2, 2, 0, 1])
    assert check_in_dist_anbncn(sequence) == False


def test_check_sequence_finished():
    sequence = torch.tensor([0, 1, 0, 1])
    assert check_sequence_finished(sequence) == False

    sequence = torch.tensor([0, 1, EOS_token.item(), 0, 1])
    assert check_sequence_finished(sequence) == False

    sequence = torch.tensor([0, 1, EOS_token.item(), 0, 1, EOS_token.item(), 0, 1])
    assert check_sequence_finished(sequence) == False

    sequence = torch.tensor([0, 1, EOS_token.item()])
    assert check_sequence_finished(sequence) == True

    sequence = torch.tensor(
        [0, 1, EOS_token.item(), EOS_token.item(), PAD_token.item()]
    )
    assert check_sequence_finished(sequence) == True


def test_generate_test_prompts(grammar):
    length = 6
    if grammar == "aNbNcN":
        prompts = generate_test_prompts(grammar, length)
        assert prompts.shape == (3**length, length + 1)
    else:
        prompts = generate_test_prompts(grammar, length)
        assert prompts.shape == (2**length, length + 1)


@pytest.mark.parametrize("grammar", ["aNbN", "abN", "aNbM", "aNbNcN"])
def test_grammar_rules(max_length, grammar, num_samples):
    rules = grammar_rules(grammar)

    aNbN_data = torch.from_numpy(
        pad(generate_aNbN_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()
    abN_data = torch.from_numpy(
        pad(generate_abN_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()
    aNbM_data = torch.from_numpy(
        pad(generate_aNbM_grammar_data(num_samples=num_samples, max_length=max_length))
    ).long()
    aNbNcN_data = torch.from_numpy(
        pad(
            generate_aNbNcN_grammar_data(num_samples=num_samples, max_length=max_length)
        )
    ).long()

    if grammar == "aNbN":
        assert torch.all(torch.tensor([rules(d) for d in aNbN_data]))
    elif grammar == "abN":
        assert torch.all(torch.tensor([rules(d) for d in abN_data]))
    elif grammar == "aNbM":
        assert torch.all(torch.tensor([rules(d) for d in aNbM_data]))
    elif grammar == "aNbNcN":
        assert torch.all(torch.tensor([rules(d) for d in aNbNcN_data]))
