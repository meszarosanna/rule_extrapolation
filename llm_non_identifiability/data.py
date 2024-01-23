import numpy as np
import torch

SOS_token = np.array([2])
EOS_token = np.array([3])
PAD_token = np.array([4])

from itertools import product

import dataclasses
from typing import Dict


@dataclasses.dataclass
class GrammarMetrics:
    as_before_bs_accuracy: float = 0.0
    same_number_as_bs_accuracy: float = 0.0
    finished_accuracy: float = 0.0
    grammatical_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def generate_aNbN_grammar_data(
    num_samples: int,
    max_length: int = 32,
    all_sequences: bool = True,
    only_even: bool = False,
    only_odd: bool = False,
) -> list:
    """
    PCFG with two rules:
    - number of a's and b's must be the same
    - a's come first, followed by b's

    :param only_even: generates only sequences with even number of a's and b's
    :param only_odd: generates only sequences with odd number of a's and b's
    :param all_sequences: generates all sequences up to max_length (i.e., the longest will have max_length // 2 a's and b's)
    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length

    """

    if all_sequences + only_even + only_odd > 1:
        raise ValueError("Only one of all_sequences, only_even, only_odd can be True")

    if all_sequences is True:
        lengths = np.linspace(
            start=1, stop=max_length // 2, num=max_length // 2, dtype=int, endpoint=True
        )
    elif only_even is True:
        lengths = np.linspace(
            start=2,
            stop=max_length // 2,
            num=max_length // 2 // 2,
            dtype=int,
            endpoint=True,
        )
    elif only_odd is True:
        lengths = np.linspace(
            start=1,
            stop=max_length // 2,
            num=max_length // 2 // 2,
            dtype=int,
            endpoint=True,
        )
    else:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)

    data = []

    for length in lengths:
        data.append(
            np.concatenate((SOS_token, np.zeros(length), np.ones(length), EOS_token))
        )

    return data


def generate_abN_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with one rule:
    - number of a's and b's must be the same

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)

    data = []

    for lengths in lengths:
        abN = np.concatenate((np.zeros(lengths), np.ones(lengths)))
        # shuffle the symbols between start and end tokens
        np.random.shuffle(abN)
        data.append(np.concatenate((SOS_token, abN, EOS_token)))

    return data


def generate_aNbM_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with one rule:
    - a's are before b's

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    lengths_a = np.random.randint(low=1, high=max_length - 2, size=num_samples)
    lengths_b = np.ones_like(lengths_a) * max_length - lengths_a - 2

    data = []

    for la, lb in zip(lengths_a, lengths_b):
        data.append(np.concatenate((SOS_token, np.zeros(la), np.ones(lb), EOS_token)))

    return data


def pad(data: list, max_seq_length: int = 0) -> np.ndarray:
    """
    Pad data with PAD token
    :param data:
    :param max_seq_length: maximum sequence length
    :return: numpy array with padded data of shape (batch_size, max_batch_length)
    """

    if max_seq_length == 0:
        # Get longest sequence in the dataset
        for seq in data:
            if len(seq) > max_seq_length:
                max_seq_length = len(seq)

    # Append padding tokens until it reaches the max length
    for i, seq in enumerate(data):
        remaining_length = max_seq_length - len(seq)

        if remaining_length > 0:
            data[i] = np.concatenate((data[i], [PAD_token.item()] * remaining_length))

    return np.array(data)


def check_as_before_bs(sequence: torch.Tensor):
    """
    Check if the first b comes after the last a
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(a_tokens := torch.where(sequence == 0)[0]) > 0:
        # find the last a
        last_a = a_tokens[-1]

        if len(b_tokens := torch.where(sequence == 1)[0]) > 0:
            # find the first b
            first_b = b_tokens[0]

            return first_b > last_a
        else:
            return True
    else:
        return False


def check_same_number_as_bs(sequence: torch.Tensor):
    """
    Check if the number of a's and b's is the same
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == 0)
    num_bs = torch.sum(sequence == 1)
    return num_as == num_bs


def check_more_as_than_bs(sequence: torch.Tensor):
    """
    Check if there are more a's than b's
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == 0)
    num_bs = torch.sum(sequence == 1)
    return num_as >= num_bs


def check_sequence_finished(sequence: torch.Tensor):
    """
    Check if the sequence is finished (EOS token)
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    # check whether there are no 0's or 1's in the sequence after the first EOS token

    # find the first EOS token
    if len(eos_tokens := torch.where(sequence == EOS_token.item())[0]) > 0:
        first_EOS = torch.where(sequence == EOS_token.item())[0][0]
        # check whether there are any 0's or 1's after the first EOS token
        return (
            torch.sum(sequence[first_EOS + 1 :] == 0)
            + torch.sum(sequence[first_EOS + 1 :] == 1)
            == 0
        )
    else:
        return False


def generate_test_prompts(length: int = 6):
    """
    Generates all prompts of a given length with symbols a and b
    :param length:
    :return:
    """

    symbols = [0, 1]
    prompts = torch.tensor(list(product(symbols, repeat=length)), dtype=torch.long)

    return prompts


def grammar_rules(grammar):
    """
    Selects the rules the grammar needs to satisfy.
    :param grammar:
    """
    if grammar == "aNbN":
        return lambda x: check_same_number_as_bs(x) and check_as_before_bs(x)
    elif grammar == "abN":
        return lambda x: check_same_number_as_bs(x)
    elif grammar == "aNbM":
        return lambda x: check_as_before_bs(x)
    else:
        raise ValueError(f"Unknown grammar {grammar}")


def prompt_grammar_rules(grammar):
    """
    Selects the rules that check whether a prompt can be completed as such that it satisfies the rules of the grammar.
    It is used to split the test_prompts into in-distribution and out-of-distribution.

    NOTE: these rules are LESS strict than the grammar_rules, because even if the prompt does not satisfy the grammar rules,
    it might be completed as such that it does.
    :param grammar:

    """
    if grammar == "aNbN":
        return lambda x: check_as_before_bs(x) and check_more_as_than_bs(x)
    elif grammar == "abN":
        return lambda x: True
    elif grammar == "aNbM":
        return lambda x: check_as_before_bs(x)
    else:
        raise ValueError(f"Unknown grammar {grammar}")
