import numpy as np
import torch

PAD_token = 4
SOS_token = np.array([2])
EOS_token = np.array([3])


def generate_aNbN_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with two rules:
    - number of a's and b's must be the same
    - a's come first, followed by b's

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length

    """

    lengths = np.random.randint(low=1, high=max_length // 2, size=num_samples)

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

    lengths = np.random.randint(low=1, high=max_length // 2, size=num_samples)

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

    lengths_a = np.random.randint(low=0, high=max_length - 2, size=num_samples)
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
            data[i] = np.concatenate((data[i], [PAD_token] * remaining_length))

    return np.array(data)


def check_as_before_bs(sequence: torch.Tensor):
    """
    Check if the first b comes after the last a
    :param sequence:
    :return:
    """
    # find the first b
    first_b = torch.where(sequence == 1)[0][0]
    # find the last a
    last_a = torch.where(sequence == 0)[0][-1]
    return first_b > last_a
