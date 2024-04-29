import numpy as np
import torch

SOS_token = np.array([2])
EOS_token = np.array([3])
PAD_token = np.array([4])
OPENING_PARENTHESIS_token = np.array([7])
CLOSING_PARENTHESIS_token = np.array([8])
OPENING_BRACKET_token = np.array([9])
CLOSING_BRACKET_token = np.array([10])

from itertools import product

import dataclasses
from typing import Dict


# to_dict: creates a dictionary: {'as_before_bs_accuracy': 0.0, 'as_before_bs_completion_accuracy':0.0, etc}
@dataclasses.dataclass
class GrammarMetrics:
    as_before_bs_accuracy: float = 0.0
    as_before_bs_completion_accuracy: float = 0.0

    same_number_as_bs_accuracy: float = 0.0
    finished_accuracy: float = 0.0
    grammatical_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


# generates aNbN grammar: all sequences, all even, all odd or sequences of random length and num_samples number
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
        lengths = np.array(list(range(2, max_length // 2 + 1, 2)))
    elif only_odd is True:
        lengths = np.array(list(range(1, max_length // 2 + 1, 2)))
    else:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)

    data = []

    for length in lengths:
        data.append(
            np.concatenate((SOS_token, np.zeros(length), np.ones(length), EOS_token))
        )

    return data  # list containing the sequences of max length max_length+2


def generate_aNbNaN_grammar_data(
    num_samples: int, max_length: int = 32, all_sequences=True
) -> list:
    """
    PCFG with two rules:
    - number of a's is twice the number of b's
    - N a's come first, followed by N b's, then N a's again

    :param all_sequences:
    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length

    """
    if all_sequences is True:
        lengths = np.linspace(
            start=1, stop=max_length // 3, num=max_length // 3, dtype=int, endpoint=True
        )
    else:
        lengths = np.random.randint(low=1, high=max_length // 3 + 1, size=num_samples)

    data = []

    for length in lengths:
        data.append(
            np.concatenate(
                (
                    SOS_token,
                    np.zeros(length),
                    np.ones(length),
                    np.zeros(length),
                    EOS_token,
                )
            )
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
        return True


def check_bs_in_the_middle(sequence: torch.Tensor):
    """
    Check if the b's are in the middle
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == 1)[0]) > 0:
        # find the first b
        first_b = b_tokens[0]
        last_b = b_tokens[-1]

        if len(sequence[:first_b]) == len(sequence[last_b + 1 :]):
            return True
        else:
            return False
    else:
        return False


def check_bs_together(sequence: torch.Tensor):
    """
    Check if the b's are in the middle
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == 1)[0]) > 0:
        # find the first b
        first_b = b_tokens[0]
        last_b = b_tokens[-1]

        if (b_subsequence := sequence[first_b:last_b]).sum() == len(b_subsequence):
            return True
        else:
            return False
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


def check_twice_many_as_than_bs(sequence: torch.Tensor):
    """
    Check if the number of a's and b's is the same
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == 0)
    num_bs = torch.sum(sequence == 1)
    return num_as == 2 * num_bs


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


def check_more_as_before_bs(sequence: torch.Tensor):
    """
    Check if there are more a's than b's
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == 1)[0]) > 0:
        first_b = b_tokens[0]

        num_as = torch.sum(sequence[:first_b] == 0)
        num_bs = torch.sum(sequence == 1)
        return num_as >= num_bs

    else:
        return True


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
        first_EOS = eos_tokens[0]
        # check whether there are any 0's or 1's after the first EOS token
        return (
            torch.sum(sequence[first_EOS + 1 :] == 0)
            + torch.sum(sequence[first_EOS + 1 :] == 1)
            == 0
        )
    else:
        return False


def generate_test_prompts(length: int = 6, grammar: str = "aNbN"):
    """
    Generates all prompts of a given length with symbols a and b
    :param length:
    :return:
    """
    num_samples = 2**length
    if grammar in ["aNbN", "abN", "aNbM", "aNbNaN"]:
        symbols = [0, 1]
        prompts = torch.tensor(list(product(symbols, repeat=length)), dtype=torch.long)

        # add SOS
        prompts = torch.cat(
            (torch.ones((prompts.shape[0], 1), dtype=torch.long) * SOS_token, prompts),
            dim=1,
        )
    elif grammar == "parentheses":
        data = torch.tensor(
            generate_matched_parentheses_data(
                num_samples=num_samples / 2, max_length=length, fixed_length=True
            ),
            dtype=torch.long,
        )
        ood_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_PARENTHESIS_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_PARENTHESIS_token,
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        id_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_PARENTHESIS_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_PARENTHESIS_token,
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        prompts = torch.cat((ood_prompts, id_prompts), dim=0)
    elif grammar == "brackets":
        data = torch.tensor(
            generate_matched_brackets_data(
                num_samples=num_samples / 2, max_length=length, fixed_length=True
            ),
            dtype=torch.long,
        )
        ood_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_BRACKET_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        id_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_BRACKET_token,
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS
        prompts = torch.cat((ood_prompts, id_prompts), dim=0)

    elif grammar == "parentheses_and_brackets":
        data = torch.tensor(
            generate_matched_parentheses_and_brackets_data(
                num_samples=num_samples / 2, max_length=length, fixed_length=True
            ),
            dtype=torch.long,
        )

        # generate torch 0-1 sequence in shape (data.shape[0], 1)
        bernoulli = torch.bernoulli(0.5 * torch.ones((data.shape[0], 1)))

        ood_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.where(
                    bernoulli == 1,
                    CLOSING_PARENTHESIS_token.item(),
                    CLOSING_BRACKET_token.item(),
                ),
                torch.where(
                    bernoulli == 1,
                    OPENING_PARENTHESIS_token.item(),
                    OPENING_BRACKET_token.item(),
                ),
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        id_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.where(
                    bernoulli == 1,
                    OPENING_PARENTHESIS_token.item(),
                    OPENING_BRACKET_token.item(),
                ),
                torch.where(
                    bernoulli == 1,
                    CLOSING_PARENTHESIS_token.item(),
                    CLOSING_BRACKET_token.item(),
                ),
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        prompts = torch.cat((ood_prompts, id_prompts), dim=0)
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
    elif grammar == "aNbNaN":
        return (
            lambda x: check_twice_many_as_than_bs(x)
            and check_bs_in_the_middle(x)
            and check_bs_together(x)
        )
    elif grammar == "brackets":
        return lambda x: check_matched_brackets(x)
    elif grammar == "parentheses":
        return lambda x: check_matched_parentheses(x)
    elif grammar == "parentheses_and_brackets":
        return lambda x: check_matched_parentheses_and_brackets(x)
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
    elif grammar == "aNbNaN":
        return lambda x: check_as_before_bs(x) and check_bs_together(x)
    elif grammar == "brackets":
        return lambda x: check_matched_brackets(x)
    elif grammar == "parentheses":
        return lambda x: check_matched_parentheses(x)
    elif grammar == "parentheses_and_brackets":
        return lambda x: check_matched_parentheses_and_brackets(x)
    else:
        raise ValueError(f"Unknown grammar {grammar}")


import random


def generate_matched_parentheses_and_brackets(n):
    """
    Generate a word of length n with paired () and [].
    """
    if n == 0:
        return np.concatenate((SOS_token, EOS_token))
    elif n % 2 == 1:
        raise ValueError("Length can only be even")
    else:
        word = []
        stack = []
        while len(word) < n:  # Each pair of parentheses or brackets adds 2 characters
            if len(stack) == 0:
                choice = random.choice(
                    [OPENING_PARENTHESIS_token, OPENING_BRACKET_token]
                )
            elif stack[-1] == OPENING_PARENTHESIS_token:
                choice = random.choice(
                    [
                        OPENING_PARENTHESIS_token,
                        OPENING_BRACKET_token,
                        CLOSING_PARENTHESIS_token,
                    ]
                )
                if len(word) + len(stack) >= n:
                    choice = CLOSING_PARENTHESIS_token

            elif stack[-1] == OPENING_BRACKET_token:
                choice = random.choice(
                    [
                        OPENING_PARENTHESIS_token,
                        OPENING_BRACKET_token,
                        CLOSING_BRACKET_token,
                    ]
                )
                if len(word) + len(stack) >= n:
                    choice = CLOSING_BRACKET_token

            if choice == OPENING_PARENTHESIS_token:
                word.append(OPENING_PARENTHESIS_token)
                stack.append(OPENING_PARENTHESIS_token)
            elif choice == OPENING_BRACKET_token:
                word.append(OPENING_BRACKET_token)
                stack.append(OPENING_BRACKET_token)
            elif choice == CLOSING_PARENTHESIS_token:
                word.append(CLOSING_PARENTHESIS_token)
                stack.pop()
            elif choice == CLOSING_BRACKET_token:
                word.append(CLOSING_BRACKET_token)
                stack.pop()

            if len(stack) == 0:
                break

        return np.concatenate((SOS_token, *word, EOS_token))


def check_matched_parentheses_and_brackets(sequence: torch.Tensor) -> bool:
    """
    Check if the parentheses and brackets are matched
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    stack = []
    for token in sequence:
        if token == OPENING_PARENTHESIS_token.item():
            stack.append(token)
        elif token == CLOSING_PARENTHESIS_token.item():
            if len(stack) == 0 or stack[-1] != OPENING_PARENTHESIS_token.item():
                return False
            stack.pop()
        elif token == OPENING_BRACKET_token.item():
            stack.append(token)
        elif token == CLOSING_BRACKET_token.item():
            if len(stack) == 0 or stack[-1] != OPENING_BRACKET_token.item():
                return False
            stack.pop()

    return len(stack) == 0


def generate_matched_parentheses(n):
    """
    Generate a word of length n with paired ().
    """
    if n == 0:
        return np.concatenate((SOS_token, EOS_token))
    elif n % 2 == 1:
        raise ValueError("Length can only be even")
    else:
        word = []
        stack = []
        while len(word) < n:  # Each pair of parentheses or brackets adds 2 characters
            if len(stack) == 0:
                choice = OPENING_PARENTHESIS_token
            elif stack[-1] == OPENING_PARENTHESIS_token:
                choice = random.choice(
                    [OPENING_PARENTHESIS_token, CLOSING_PARENTHESIS_token]
                )
                if len(word) + len(stack) >= n:
                    choice = CLOSING_PARENTHESIS_token

            if choice == OPENING_PARENTHESIS_token:
                word.append(OPENING_PARENTHESIS_token)
                stack.append(OPENING_PARENTHESIS_token)

            elif choice == CLOSING_PARENTHESIS_token:
                word.append(CLOSING_PARENTHESIS_token)
                stack.pop()

            if len(stack) == 0:
                break

        return np.concatenate((SOS_token, *word, EOS_token))


def check_matched_parentheses(sequence: torch.Tensor) -> bool:
    """
    Check if the parentheses are matched
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    stack = []
    for token in sequence:
        if token == OPENING_PARENTHESIS_token.item():
            stack.append(token)
        elif token == CLOSING_PARENTHESIS_token.item():
            if len(stack) == 0:
                return False
            stack.pop()

    return len(stack) == 0


def generate_matched_brackets(n):
    """
    Generate a word of length n with paired [].
    """
    if n == 0:
        return np.concatenate((SOS_token, EOS_token))
    elif n % 2 == 1:
        raise ValueError("Length can only be even")
    else:
        word = []
        stack = []
        while len(word) < n:  # Each pair of parentheses or brackets adds 2 characters
            if len(stack) == 0:
                choice = OPENING_BRACKET_token

            elif stack[-1] == OPENING_BRACKET_token:
                choice = random.choice([2, CLOSING_BRACKET_token])
                if len(word) + len(stack) >= n:
                    choice = CLOSING_BRACKET_token

            if choice == OPENING_BRACKET_token:
                word.append(OPENING_BRACKET_token)
                stack.append(OPENING_BRACKET_token)
            elif choice == CLOSING_BRACKET_token:
                word.append(CLOSING_BRACKET_token)
                stack.pop()

            if len(stack) == 0:
                break

        return np.concatenate((SOS_token, *word, EOS_token))


def check_matched_brackets(sequence: torch.Tensor) -> bool:
    """
    Check if the brackets are matched
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    stack = []
    for token in sequence:
        if token == OPENING_BRACKET_token.item():
            stack.append(token)
        elif token == CLOSING_BRACKET_token.item():
            if len(stack) == 0:
                return False
            stack.pop()

    return len(stack) == 0


def generate_matched_parentheses_data(
    num_samples: int, max_length: int = 32, fixed_length=False
) -> list:
    """


    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    if fixed_length is False:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)
        data = [generate_matched_parentheses(2 * l) for l in lengths]
    else:
        data = []
        while len(data) < num_samples:
            sample = generate_matched_parentheses(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)

    return data


def generate_matched_brackets_data(
    num_samples: int, max_length: int = 32, fixed_length=False
) -> list:
    """


    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    if fixed_length is False:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)
        data = [generate_matched_brackets(2 * l) for l in lengths]
    else:
        data = []
        while len(data) < num_samples:
            sample = generate_matched_parentheses(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)

    return data


def generate_matched_parentheses_and_brackets_data(
    num_samples: int, max_length: int = 32, fixed_length=False
) -> list:
    """


    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    if fixed_length is False:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)
        data = [generate_matched_parentheses_and_brackets(2 * l) for l in lengths]
    else:
        data = []
        while len(data) < num_samples:
            sample = generate_matched_parentheses_and_brackets(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)

    return data
