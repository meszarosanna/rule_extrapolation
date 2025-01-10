import numpy as np
import torch
import random

# these are always used
SOS_token = np.array([0])
EOS_token = np.array([1])
PAD_token = np.array([2])

# only for aNbNcN and variants
A_token = np.array([3])
B_token = np.array([4])
C_token = np.array([5])

# only for parentheses and brackets
OPENING_PARENTHESIS_token = np.array([3])
CLOSING_PARENTHESIS_token = np.array([4])
OPENING_BRACKET_token = np.array([5])
CLOSING_BRACKET_token = np.array([6])

from itertools import product

import dataclasses
from typing import Dict


# to_dict: creates a dictionary: {'as_before_bs_accuracy': 0.0, 'as_before_bs_completion_accuracy':0.0, etc}
@dataclasses.dataclass
class GrammarMetrics:
    rule_2_accuracy: float = 0.0
    rule_2_completion_accuracy: float = 0.0

    rule_1_accuracy: float = 0.0
    rule_3_accuracy: float = 0.0
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
            np.concatenate(
                (
                    SOS_token,
                    A_token * np.ones(length),
                    B_token * np.ones(length),
                    EOS_token,
                )
            )
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
                    A_token * np.ones(length),
                    B_token * np.ones(length),
                    A_token * np.ones(length),
                    EOS_token,
                )
            )
        )

    return data


def generate_aNbNcN_grammar_data(
    num_samples: int, max_length: int = 32, all_sequences=True
) -> list:
    """
    PCFG with two rules:
    - number of a's is equal to the number of b's, equal to the number of c's
    - N a's come first, followed by N b's, then N c's

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
                    A_token * np.ones(length),
                    B_token * np.ones(length),
                    C_token * np.ones(length),
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
        abN = np.concatenate((A_token * np.ones(lengths), B_token * np.ones(lengths)))
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
        data.append(
            np.concatenate(
                (SOS_token, A_token * np.ones(la), B_token * np.ones(lb), EOS_token)
            )
        )

    return data


def generate_bNaM_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with one rule:
    - b's are before a's (begins with b, without SOS, EOS)

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    lengths_b = np.random.randint(low=1, high=max_length, size=num_samples)
    lengths_a = np.ones_like(lengths_b) * max_length - lengths_b

    data = []

    for lb, la in zip(lengths_b, lengths_a):
        data.append(np.concatenate((B_token * np.ones(la), A_token * np.ones(lb))))

    return data


def generate_baN_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with two rules:
    - begins with b
    - even number of a's

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    lengths = np.random.randint(low=1, high=max_length + 1, size=num_samples)

    data = []

    for l in lengths:
        num_a = np.random.randint(low=0, high=(l - 1) // 2 + 1)
        second_part = np.concatenate(
            (A_token * np.ones(num_a * 2), B_token * np.ones(l - 1 - num_a * 2))
        )
        # shuffle the symbols
        np.random.shuffle(second_part)

        data.append(np.concatenate((SOS_token, B_token, second_part, EOS_token)))

    return data


def generate_bbaN_grammar_data(num_samples: int, max_length: int = 32) -> list:
    """
    PCFG with two rules:
    - b's before a's ('bbbb' ok but 'aaaa' not)
    - even number of a's

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    lengths = np.random.randint(low=1, high=max_length + 1, size=num_samples)

    data = []

    for l in lengths:
        num_a = np.random.randint(low=0, high=(l - 1) // 2 + 1)
        second_part = np.concatenate(
            (B_token * np.ones(l - 1 - num_a * 2), A_token * np.ones(num_a * 2))
        )

        data.append(
            np.concatenate(
                (
                    SOS_token,
                    B_token * np.ones(l - num_a * 2),
                    A_token * np.ones(num_a * 2),
                    EOS_token,
                )
            )
        )

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


def check_parity(sequence: torch.Tensor):
    """
    Check if sequence has an even number of elements.
    :param sequence:
    :return:
    """
    if len(sequence) % 2 == 0:
        return True
    else:
        return False


def check_as_before_bs(sequence: torch.Tensor):
    """
    Check if the first b comes after the last a
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(a_tokens := torch.where(sequence == A_token.item())[0]) > 0:
        # find the last a
        last_a = a_tokens[-1]

        if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
            # find the first b
            first_b = b_tokens[0]

            return first_b > last_a
        else:
            return True
    else:
        return True


def check_bs_before_as(sequence: torch.Tensor):
    """
    Check if the first a comes after the last b. 'bbbb' ok, 'aaaa' not
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
        # find the last b
        last_b = b_tokens[-1]

        if len(a_tokens := torch.where(sequence == A_token.item())[0]) > 0:
            # find the first a
            first_a = a_tokens[0]

            return first_a > last_b
        else:
            return True
    else:
        return False


def check_as_before_cs(sequence: torch.Tensor):
    """
    Check if the first c comes after the last a
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(a_tokens := torch.where(sequence == A_token.item())[0]) > 0:
        # find the last a
        last_a = a_tokens[-1]

        if len(c_tokens := torch.where(sequence == C_token.item())[0]) > 0:
            # find the first c
            first_c = c_tokens[0]

            return first_c > last_a
        else:
            return True
    else:
        return True


def check_bs_before_cs(sequence: torch.Tensor):
    """
    Check if the first c comes after the last b
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
        # find the last b
        last_b = b_tokens[-1]

        if len(c_tokens := torch.where(sequence == C_token.item())[0]) > 0:
            # find the first c
            first_c = c_tokens[0]

            return first_c > last_b
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

    if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
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

    if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
        # find the first b
        first_b = b_tokens[0]
        last_b = b_tokens[-1]

        if (
            (b_subsequence := sequence[first_b : last_b + 1]) == B_token.item()
        ).sum() == len(b_subsequence):
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

    num_as = torch.sum(sequence == A_token.item())
    num_bs = torch.sum(sequence == B_token.item())
    return num_as == num_bs


def check_twice_many_as_than_bs(sequence: torch.Tensor):
    """
    Check if the number of a's and b's is the same
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == A_token.item())
    num_bs = torch.sum(sequence == B_token.item())
    return num_as == 2 * num_bs


def check_more_as_than_bs(sequence: torch.Tensor):
    """
    Check if there are more a's than b's
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == A_token.item())
    num_bs = torch.sum(sequence == B_token.item())
    return num_as >= num_bs


def check_more_bs_than_cs(sequence: torch.Tensor):
    """
    Check if there are more b's than c's
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_bs = torch.sum(sequence == B_token.item())
    num_cs = torch.sum(sequence == C_token.item())
    return num_bs >= num_cs


def check_more_as_before_bs(sequence: torch.Tensor):
    """
    Check if there are more a's than b's
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
        first_b = b_tokens[0]

        num_as = torch.sum(sequence[:first_b] == A_token.item())
        num_bs = torch.sum(sequence == B_token.item())
        return num_as >= num_bs

    else:
        return True


def check_same_number_as_bs_cs(sequence: torch.Tensor):
    """
    Check if the number of a's, b's and c's is the same
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == A_token.item())
    num_bs = torch.sum(sequence == B_token.item())
    num_cs = torch.sum(sequence == C_token.item())
    return (num_as == num_bs) and (num_bs == num_cs)


def check_as_before_bs_before_cs(sequence: torch.Tensor):
    """
    Check if the first b comes after the last a and the first c comes after the last b
    :param sequence:
    :return:
    """

    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(c_tokens := torch.where(sequence == C_token.item())[0]) > 0:
        # find the first c
        first_c = c_tokens[0]

        if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
            # find the first and last b
            last_b = b_tokens[-1]
            first_b = b_tokens[0]

            if len(a_tokens := torch.where(sequence == A_token.item())[0]) > 0:
                # find the last a
                last_a = a_tokens[-1]
                if (last_a < first_b) and (last_b < first_c):
                    return True
                else:
                    return False
            else:
                return check_bs_before_cs(sequence)
        else:
            return check_as_before_cs(sequence)
    else:
        return check_as_before_bs(sequence)


def check_in_dist_anbncn(sequence: torch.Tensor):
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(c_tokens := torch.where(sequence == C_token.item())[0]) == 0:
        if len(b_tokens := torch.where(sequence == B_token.item())[0]) == 0:
            return True
        else:
            return check_as_before_bs(sequence) and check_more_as_than_bs(sequence)
    else:
        return (
            check_as_before_bs(sequence)
            and check_bs_before_cs(sequence)
            and check_same_number_as_bs(sequence)
            and check_more_bs_than_cs(sequence)
        )


def check_sequence_finished(sequence: torch.Tensor):
    """
    Check if the sequence is finished (EOS token is in the sequence)
    :param sequence:
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    # find the first EOS token
    return len(torch.where(sequence == EOS_token.item())[0]) > 0


def check_even_number_of_as(sequence: torch.Tensor):
    """
    Check if the sequence has even number of a's
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    num_as = torch.sum(sequence == A_token.item())

    return num_as % 2 == 0


def check_even_number_of_as_end(sequence: torch.Tensor):
    """
    Check if the sequence has even number of a's after the last b
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if len(a_tokens := torch.where(sequence == A_token.item())[0]) > 0:
        if len(b_tokens := torch.where(sequence == B_token.item())[0]) > 0:
            last_b = b_tokens[-1]
        else:
            last_b = torch.tensor(-1)
        num_as = torch.sum(sequence[last_b + 1 :] == A_token.item())
        return num_as % 2 == 0
    else:
        return True


def check_begins_with_b(sequence: torch.Tensor):
    """
    Check if the sequence begins with a B_token (after SOS)
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if sequence[0] == SOS_token.item():
        return sequence[1] == B_token.item()
    else:
        return sequence[0] == B_token.item()


def check_separated_brackets_and_parentheses(sequence: torch.Tensor):
    """
    Check if matched brackets only or matched parentheses only
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    stack = []
    if sequence[0] == SOS_token.item():
        start_token = sequence[1]
    else:
        start_token = sequence[0]

    if start_token in [OPENING_BRACKET_token.item(), CLOSING_BRACKET_token.item()]:
        for token in sequence:
            if token == OPENING_BRACKET_token.item():
                stack.append(token)
            elif token == CLOSING_BRACKET_token.item():
                if len(stack) == 0:
                    return False
                stack.pop()
            elif token == OPENING_PARENTHESIS_token.item():
                return False
            elif token == CLOSING_PARENTHESIS_token.item():
                return False
    else:
        for token in sequence:
            if token == OPENING_PARENTHESIS_token.item():
                stack.append(token)
            elif token == CLOSING_PARENTHESIS_token.item():
                if len(stack) == 0:
                    return False
                stack.pop()
            elif token == OPENING_BRACKET_token.item():
                return False
            elif token == CLOSING_BRACKET_token.item():
                return False

    return len(stack) == 0


def check_separated_brackets_and_parentheses_prompts(sequence: torch.Tensor):
    if sequence[0] == SOS_token.item():
        start_tokens = sequence[1:3]
    else:
        start_tokens = sequence[0:2]
    if (
        start_tokens[0] == OPENING_PARENTHESIS_token.item()
        and start_tokens[1] == OPENING_BRACKET_token.item()
    ):
        return False
    else:
        return True


def generate_test_prompts(length: int = 6, grammar: str = "aNbN"):
    """
    Generates all prompts of a given length with symbols a and b or (and c)
    :param length:
    :return:
    """

    num_samples = 2**length
    if grammar in ["aNbN", "abN", "aNbM", "aNbNaN", "baN"]:
        symbols = [A_token.item(), B_token.item()]
        prompts = torch.tensor(list(product(symbols, repeat=length)), dtype=torch.long)

        # add SOS
        prompts = torch.cat(
            (torch.ones((prompts.shape[0], 1), dtype=torch.long) * SOS_token, prompts),
            dim=1,
        )
    elif grammar == "aNbNcN":
        symbols = [A_token.item(), B_token.item(), C_token.item()]
        prompts = torch.tensor(list(product(symbols, repeat=length)), dtype=torch.long)

        # add SOS
        prompts = torch.cat(
            (torch.ones((prompts.shape[0], 1), dtype=torch.long) * SOS_token, prompts),
            dim=1,
        )
    elif grammar == "bbaN":
        ID_data = torch.tensor(
            np.array(
                generate_bNaM_grammar_data(
                    num_samples=num_samples // 2, max_length=length
                )
            ),
            dtype=torch.long,
        )
        OOD_data = torch.tensor(
            np.array(
                generate_bNaM_grammar_data(
                    num_samples=num_samples // 2, max_length=length - 1
                )
            ),
            dtype=torch.long,
        )
        id_prompts = torch.cat(
            (torch.ones((ID_data.shape[0], 1), dtype=torch.long) * SOS_token, ID_data),
            dim=1,
        )
        ood_prompts = torch.cat(
            (
                torch.ones((OOD_data.shape[0], 1), dtype=torch.long) * SOS_token,
                torch.ones((OOD_data.shape[0], 1), dtype=torch.long) * A_token,
                OOD_data,
            ),
            dim=1,
        )
        prompts = torch.cat((ood_prompts, id_prompts), dim=0)

    elif grammar == "parentheses":
        data = torch.tensor(
            np.array(
                generate_matched_parentheses_data(
                    num_samples=num_samples / 2, max_length=length, fixed_length=True
                )
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
            np.array(
                generate_matched_brackets_data(
                    num_samples=num_samples / 2, max_length=length, fixed_length=True
                )
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

    elif grammar == "separated_brackets_and_parentheses":
        # id prompts
        id_brackets_data = torch.tensor(
            np.array(
                generate_matched_brackets_data(
                    num_samples=num_samples // 4, max_length=length, fixed_length=True
                )
            ),
            dtype=torch.long,
        )
        id_parentheses_data = torch.tensor(
            np.array(
                generate_matched_parentheses_data(
                    num_samples=num_samples // 4, max_length=length, fixed_length=True
                )
            ),
            dtype=torch.long,
        )
        id_b_prompts = torch.cat(
            (
                id_brackets_data[:, 0].view(-1, 1),
                torch.ones((id_brackets_data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                torch.ones((id_brackets_data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                id_brackets_data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS
        id_p_prompts = torch.cat(
            (
                id_parentheses_data[:, 0].view(-1, 1),
                torch.ones((id_parentheses_data.shape[0], 1), dtype=torch.long)
                * OPENING_PARENTHESIS_token,
                torch.ones((id_parentheses_data.shape[0], 1), dtype=torch.long)
                * OPENING_PARENTHESIS_token,
                id_parentheses_data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS
        id_prompts = torch.cat((id_b_prompts, id_p_prompts), dim=0)

        # the ood prompts here include both [] and ()
        ood_data = torch.tensor(
            np.array(
                generate_matched_parentheses_and_brackets_data(
                    num_samples=num_samples // 2, max_length=length, fixed_length=True
                )
            ),
            dtype=torch.long,
        )
        ood_prompts = torch.cat(
            (
                ood_data[:, 0].view(-1, 1),
                torch.ones((ood_data.shape[0], 1), dtype=torch.long)
                * OPENING_PARENTHESIS_token,
                torch.ones((ood_data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                ood_data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        prompts = torch.cat((ood_prompts, id_prompts), dim=0)

    elif grammar == "parentheses_and_brackets":
        data = torch.tensor(
            np.array(
                generate_matched_parentheses_and_brackets_data(
                    num_samples=num_samples / 2, max_length=length, fixed_length=True
                )
            ),
            dtype=torch.long,
        )
        # generate torch 0-1 sequence in shape (data.shape[0], 1)
        ood_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_PARENTHESIS_token,
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
                * OPENING_PARENTHESIS_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
                data[:, 1:-1],
            ),
            dim=1,
        )  # remove EOS

        prompts = torch.cat((ood_prompts, id_prompts), dim=0)

    elif grammar == "not_nested_parentheses_and_brackets":
        data = torch.tensor(
            np.array(
                generate_not_nested_matched_parentheses_and_brackets_data(
                    num_samples=num_samples / 2, max_length=length, fixed_length=True
                )
            ),
            dtype=torch.long,
        )
        # generate torch 0-1 sequence in shape (data.shape[0], 1)
        ood_prompts = torch.cat(
            (
                data[:, 0].view(-1, 1),
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * CLOSING_PARENTHESIS_token,
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
                * OPENING_PARENTHESIS_token,
                torch.ones((data.shape[0], 1), dtype=torch.long)
                * OPENING_BRACKET_token,
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
    elif grammar == "aNbNcN":
        return lambda x: check_same_number_as_bs_cs(x) and check_as_before_bs_before_cs(
            x
        )
    elif grammar == "baN":
        return lambda x: check_even_number_of_as(x) and check_begins_with_b(x)
    elif grammar == "bbaN":
        return lambda x: check_even_number_of_as_end(x) and check_bs_before_as(x)
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
    elif grammar == "not_nested_parentheses_and_brackets":
        return lambda x: check_matched_parentheses(x) and check_matched_brackets(x)
    elif grammar == "separated_brackets_and_parentheses":
        return lambda x: check_separated_brackets_and_parentheses(x)
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
    elif grammar == "aNbNcN":
        return lambda x: check_in_dist_anbncn(x)
    elif grammar == "abN":
        return lambda x: True
    elif grammar == "baN":
        return lambda x: check_begins_with_b(x)
    elif grammar == "bbaN":
        return lambda x: check_begins_with_b(x)
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
    elif grammar == "not_nested_parentheses_and_brackets":
        return lambda x: check_matched_brackets(x) and check_matched_parentheses(x)
    elif grammar == "separated_brackets_and_parentheses":
        return lambda x: check_separated_brackets_and_parentheses_prompts(x)
    else:
        raise ValueError(f"Unknown grammar {grammar}")


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


def generate_not_nested_matched_parentheses_and_brackets(n):
    """
    Generate a word of length n with paired () and [].
    """
    if n == 0:
        return np.concatenate((SOS_token, EOS_token))
    elif n % 2 == 1:
        raise ValueError("Length can only be even")
    else:
        word = []
        stack_b = []
        stack_p = []
        while len(word) < n:  # Each pair of parentheses or brackets adds 2 characters
            if len(stack_b) + len(stack_p) == 0:
                choice = random.choice(
                    [OPENING_PARENTHESIS_token, OPENING_BRACKET_token]
                )
            elif len(stack_b) == 0 and len(stack_p) != 0:
                choice = random.choice(
                    [
                        OPENING_BRACKET_token,
                        OPENING_PARENTHESIS_token,
                        CLOSING_PARENTHESIS_token,
                    ]
                )
                if len(word) + len(stack_p) >= n:
                    choice = CLOSING_PARENTHESIS_token
            elif len(stack_b) != 0 and len(stack_p) == 0:
                choice = random.choice(
                    [
                        OPENING_PARENTHESIS_token,
                        OPENING_BRACKET_token,
                        CLOSING_BRACKET_token,
                    ]
                )
                if len(word) + len(stack_b) >= n:
                    choice = CLOSING_BRACKET_token
            else:
                choice = random.choice(
                    [
                        OPENING_PARENTHESIS_token,
                        OPENING_BRACKET_token,
                        CLOSING_BRACKET_token,
                        CLOSING_PARENTHESIS_token,
                    ]
                )
                if len(word) + len(stack_b) + len(stack_p) >= n:
                    choice = random.choice(
                        [CLOSING_BRACKET_token, CLOSING_PARENTHESIS_token]
                    )

            if choice == OPENING_PARENTHESIS_token:
                word.append(OPENING_PARENTHESIS_token)
                stack_p.append(OPENING_PARENTHESIS_token)
            elif choice == OPENING_BRACKET_token:
                word.append(OPENING_BRACKET_token)
                stack_b.append(OPENING_BRACKET_token)
            elif choice == CLOSING_PARENTHESIS_token:
                word.append(CLOSING_PARENTHESIS_token)
                stack_p.pop()
            elif choice == CLOSING_BRACKET_token:
                word.append(CLOSING_BRACKET_token)
                stack_b.pop()

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


def check_begins_with_opening_parenthesis(sequence: torch.Tensor) -> bool:
    """
    Check if begins with opening parenthesis
    :return:
    """
    if type(sequence) == np.ndarray:
        sequence = torch.from_numpy(sequence)

    if sequence[0] == SOS_token.item():
        return sequence[1] == OPENING_PARENTHESIS_token.item()
    else:
        return sequence[0] == OPENING_PARENTHESIS_token.item()


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


def generate_matched_parentheses_and_matched_brackets_data(
    num_samples: int, max_length: int = 32, fixed_length=False
) -> list:
    if fixed_length is False:
        lengths_p = np.random.randint(
            low=1, high=max_length // 2 + 1, size=num_samples // 2
        )
        lengths_b = np.random.randint(
            low=1, high=max_length // 2 + 1, size=num_samples // 2
        )
        data = [generate_matched_parentheses(2 * l) for l in lengths_p]
        data.extend(generate_matched_brackets(2 * l) for l in lengths_b)
    else:
        data = []
        while len(data) < num_samples // 2:
            sample = generate_matched_parentheses(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)
        while len(data) < num_samples:
            sample = generate_matched_brackets(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)

    np.random.shuffle(data)
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


def generate_not_nested_matched_parentheses_and_brackets_data(
    num_samples: int, max_length: int = 32, fixed_length=False
) -> list:
    """

    :param num_samples: number of samples
    :param max_length: maximum sequence length (inclusive SOS and EOS tokens)
    :return: list of length num_samples with maximal sequences of length max_length
    """

    if fixed_length is False:
        lengths = np.random.randint(low=1, high=max_length // 2 + 1, size=num_samples)
        data = [
            generate_not_nested_matched_parentheses_and_brackets(2 * l) for l in lengths
        ]
    else:
        data = []
        while len(data) < num_samples:
            sample = generate_matched_parentheses_and_brackets(max_length)
            if len(sample) == (max_length + 2):  # +SOS, EOS
                data.append(sample)

    return data
