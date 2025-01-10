import numpy as np
import pytest
import torch

from rule_extrapolation.data import check_as_before_bs, check_same_number_as_bs
from rule_extrapolation.data import (
    generate_aNbN_grammar_data,
    generate_abN_grammar_data,
    generate_baN_grammar_data,
    generate_bbaN_grammar_data,
    generate_aNbM_grammar_data,
    generate_aNbNaN_grammar_data,
    pad,
    PAD_token,
    check_sequence_finished,
    check_twice_many_as_than_bs,
    check_bs_in_the_middle,
    check_bs_together,
    check_more_as_before_bs,
    check_bs_before_as,
    check_as_before_cs,
    check_bs_before_cs,
    check_more_bs_than_cs,
    check_same_number_as_bs_cs,
    check_as_before_bs_before_cs,
    check_in_dist_anbncn,
    check_even_number_of_as,
    check_even_number_of_as_end,
    check_begins_with_b,
    EOS_token,
    SOS_token,
    generate_test_prompts,
    grammar_rules,
    generate_matched_parentheses_and_brackets_data,
    generate_not_nested_matched_parentheses_and_brackets_data,
    generate_matched_parentheses_and_matched_brackets_data,
    generate_matched_brackets_data,
    generate_matched_parentheses_data,
    generate_aNbNcN_grammar_data,
    A_token,
    B_token,
    C_token,
)

A = A_token.item()
B = B_token.item()
C = C_token.item()


def test_aNbN_grammar_equal_as_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length, all_sequences=False)
    for sequence in sequences:
        assert check_same_number_as_bs(sequence)


def test_aNbN_grammar_as_before_bs(num_samples, max_length):
    sequences = generate_aNbN_grammar_data(num_samples, max_length, all_sequences=False)
    for sequence in sequences:
        assert check_as_before_bs(sequence)


def test_baN_even_number_of_as(num_samples, max_length):
    sequences = generate_baN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        assert check_even_number_of_as(sequence)


def test_baN_check_begins_with_b(num_samples, max_length):
    sequences = generate_baN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        assert check_begins_with_b(sequence)


def test_bbaN_even_number_of_as(num_samples, max_length):
    sequences = generate_bbaN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        assert check_even_number_of_as(sequence)


def test_bbaN_bs_before_as(num_samples, max_length):
    sequences = generate_bbaN_grammar_data(num_samples, max_length)
    for sequence in sequences:
        assert check_even_number_of_as(sequence)


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
        num_a = np.sum(sequence == A)
        num_b = np.sum(sequence == B)
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
    sequence = torch.tensor([A, A, B, A, B])
    assert check_as_before_bs(sequence) == False

    sequence = torch.tensor([A, A, B, B])
    assert check_as_before_bs(sequence) == True

    sequence = torch.tensor([B, B])
    assert check_as_before_bs(sequence) == True

    sequence = torch.tensor([A, A])
    assert check_as_before_bs(sequence) == True


def test_check_more_as_before_bs():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_more_as_before_bs(sequence) == True

    sequence = torch.tensor([A, B, A, B, A, A])
    assert check_more_as_before_bs(sequence) == False

    sequence = torch.tensor([A, A, B, B])
    assert check_more_as_before_bs(sequence) == True

    sequence = torch.tensor([B, B])
    assert check_more_as_before_bs(sequence) == False

    sequence = torch.tensor([A, A])
    assert check_more_as_before_bs(sequence) == True


def test_check_more_bs_than_cs():
    sequence = torch.tensor([B, B, C, C])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([C, B, B])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([A, A])
    assert check_more_bs_than_cs(sequence) == True

    sequence = torch.tensor([A, C, A, C, A, B])
    assert check_more_bs_than_cs(sequence) == False

    sequence = torch.tensor([A, A, C])
    assert check_more_bs_than_cs(sequence) == False


def test_check_bs_together():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_bs_together(sequence) == False

    sequence = torch.tensor([A, A, B, B, A, A])
    assert check_bs_together(sequence) == True

    sequence = torch.tensor([B, B])
    assert check_bs_together(sequence) == True

    sequence = torch.tensor([A, A])
    assert check_bs_together(sequence) == False


def test_check_bs_in_the_middle():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_bs_in_the_middle(sequence) == False

    sequence = torch.tensor([A, A, B, B, A, A])
    assert check_bs_in_the_middle(sequence) == True

    sequence = torch.tensor([B, B])
    assert check_bs_in_the_middle(sequence) == True

    sequence = torch.tensor([A, A])
    assert check_bs_in_the_middle(sequence) == False


def test_check_same_number_as_bs():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_same_number_as_bs(sequence) == False

    sequence = torch.tensor([A, A, B, B])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([A, B, B, A])
    assert check_same_number_as_bs(sequence) == True

    sequence = torch.tensor([B, B, A, A])
    assert check_same_number_as_bs(sequence) == True


def test_check_twice_many_as_than_bs():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_twice_many_as_than_bs(sequence) == False

    sequence = torch.tensor([A, A, B, B, A, A])
    assert check_twice_many_as_than_bs(sequence) == True

    sequence = torch.tensor([B, A, A, A, A, B])
    assert check_twice_many_as_than_bs(sequence) == True


def test_check_in_distr_anbncn():
    sequence = torch.tensor([A, A, B, A, B])
    assert check_in_dist_anbncn(sequence) == False

    sequence = torch.tensor([A, A, B, B])
    assert check_in_dist_anbncn(sequence) == True

    sequence = torch.tensor([A, A, A, B, B, B, C])
    assert check_in_dist_anbncn(sequence) == True

    sequence = torch.tensor([C, C, A, B])
    assert check_in_dist_anbncn(sequence) == False


def test_check_sequence_finished():
    sequence = torch.tensor([A, B, A, B])
    assert check_sequence_finished(sequence) == False

    sequence = torch.tensor([A, B, EOS_token.item(), A, B])
    assert check_sequence_finished(sequence) == True

    sequence = torch.tensor([A, B, EOS_token.item(), A, B, EOS_token.item(), A, B])
    assert check_sequence_finished(sequence) == True

    sequence = torch.tensor([A, B, EOS_token.item()])
    assert check_sequence_finished(sequence) == True

    sequence = torch.tensor(
        [A, B, EOS_token.item(), EOS_token.item(), PAD_token.item()]
    )
    assert check_sequence_finished(sequence) == True


def test_check_even_number_of_as():
    sequence = torch.Tensor([B, A, A])
    assert check_even_number_of_as(sequence) == True

    sequence = torch.Tensor([SOS_token.item(), A, B, A, EOS_token.item()])
    assert check_even_number_of_as(sequence) == True

    sequence = torch.Tensor([SOS_token.item(), A, A, B, A])
    assert check_even_number_of_as(sequence) == False

    sequence = torch.Tensor([A, B, B, B])
    assert check_even_number_of_as(sequence) == False


def test_check_even_number_of_as_end():
    sequence = torch.Tensor([B, A, A])
    assert check_even_number_of_as_end(sequence) == True

    sequence = torch.Tensor([SOS_token.item(), A, B, A, EOS_token.item()])
    assert check_even_number_of_as_end(sequence) == False

    sequence = torch.Tensor([SOS_token.item(), A, A, B, A])
    assert check_even_number_of_as_end(sequence) == False

    sequence = torch.Tensor([A, A, A, A])
    assert check_even_number_of_as_end(sequence) == True

    sequence = torch.Tensor([B, B, B, B])
    assert check_even_number_of_as_end(sequence) == True


def test_check_begins_with_b():
    sequence = torch.Tensor([B, A, A])
    assert check_begins_with_b(sequence) == True

    sequence = torch.Tensor([SOS_token.item(), B, A, B, A, EOS_token.item()])
    assert check_begins_with_b(sequence) == True

    sequence = torch.Tensor([SOS_token.item(), A, A, B, A])
    assert check_begins_with_b(sequence) == False

    sequence = torch.Tensor([A, B, B, B])
    assert check_begins_with_b(sequence) == False


@pytest.mark.parametrize(
    "grammar",
    [
        "aNbN",
        "abN",
        "baN",
        "bbaN",
        "aNbM",
        "aNbNcN",
        "parentheses",
        "brackets",
        "parentheses_and_brackets",
        "not_nested_parentheses_and_brackets",
        "separated_brackets_and_parentheses",
    ],
)
def test_generate_test_prompts(grammar):
    length = 8
    prompts = generate_test_prompts(length, grammar=grammar)

    if grammar in ["aNbN", "abN", "aNbM", "baN", "bbaN"]:
        assert prompts.shape == (2**length, length + 1)
    elif grammar == "aNbNcN":
        assert prompts.shape == (3**length, length + 1)
    elif grammar in [
        "parentheses",
        "brackets",
        "parentheses_and_brackets",
        "not_nested_parentheses_and_brackets",
        "separated_brackets_and_parentheses",
    ]:
        assert prompts.shape == (2**length, length + 3)


@pytest.mark.parametrize(
    "grammar",
    [
        "aNbN",
        "abN",
        "baN",
        "bbaN",
        "aNbM",
        "aNbNcN",
        "parentheses",
        "brackets",
        "parentheses_and_brackets",
        "not_nested_parentheses_and_brackets",
        "separated_brackets_and_parentheses",
    ],
)
def test_grammar_rules(max_length, grammar, num_samples):
    rules = grammar_rules(grammar)

    if grammar == "aNbN":
        data = torch.from_numpy(
            pad(
                generate_aNbN_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()
    elif grammar == "abN":
        data = torch.from_numpy(
            pad(
                generate_abN_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()
    elif grammar == "baN":
        data = torch.from_numpy(
            pad(
                generate_baN_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()
    elif grammar == "bbaN":
        data = torch.from_numpy(
            pad(
                generate_bbaN_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()
    elif grammar == "aNbM":
        data = torch.from_numpy(
            pad(
                generate_aNbM_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()
    elif grammar == "aNbNcN":
        data = torch.from_numpy(
            pad(
                generate_aNbNcN_grammar_data(
                    num_samples=num_samples, max_length=max_length
                )
            )
        ).long()

    elif grammar == "parentheses":
        data = torch.from_numpy(
            pad(generate_matched_parentheses_data(max_length))
        ).long()
    elif grammar == "brackets":
        data = torch.from_numpy(pad(generate_matched_brackets_data(max_length))).long()

    elif grammar == "parentheses_and_brackets":
        data = torch.from_numpy(
            pad(generate_matched_parentheses_and_brackets_data(max_length))
        ).long()
    elif grammar == "not_nested_parentheses_and_brackets":
        data = torch.from_numpy(
            pad(generate_not_nested_matched_parentheses_and_brackets_data(max_length))
        ).long()
    elif grammar == "separated_brackets_and_parentheses":
        data = torch.from_numpy(
            pad(generate_matched_parentheses_and_matched_brackets_data(max_length))
        ).long()

    assert torch.all(torch.tensor([rules(d) for d in data]))


from rule_extrapolation.data import (
    generate_matched_brackets,
    generate_matched_parentheses,
    generate_matched_parentheses_and_brackets,
    generate_not_nested_matched_parentheses_and_brackets,
    check_matched_parentheses,
    check_matched_brackets,
    check_matched_parentheses_and_brackets,
    check_separated_brackets_and_parentheses,
    check_separated_brackets_and_parentheses_prompts,
    OPENING_BRACKET_token,
    OPENING_PARENTHESIS_token,
    CLOSING_PARENTHESIS_token,
    CLOSING_BRACKET_token,
)


def test_generate_matched_brackets():
    generate_matched_brackets(40)


def test_generate_matched_parentheses():
    generate_matched_parentheses(40)


def test_generate_matched_parentheses_and_brackets():
    generate_matched_parentheses_and_brackets(40)


def test_generate_not_nested_matched_parentheses_and_brackets():
    generate_not_nested_matched_parentheses_and_brackets(40)


def test_check_matched_parentheses():
    op = OPENING_PARENTHESIS_token.item()
    cp = CLOSING_PARENTHESIS_token.item()
    random_token = op + cp
    sequence = torch.tensor([op, cp, op, cp, op, cp])  # ()()()
    assert check_matched_parentheses(sequence) == True

    sequence = torch.tensor([op, cp, op, op, cp, cp])  # ()(())
    assert check_matched_parentheses(sequence) == True

    sequence = torch.tensor([op, cp, cp, op, op, cp])  # ())(()
    assert check_matched_parentheses(sequence) == False

    sequence = torch.tensor([op, cp, cp, op, op, cp, cp, op])  # ())(())(
    assert check_matched_parentheses(sequence) == False

    sequence = torch.tensor([op, cp, cp, op, op, cp, cp, op, cp, cp])  # ())(())(())
    assert check_matched_parentheses(sequence) == False

    sequence = torch.tensor(
        [op, random_token, cp, op, cp, random_token, op, cp]
    )  # (X)()X()
    assert check_matched_parentheses(sequence) == True


def test_check_matched_brackets():
    ob = OPENING_BRACKET_token.item()
    cb = CLOSING_BRACKET_token.item()
    random_token = ob + cb
    sequence = torch.tensor([ob, cb, ob, cb, ob, cb])  # [][][]
    assert check_matched_brackets(sequence) == True

    sequence = torch.tensor([ob, cb, ob, ob, cb, cb])  # [][[]]
    assert check_matched_brackets(sequence) == True

    sequence = torch.tensor(
        [ob, cb, random_token, ob, random_token, ob, cb, cb]
    )  # []X[X[]]
    assert check_matched_brackets(sequence) == True

    sequence = torch.tensor([ob, cb, cb, ob, ob, cb])  # []][[]
    assert check_matched_brackets(sequence) == False

    sequence = torch.tensor([ob, cb, cb, ob, ob, cb, cb, ob])  # []][[]][
    assert check_matched_brackets(sequence) == False

    sequence = torch.tensor([ob, cb, cb, ob, ob, cb, cb, ob, cb, cb])  # []][[]][]]
    assert check_matched_brackets(sequence) == False


def test_check_matched_parentheses_and_brackets():
    op = OPENING_PARENTHESIS_token.item()
    cp = CLOSING_PARENTHESIS_token.item()
    ob = OPENING_BRACKET_token.item()
    cb = CLOSING_BRACKET_token.item()

    sequence = torch.tensor([op, cp, ob, cb, op, cp, ob, cb])  # ()[]()[]
    assert check_matched_parentheses_and_brackets(sequence) == True

    sequence = torch.tensor([op, cp, ob, cb, op, cp, ob, ob, cb, cb])  # ()[]()[[]]
    assert check_matched_parentheses_and_brackets(sequence) == True

    sequence = torch.tensor([op, cp, ob, cb, cp, op, ob, cb])  # ()[])([]
    assert check_matched_parentheses_and_brackets(sequence) == False

    sequence = torch.tensor([op, cp, ob, cb, cp, op, ob, cb, cp, op])  # ()[])([])(
    assert check_matched_parentheses_and_brackets(sequence) == False

    sequence = torch.tensor(
        [op, cp, ob, cb, cp, op, ob, cb, cp, op, cp, cp]
    )  # ()[])([])())
    assert check_matched_parentheses_and_brackets(sequence) == False


def test_check_not_nested_matched_parentheses_and_brackets():
    op = OPENING_PARENTHESIS_token.item()
    cp = CLOSING_PARENTHESIS_token.item()
    ob = OPENING_BRACKET_token.item()
    cb = CLOSING_BRACKET_token.item()

    sequence = torch.tensor([op, ob, cp, cb, op, cp])  # ([)]()
    assert (
        check_matched_parentheses(sequence) == True
        and check_matched_brackets(sequence) == True
    )

    sequence = torch.tensor([ob, op, ob, cp, cb, cb])  # [([)]]
    assert (
        check_matched_parentheses(sequence) == True
        and check_matched_brackets(sequence) == True
    )

    sequence = torch.tensor([op, cp, ob, cb, op, cp, ob, ob, cb, cb])  # ()[]()[[]]
    assert (
        check_matched_parentheses(sequence) == True
        and check_matched_brackets(sequence) == True
    )

    sequence = torch.tensor([op, cp, ob, cb, cp, op, ob, cb])  # ()[])([]
    assert (
        check_matched_parentheses(sequence) == False
        or check_matched_brackets(sequence) == False
    )

    sequence = torch.tensor([op, cp, ob, cb, cp, op, ob, cb, cp, op])  # ()[])([])(
    assert (
        check_matched_parentheses(sequence) == False
        or check_matched_brackets(sequence) == False
    )

    sequence = torch.tensor(
        [op, cp, ob, cb, cp, op, ob, cb, cp, op, cp, cp]
    )  # ()[])([])())
    assert (
        check_matched_parentheses(sequence) == False
        or check_matched_brackets(sequence) == False
    )


def test_check_separated_brackets_and_parentheses():
    op = OPENING_PARENTHESIS_token.item()
    cp = CLOSING_PARENTHESIS_token.item()
    ob = OPENING_BRACKET_token.item()
    cb = CLOSING_BRACKET_token.item()

    sequence = torch.tensor([op, op, cp, op, cp, cp])  # (()())
    assert check_separated_brackets_and_parentheses(sequence) == True

    sequence = torch.tensor([SOS_token.item(), op, op, cp, op, cp, cp])  # (()())
    assert check_separated_brackets_and_parentheses(sequence) == True

    sequence = torch.tensor([ob, ob, cb, ob, cb, cb])  # [[][]]
    assert check_separated_brackets_and_parentheses(sequence) == True

    sequence = torch.tensor([SOS_token.item(), ob, ob, cb, ob, cb, cb])  # [[][]]
    assert check_separated_brackets_and_parentheses(sequence) == True

    sequence = torch.tensor([op, ob, op, cp, op, cp, ob, cb, cb, cp])  # ([()()[]])
    assert check_separated_brackets_and_parentheses(sequence) == False

    sequence = torch.tensor(
        [SOS_token.item(), op, ob, op, cp, op, cp, ob, cb, cb, cp]
    )  # ([()()[]])
    assert check_separated_brackets_and_parentheses(sequence) == False


def test_check_separated_brackets_and_parentheses_prompts():
    op = OPENING_PARENTHESIS_token.item()
    cp = CLOSING_PARENTHESIS_token.item()
    ob = OPENING_BRACKET_token.item()
    cb = CLOSING_BRACKET_token.item()

    sequence = torch.tensor([op, op, cp, op, cp])  # (()()
    assert check_separated_brackets_and_parentheses_prompts(sequence) == True

    sequence = torch.tensor([SOS_token.item(), op, op, cp, op, cp])  # (()()
    assert check_separated_brackets_and_parentheses_prompts(sequence) == True

    sequence = torch.tensor([ob, ob, cb, ob, cb])  # [[][]
    assert check_separated_brackets_and_parentheses_prompts(sequence) == True

    sequence = torch.tensor([SOS_token.item(), ob, ob, cb, ob, cb])  # [[][]
    assert check_separated_brackets_and_parentheses_prompts(sequence) == True

    sequence = torch.tensor([op, ob, op, cp, op, cp, ob, cb])  # ([()()[]
    assert check_separated_brackets_and_parentheses_prompts(sequence) == False

    sequence = torch.tensor(
        [SOS_token.item(), op, ob, op, cp, op, cp, ob, cb]
    )  # ([()()[]
    assert check_separated_brackets_and_parentheses_prompts(sequence) == False
