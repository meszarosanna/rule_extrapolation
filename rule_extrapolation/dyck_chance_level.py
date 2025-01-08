from itertools import product
import numpy as np
import torch

from rule_extrapolation.data import check_matched_parentheses, check_matched_brackets

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

n = 20
symbols = [
    OPENING_PARENTHESIS_token.item(),
    CLOSING_PARENTHESIS_token.item(),
    OPENING_BRACKET_token.item(),
    CLOSING_BRACKET_token.item(),
]
count_id_r1 = 0
count_id_r2 = 0
count_ood_r1 = 0
count_ood_r2 = 0
sum_of_sec = 0

for i in range(n):
    sum_of_sec += 4**i

    data = torch.tensor(list(product(symbols, repeat=i)), dtype=torch.long)
    count_ood_r2 += sum(list(map(check_matched_parentheses, data)))

    id_prompts = torch.cat(
        (
            torch.ones((data.shape[0], 1), dtype=torch.long)
            * OPENING_PARENTHESIS_token,
            torch.ones((data.shape[0], 1), dtype=torch.long) * OPENING_BRACKET_token,
            data,
        ),
        dim=1,
    )
    id_prompts.tolist()
    count_id_r1 += sum(list(map(check_matched_brackets, id_prompts)))
    count_id_r2 += sum(list(map(check_matched_parentheses, id_prompts)))

    ood_prompts = torch.cat(
        (
            torch.ones((data.shape[0], 1), dtype=torch.long)
            * CLOSING_PARENTHESIS_token,
            torch.ones((data.shape[0], 1), dtype=torch.long) * OPENING_BRACKET_token,
            data,
        ),
        dim=1,
    )
    ood_prompts.tolist()
    count_ood_r1 += sum(list(map(check_matched_brackets, ood_prompts)))


print(count_id_r1 / sum_of_sec)
print(count_id_r2 / sum_of_sec)
print(count_ood_r1 / sum_of_sec)
print(count_ood_r2 / sum_of_sec)
