import math

import torch
from torch import nn as nn
import torch.nn.functional as F


from llm_non_identifiability.data import PAD_token


def get_tgt_mask(size, device) -> torch.Tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.float))
    mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask


def create_pad_mask(
    matrix: torch.Tensor, pad_token: int = PAD_token.item()
) -> torch.Tensor:
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    return torch.as_tensor(matrix == pad_token, device=matrix.device)


class LinearLLM(nn.Module):
    def __init__(
        self,
        max_data_length: int = 256,
        num_tokens=5,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.max_data_length = max_data_length
        self.num_tokens = num_tokens
        self.device = device
        # Weight matrix; +1 because the input has a SOS token at the beginning
        self.weight = torch.nn.Parameter(
            torch.empty(
                (max_data_length + 1, num_tokens, max_data_length + 1, num_tokens),
                **factory_kwargs
            )
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((max_data_length + 1, num_tokens), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.mask = torch.tril(
            torch.ones(
                max_data_length + 1,
                max_data_length + 1,
                device=device,
                dtype=torch.float,
            )
        )
        self.mask.to(device)

    def reset_parameters(self):
        # Initialize parameters as desired
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, src):
        src = torch.nn.functional.one_hot(src, num_classes=self.num_tokens)
        if src.shape[1] != (self.max_data_length + 1):
            zeros_tensor = torch.zeros(
                src.shape[0],
                self.max_data_length + 1 - src.shape[1],
                src.shape[2],
                device=self.device,
            )
            src = torch.cat((src, zeros_tensor), dim=1)
        out = torch.einsum("bsw,swtv,st->btv", src.float(), self.weight, self.mask)
        if self.bias != None:
            out = out + self.bias[None, :, :]
        return out
