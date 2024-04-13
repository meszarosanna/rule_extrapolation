import torch
from torch import nn as nn

from llm_non_identifiability.model import create_pad_mask


class LinearLLM(nn.Module):
    def __init__(
        self,
        max_data_length: int = 256,
        num_tokens=5,
        embedding_dim: int = 32,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.max_data_length = max_data_length
        self.num_tokens = num_tokens
        self.device = device
        self.embedding = nn.Embedding(num_tokens, embedding_dim)

        # Weight matrix; +1 because the input has a SOS token at the beginning
        self.weight = torch.nn.Parameter(
            torch.empty(
                (max_data_length + 1, embedding_dim, max_data_length + 1, num_tokens),
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

    def forward(self, src, apply_pad_mask: bool = True):
        src = self.embedding(src)
        if src.shape[1] != (self.max_data_length + 1):
            zeros_tensor = torch.zeros(
                src.shape[0],
                self.max_data_length + 1 - src.shape[1],
                src.shape[2],
                device=self.device,
            )
            src = torch.cat((src, zeros_tensor), dim=1)

        if apply_pad_mask:
            src = (
                src * create_pad_mask(src).logical_not()
            )  # logical not needed as we want to mask the pad tokens

        out = torch.einsum("bsw,swtv,st->btv", src.float(), self.weight, self.mask)
        if self.bias != None:
            out = out + self.bias[None, :, :]
        return out.permute(0, 2, 1)
