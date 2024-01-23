import math

import torch
from torch import nn as nn
import torch.nn.functional as F


from llm_non_identifiability.data import PAD_token


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]  # type: ignore[index]
        )


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


class TransformerDecoder(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_decoder_layers,
        dropout_p,
        dim_feedforward,
        layer_norm_eps,
        relu_rescale: float = 0.0,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.relu_rescale = relu_rescale

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)

        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout_p,
            dim_feedforward=dim_feedforward,
            layer_norm_eps=layer_norm_eps,
        )

        if self.relu_rescale > 0:
            layer.activation = (
                lambda x: F.relu(x * self.relu_rescale) / self.relu_rescale
            )

        self.decoder = nn.TransformerEncoder(layer, num_decoder_layers)

        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.decoder(
            src=src,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        out = self.out(transformer_out)

        return out
