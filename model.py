import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for queries
    n_kv_heads: Optional[int] = None # number of heads for the key and values, grouped/multi-query attention
    vocab_size: int = -1 # will be set when we load tokenizer
    # with grouped query attention heads, the number of parameters of the architecture reduces
    # hence to maintain the same number of model params, we use the below two args to retain original number of params
    # to ensure fair comparison b/w diff model archs of same no of params
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # parameters for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 1e4):
    assert head_dim % 2 == 0, "The hidden dimension must be even (atleast in the attention head - my note)"
    # theta_i = theta ** -2 * (i-1)/d
    # i E [1, 2, ... d/2] => (i-1) E [0, 1, 2, ... d/2 - 1]
    exponents = -2.0 * torch.arange(0, head_dim/2).float() / head_dim
    theta_i = (theta ** exponents).to(device) # shape (head_dim/2)

    # # my implementation - only suitable when indexing for single token position from final list, for multiple token positions will require for loop - slow
    # theta_i = (theta ** exponents).reshape(-1, 1) # shape (head_dim/2)
    # theta_i_duplicate = torch.concat([theta_i, theta_i], dim=-1).reshape(-1).to(device)
    # freqs_complex = [(None, None)] * seq_len
    # for m in range(seq_len):
    #     cos_comp = torch.cos(m * theta_i_duplicate)
    #     sin_comp = torch.sin(m * theta_i_duplicate)
    #     freqs_complex[m] = (cos_comp, sin_comp)
    # return freqs_complex

    m = torch.arange(seq_len, device=device)
    # shape: (seq_len) *outer_product (head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float() #
    # euler's formula e^(ix) = cos(x) + isin(x)
    # we can compute complex numbers in polar form c = R * exp(i * m * theta), where R = 1
    # (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(abs = torch.ones_like(freqs), angle=freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # need to precompute RoPE frequencies before hand
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device) # Shape: (seq_len, dim/2)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # input: (batch_size, seq_len)
        # to make use of KV-cache, the sequence length is always 1
        # note the KV cache is only used during inference and not during training
        # KV cache suboptimal for training
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed during an inference forward pass"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output

