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
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

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

