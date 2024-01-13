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
    freqs = torch.outer(m, theta_i).float() #
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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x:torch.Tensor):
        # (B, seq_len, dim)
        # rsqrt(x) = 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor):
        # (dim) * (B, seq_len, dim) = (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len_kvcache, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(batch_size, seq_len_kvcache, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len_kvcache, n_kv_heads * n_rep, head_dim)

class SelfAttention(nn.Module):
    """Self attention with Grouped Query Attention and KV Cache. Seq_len = 1, for inference
    """    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads

        # Indicates how many times the key and value heads need to be repeated to match the query heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_q_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        # seq_len should ideally be 1

        # (B, seq_len, dim) -> (B, seq_len, n_qh * head_dim)
        q = self.wq(x)
        # (B, seq_len, dim) -> (B, seq_len, n_kvh * head_dim)
        k = self.wk(x)
        v = self.wv(x)

        # now need to apply rotary position embeddings to query and key vectors
        # (B, seq_len, n_qh * head_dim) -> (B, seq_len, n_qh * head_dim)
        q = apply_rotary_embeddings(q, freqs_complex, device=x.device)
        k = apply_rotary_embeddings(k, freqs_complex, device=x.device)

        # (B, seq_len, n_qh * head_dim)  -> (B, seq_len, n_qh, head_dim)
        q = q.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, seq_len, n_kvh * head_dim)  -> (B, seq_len, n_kvh, head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Replace the entry in the KV cache for this token
        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = k
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = v

        # Shape: (B, seq_len_kv, n_kvh, head_dim)
        keys = self.cache_k[:batch_size, 0: start_pos+seq_len]
        values = self.cache_v[:batch_size, 0: start_pos+seq_len]

        # n_kvh may not be equal to n_qh, hence we would need to repeat the keys and values self.n_rep times
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, seq_len, n_qh, head_dim) -> (B, n_qh, seq_len, head_dim)
        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, n_qh, seq_len, head_dim) @ (B, n_qh, head_dim, seq_len) -> (B, n_qh, seq_len, seq_len)  
        attention_scores = F.softmax(torch.matmul(q, keys.transpose(2, 3))/torch.sqrt(self.head_dim), dim=-1).type_as(q)
        # (B, n_qh, seq_len, seq_len) @ (B, n_qh, seq_len, head_dim)  -> (B, n_qh, seq_len, head_dim)
        out = torch.matmul(attention_scores, values)
    
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (B, seq_len, dim) -> (B, seq_len, dim)
        return self.wo(out)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden dim to th nearest (next) multiple of the args.multiple_of param
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1)//args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x)) # silu(x) = x * sigmoid(x); swish(x) = x*sigmoid(beta*x)
        x_V = self.w3(x)
        x = swish * x_V # element wise multiplication
        x = self.w2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before attention
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # normalization before feedforward
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim)
        # 0. Input
        # 1. Pre attention RMS norm  
        # 2. Self Attention with RoPE
        # 3. Add residual from step 0.
        # 4. Pre feedforward RMS norm
        # 5. SwiGLU
        # 6. Feedforward
        # 7. Add residual from step 3.

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



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
            self.layers.append(TransformerBlock(args))
        
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
        output = self.output(h).float() # just outputs logits, did not perform softmax yet 
        return output

