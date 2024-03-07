import torch
import torch.nn as nn   
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()
        assert d_embed % n_heads == 0
        # Wq, Wk, Wv matrices
        self.q   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)   

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: (B, SeqLen, Dim)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interm_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # q, k, v: (B, seq_len, dim)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)
        # q, k, v: (B, H, seq_len, d_head)

        weight = q @ k.transpose(-1, -2)
        # weight: (B, H, seq_len, seq_len)


        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1 (True)
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        # output: (B, H, seq_len, d_head)
        output = output.transpose(1,2)
        # output: (B, seqlen, H, d_head)
        output = output.reshape(input_shape)
        # output: (B, seqlen, d_embed)
        output = self.out_proj(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        assert d_embed % n_heads == 0
        # Wq, Wk, Wv matrices
        self.q   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # latent x: (B, seqlen, dimQ)
        # context y: (B, context_seqlen, context_dim) = (B, longest_setentence_length, 768) (for clip)
        input_shape = x.shape
        batch_size, seq_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q(x)
        # q: (B, seqlen, dimL)
        k = self.k(y)
        v = self.v(y)
        # k,v: (B, seqlenC, dimQ)

        q = q.view(interim_shape).transpose(1,2)
        # q: (B, H, seqlen, d_head) ; n_head = dimQ/H
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        # k,v: (B, H, seqlenContext, d_head)

        weight = q @ k.transpose(-1, -2)
        # weight: (B, H, seqLenLatent, seqlenContext)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        # output: (B, H, seqlenLatent, d_head)
        output = output.transpose(1, 2).contiguous()
        # output: (B, seqlenLatent, H, d_head)
        output = output.view(input_shape)
        # output: (B, seqlenLatent, dimLatent)
        output = self.out_proj(output)
        return output


