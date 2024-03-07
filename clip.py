import torch
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) # (vocab_size, dimension, longest_sentence)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens):
        # tokens: (B, seqlen)
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        # state: (B, seqlen, dim) 
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_tokens) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # tokens: (B, seqlen)   # seqlen is constant (n_tokens) because of padding
        x = self.token_embedding(tokens)
        # x: (B, seqlen, n_embed)
        x += self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(n_embed)
        self.lnorm2 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x):
        # x: (B, seqlen, Dim)
        residue = x
        x = self.lnorm1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.lnorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        x = self.linear2(x)
        x += residue

        return x 
