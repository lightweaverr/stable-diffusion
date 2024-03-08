import torch
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention






class TimeEmbedding(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_embed, n_embed * 4)
        self.linear2 = nn.Linear(n_embed * 4, n_embed * 4)

    def forward(self, x):
        # x: (1, n_embed)
        x = self.linear1(x)
        # x: (1, 4 * n_embed)
        x = F.silu(x)
        x = self.linear2(x)
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, time):
        # x: (B, in_channels, height, width)
        # time: (1, 1280)
        residue = x
        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)
        # x: (B, out_channels, height, width)
        time = F.silu(time)
        time = self.linear_time(time)
        # time: (1, out_channels)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        merged += self.residual_layer(residue)
        return merged

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embed, d_context=768) -> None:
        super().__init__()