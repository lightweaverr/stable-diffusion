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
    def __init__(self, n_heads, n_embed, d_context=768) -> None:
        super().__init__()
        channels = n_heads * n_embed
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (Batch, channels, height, width)
        # context: (B, seqlen, dim)
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        b, c, h, w = x.shape
        x = x.view((b, c, h*w))
        x = x.transpose(-1,-2)
        # x: (b, h * w, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1,-2)
        x.view((b, c, h, w))
        x = self.conv_output(x) + residue_long
        return x
    

class Upsample(nn.Module):
    def __init(self, channels):
        super().__init__()
        self.conv = nn.Conv2(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = F.interpolate(x,scale_factor=2, mode='nearest')
        # x: (B, C, 2H, 2W)
        return self.conv(x)