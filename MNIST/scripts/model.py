import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Sinusoidal Time Embedding ===
def sinusoidal_embedding(timesteps, dim):
    assert dim % 2 == 0
    half_dim = dim // 2
    emb_factor = math.log(10000) / (half_dim - 1)
    exponents = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponents = torch.exp(-emb_factor * exponents)
    sinusoid = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
    emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
    return emb
# === Self-Attention Block ===
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)

# === Residual Block with Time Embedding ===
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = self.block2(h)
        return h + self.residual(x)

# === Full UNet ===
class UNet(nn.Module):
    def __init__(self, time_embedding_dim=128):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.label_emb = nn.Embedding(10, time_embedding_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )

        # Downsampling
        self.down1 = ResBlock(1, 64, 512)
        self.downsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 28 → 14
        self.down2 = ResBlock(64, 128, 512)
        self.downsample2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # 14 → 7

        # Bottleneck
        self.middle = ResBlock(128, 128, 512)
        self.attn_mid = SelfAttention(128)

        # Upsampling
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # 7 → 14
        self.up2 = ResBlock(256, 64, 512)  # 128 + 128 = 256
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 14 → 28
        self.up1 = ResBlock(128, 64, 512)  # 64 + 64 = 128

        self.out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, t, y=None):
        # Time embedding
        t_embed = sinusoidal_embedding(t, self.time_embedding_dim).to(x.device)
        if y is not None:
            t_embed = t_embed + self.label_emb(y.to(x.device))
        t_embed = self.time_mlp(t_embed)

        # Down
        x1 = self.down1(x, t_embed)          # [B, 64, 28, 28]
        x1_ds = self.downsample1(x1)         # [B, 64, 14, 14]
        x2 = self.down2(x1_ds, t_embed)      # [B, 128, 14, 14]
        x2_ds = self.downsample2(x2)         # [B, 128, 7, 7]

        # Middle
        m = self.middle(x2_ds, t_embed)      # [B, 128, 7, 7]
        m = self.attn_mid(m)

        # Up
        u2 = self.upsample1(m)               # [B, 128, 14, 14]
        if u2.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(x2, size=u2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1), t_embed)

        u1 = self.upsample2(u2)              # [B, 64, 28, 28]
        if u1.shape[-2:] != x1.shape[-2:]:
            x1 = F.interpolate(x1, size=u1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1), t_embed)

        return self.out(u1)
