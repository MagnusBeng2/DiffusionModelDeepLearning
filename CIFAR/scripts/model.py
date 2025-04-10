import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Sinusoidal Time Embedding ===
def sinusoidal_embedding(timesteps, dim):
    assert dim % 2 == 0
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

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

# === Residual Block ===
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

    def forward(self, x, t_emb):
        h = self.block1(x)
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.block2(h)
        return h + self.residual(x)

# === Full UNet ===
class UNet(nn.Module):
    def __init__(self, time_emb_dim=256, num_classes=10):
        super().__init__()
        self.time_embedding_dim = time_emb_dim

        # === Embeddings ===
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # === Channel configuration (as in paper) ===
        self.channels = [128, 128, 256, 256]

        # === Downsampling ===
        self.input_conv = nn.Conv2d(3, self.channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        chs = self.channels
        for i in range(len(chs)):
            blocks = nn.ModuleList([
                ResBlock(chs[i-1] if i > 0 else chs[0], chs[i], time_emb_dim),
                ResBlock(chs[i], chs[i], time_emb_dim)
            ])
            attn = SelfAttention(chs[i]) if i in [1, 2] else nn.Identity()
            downsample = nn.Conv2d(chs[i], chs[i], 4, 2, 1) if i < len(chs) - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([blocks, attn, downsample]))

        # === Bottleneck ===
        self.mid_block1 = ResBlock(chs[-1], chs[-1], time_emb_dim)
        self.mid_attn = SelfAttention(chs[-1])
        self.mid_block2 = ResBlock(chs[-1], chs[-1], time_emb_dim)

        # === Upsampling ===
        self.ups = nn.ModuleList()
        reversed_chs = list(reversed(chs))
        for i in range(len(reversed_chs)):
            in_ch = reversed_chs[i] * 2
            out_ch = reversed_chs[i]
            next_ch = reversed_chs[i + 1] if i + 1 < len(reversed_chs) else out_ch
            blocks = nn.ModuleList([
                ResBlock(in_ch, out_ch, time_emb_dim),
                ResBlock(out_ch, out_ch, time_emb_dim)
            ])
            attn = SelfAttention(out_ch) if i in [1, 2] else nn.Identity()
            upsample = (
                nn.ConvTranspose2d(out_ch, next_ch, 4, 2, 1) if i + 1 < len(reversed_chs) else nn.Identity()
            )
            self.ups.append(nn.ModuleList([blocks, attn, upsample]))

        # === Final output ===
        self.out = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], 3, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y=None):
        # === Time and Label Embedding ===
        t_emb = sinusoidal_embedding(t, self.time_embedding_dim)
        if y is not None:
            t_emb = t_emb + self.label_emb(y.to(x.device))
        t_emb = self.time_mlp(t_emb)

        # === Downsample ===
        x = self.input_conv(x)
        residuals = []
        for blocks, attn, down in self.downs:
            for block in blocks:
                x = block(x, t_emb)
            x = attn(x)
            residuals.append(x)
            x = down(x)

        # === Bottleneck ===
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # === Upsample ===
        for blocks, attn, up in self.ups:
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)
            for block in blocks:
                x = block(x, t_emb)
            x = attn(x)
            x = up(x)

        return self.out(x)
