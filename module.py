
    


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities

def sinusoidal_time_embedding(t, dim):
    """
    t: (B,) or (B,1) float tensor of timesteps (already scaled to [0, 1] or raw indices you later scale)
    returns: (B, dim)
    """
    if t.dim() == 1:
        t = t[:, None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )  # (half,)
    args = t * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=t.device, dtype=t.dtype)], dim=-1)
    return emb  # (B, dim)

# ---------- Blocks

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False, groups=32):
        super().__init__()
        self.residual = residual
        mid_ch = out_ch if mid_ch is None else mid_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, mid_ch), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = None
        if residual and in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            s = x if self.skip is None else self.skip(x)
            y = F.silu(y + s)
        return y

class AddTime(nn.Module):
    """Inject a time (and optional class) embedding as a channel-wise bias."""
    def __init__(self, time_dim, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x, temb):
        # x: (B, C, H, W), temb: (B, time_dim)
        b, c, h, w = x.shape
        e = self.proj(temb).view(b, c, 1, 1)
        return x + e

class SelfAttention2D(nn.Module):
    """MHSA over (H*W) tokens, returns (B, C, H, W)."""
    def __init__(self, channels, num_heads=4, residual=False):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)  # (B, N, C)
        res = seq
        y = self.norm(seq)
        y, _ = self.attn(y, y, y)      # (B, N, C)
        y = self.mlp(y)                 # (B, N, C)
        if self.residual:
            y = F.gelu(y + res)
        y = y.permute(0, 2, 1).view(b, c, h, w)
        return y

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, residual=True)
        self.block2 = ConvBlock(out_ch, out_ch, residual=False)
        self.add_time = AddTime(time_dim, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, temb):
        x = self.block1(x)
        x = self.block2(x)
        x = self.add_time(x, temb)
        x_down = self.pool(x)
        return x, x_down  # return skip and downsampled

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block1 = ConvBlock(in_ch + skip_ch, out_ch, residual=True)
        self.block2 = ConvBlock(out_ch, out_ch, residual=False)
        self.add_time = AddTime(time_dim, out_ch)

    def forward(self, x, skip, temb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.add_time(x, temb)
        return x

# ---------- UNet

class UNetDDPM(nn.Module):
    def __init__(self, in_channels=1, base=32, time_dim=256, num_classes=None, attn_levels=(1,2)):
        """
        attn_levels: which down/up index levels include attention (0=highest resolution)
        """
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, time_dim) if num_classes is not None else None

        # Stem
        self.stem = ConvBlock(in_channels, base, residual=False)

        # Down path
        self.down1 = Down(base,   base*2, time_dim)  # level 0
        self.att1d = SelfAttention2D(base*2, residual=True) if 0 in attn_levels else nn.Identity()
        self.down2 = Down(base*2, base*4, time_dim)  # level 1
        self.att2d = SelfAttention2D(base*4, residual=True) if 1 in attn_levels else nn.Identity()
        self.down3 = Down(base*4, base*4, time_dim)  # level 2
        self.att3d = SelfAttention2D(base*4, residual=True) if 2 in attn_levels else nn.Identity()

        # Bottleneck
        self.bot1 = ConvBlock(base*4, base*8, residual=True)
        self.bot2 = ConvBlock(base*8, base*8, residual=True)

        # Up path (mirror; pay attention to skip channels)
        self.up1  = Up(in_ch=base*8, skip_ch=base*4, out_ch=base*4, time_dim=time_dim)  # with skip from down3
        self.att1u = SelfAttention2D(base*4, residual=True) if 2 in attn_levels else nn.Identity()
        self.up2  = Up(in_ch=base*4, skip_ch=base*4, out_ch=base*2, time_dim=time_dim)  # with skip from down2, CORRECTED skip_ch
        self.att2u = SelfAttention2D(base*2, residual=True) if 1 in attn_levels else nn.Identity()
        self.up3  = Up(in_ch=base*2, skip_ch=base*2, out_ch=base,   time_dim=time_dim)  # with skip from down1, CORRECTED skip_ch
        self.att3u = SelfAttention2D(base, residual=True) if 0 in attn_levels else nn.Identity()

        # Head
        self.head = nn.Conv2d(base, in_channels, kernel_size=1)

    def time_label_embed(self, t, label=None):
        temb = sinusoidal_time_embedding(t, self.time_dim)
        if self.label_emb is not None and label is not None:
            temb = temb + self.label_emb(label)
        return temb

    def forward(self, x, t, label=None):
        """
        x: (B, 3, H, W)
        t: (B,) float or long timesteps
        label: (B,) optional class labels for conditional DDPM
        """
        if t.dtype in (torch.long, torch.int32, torch.int64):
            t = t.float()
        temb = self.time_label_embed(t, label)  # (B, time_dim)

        x0 = self.stem(x)

        s1, x1 = self.down1(x0, temb) ; x1 = self.att1d(x1)
        s2, x2 = self.down2(x1, temb) ; x2 = self.att2d(x2)
        s3, x3 = self.down3(x2, temb) ; x3 = self.att3d(x3)

        b  = self.bot1(x3)
        b  = self.bot2(b)

        u1 = self.up1(b,  s3, temb) ; u1 = self.att1u(u1)
        u2 = self.up2(u1, s2, temb) ; u2 = self.att2u(u2)
        u3 = self.up3(u2, s1, temb) ; u3 = self.att3u(u3)

        out = self.head(u3)
        return out
        
