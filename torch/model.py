"""Parallel dual-domain network for sparse-view CT.

Sinogram branch: a lightweight FCN (three 5x5 convolutions with tanh, 32 feature maps, and a 1x1 output layer) predicts
the missing views on top of the linear interpolation; the views are then used as tokens (dim 360) by p Transformer
encoders (8 heads, feed-forward 1440, GELU, dropout 0.1, no positional encoding). The completed sinogram is
reconstructed by FBP.
Image branch: FBP of the linearly interpolated sinogram, enhanced by a U-Net (32/64/128/256 feature maps).
Fusion: the two images are concatenated and fused by a convolution, channel attention (CA) and spatial attention (SA)
with reduction ratio 4, and a convolution; the result is added to the image-branch output.
Glorot-uniform initialisation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FBP, interp_views

PAD_DET = 3     # 357 -> 360
CUT = 6         # wrap/zero padding consumed by the three 'valid' 5x5 convs


def glorot_(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


# ----------------------------------------------------------------------------- sinogram branch
class TBlock(nn.Module):
    def __init__(self, d=360, heads=8, mlp=1440, p=0.1):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.drop(self.proj(a)))        
        y = self.drop(gelu_tanh(self.fc1(self.n2(x))))
        return x + self.drop(self.fc2(y))


class SinoTransformer(nn.Module):
    def __init__(self, c, layers=2):
        super().__init__()
        self.c = c
        self.convs = nn.ModuleList([nn.Conv2d(1, 32, 5), nn.Conv2d(32, 32, 5), nn.Conv2d(32, 32, 5)])
        self.head = nn.Conv2d(32, c - 1, 1)
        self.blocks = nn.ModuleList([TBlock() for _ in range(layers)])
        glorot_(self)

    def forward(self, s_sparse):
        """s_sparse: (B, n, 357) raw -> missing views (B, n*(c-1), 360) and completed sinogram (B, 360, 357)."""
        c = self.c
        s = F.pad(s_sparse, (0, PAD_DET))                                     # (B, n, 360)
        B, n, D = s.shape
        full_lin = interp_views(s, c).reshape(B, n, c, D)                     # view k*c+j
        interp = full_lin[:, :, 1:]                                           # (B, n, c-1, D)
        ex = torch.cat([s[:, -CUT:], s, s[:, :CUT]], 1)                       # wrap views
        ex = F.pad(ex, (CUT, CUT))[:, None]                                   # (B, 1, n+12, 372)
        h = ex
        for conv in self.convs:
            h = torch.tanh(conv(h))
        h = self.head(h).permute(0, 2, 1, 3)                                  # (B, n, c-1, 360)
        x = (h + interp).reshape(B, n * (c - 1), D)                           # tokens: view k*c+j
        for blk in self.blocks:
            x = blk(x)
        full = torch.cat([s[:, :, None], x.reshape(B, n, c - 1, D)], 2).reshape(B, n * c, D)
        return x, full[..., :D - PAD_DET]


def missing_views_label(s_full, c):
    """Label of the sinogram branch: all non-sampled views, order k*c+j (j>=1), detector padded to 360."""
    s = F.pad(s_full, (0, PAD_DET))
    B, V, D = s.shape
    return s.reshape(B, V // c, c, D)[:, :, 1:].reshape(B, -1, D)


# ----------------------------------------------------------------------------- image branch
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        a = lambda t: F.leaky_relu(t, 0.2)
        self.a = a
        C = lambda i, o, s=1: nn.Conv2d(i, o, 3, stride=s, padding=1)
        T = lambda i, o: nn.ConvTranspose2d(i, o, 3, stride=2, padding=1, output_padding=1)
        self.inp = nn.Conv2d(1, 32, 1)
        self.c1a, self.c1b = C(32, 64, 2), C(64, 64)
        self.c2a, self.c2b = C(64, 128, 2), C(128, 128)
        self.c3a, self.c3b = C(128, 256, 2), C(256, 256)
        self.t2, self.e2 = T(256, 128), C(256, 128)
        self.t3, self.e3 = T(128, 64), C(128, 64)
        self.t4, self.e4 = T(64, 32), C(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
        glorot_(self)

    def forward(self, x):
        a = self.a
        i0 = a(self.inp(x))
        c1 = a(self.c1b(a(self.c1a(i0))))
        c2 = a(self.c2b(a(self.c2a(c1))))
        c3 = a(self.c3b(a(self.c3a(c2))))
        e2 = a(self.e2(torch.cat([self.t2(c3), c2], 1)))
        e3 = a(self.e3(torch.cat([self.t3(e2), c1], 1)))
        e4 = a(self.e4(torch.cat([self.t4(e3), i0], 1)))
        return x - a(self.out(e4))


# ----------------------------------------------------------------------------- fusion
class FrozenBN(nn.Module):
    """Batch normalisation with fixed statistics (learnable affine)."""

    def __init__(self, ch, eps=1e-3):
        super().__init__()
        self.g, self.b, self.k = nn.Parameter(torch.ones(ch)), nn.Parameter(torch.zeros(ch)), 1 / math.sqrt(1 + eps)

    def forward(self, x):
        return x * (self.g * self.k)[None, :, None, None] + self.b[None, :, None, None]


class GAM(nn.Module):
    """Channel attention followed by spatial attention."""

    def __init__(self, c=32, r=4):
        super().__init__()
        self.c = c
        self.ch = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c))
        self.sp = nn.Sequential(nn.Conv2d(c, c // r, 7, padding=3), FrozenBN(c // r), nn.ReLU(),
                                nn.Conv2d(c // r, c, 7, padding=3), FrozenBN(c))

    def forward(self, x):                   # (B, C, H, W)
        B, C, H, W = x.shape
        x = x * self.ch(x.reshape(B, -1, C)).reshape(B, C, H, W)
        return x * torch.sigmoid(self.sp(x))


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cin = nn.Conv2d(2, 32, 5, padding=2)
        self.gam = GAM(32)
        self.c2 = nn.Conv2d(32, 32, 5, padding=2)
        self.cout = nn.Conv2d(32, 1, 3, padding=1)
        glorot_(self)

    def forward(self, fbp_sino, ct_out):
        f = self.gam(self.cin(torch.cat([fbp_sino, ct_out], 1)))
        return self.cout(self.c2(f)) + ct_out


# ----------------------------------------------------------------------------- full network
class ParallelNet(nn.Module):
    def __init__(self, c, fbp, p=2):
        super().__init__()
        self.c = c
        self.sino = SinoTransformer(c, layers=p)
        self.fbp_sino = fbp                       # FBP of the sinogram branch
        self.fbp_ct = FBP(None, None, share=fbp)  # FBP of the image branch
        self.unet = UNet()
        self.fusion = Fusion()

    def forward(self, s_sparse):
        _, full = self.sino(s_sparse)
        fbp_s = self.fbp_sino(full)
        ct_out = self.unet(self.fbp_ct(interp_views(s_sparse, self.c)))
        return {'img': self.fusion(fbp_s, ct_out), 'fbp': fbp_s, 'ct': ct_out, 'sino': full}
