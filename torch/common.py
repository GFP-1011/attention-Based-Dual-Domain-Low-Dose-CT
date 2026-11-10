"""Data, FBP operator and metrics shared by train.py and test.py.

* precision: fp32 (TF32 disabled).
* data:      <data_dir>/mymodel/My_data_<dataset>_256_2000.npz, 2000 slices ordered by patient (10 patients x 200):
             f_img (N, 360, 357) float32 fan-beam sinograms, u_img (N, 256, 256) float32 images in [0, 1].
* split:     patient level: 9 patients for training, 1 patient (default: patient 10) for testing.
* sparse:    sinogram[:, 0::S, :]; missing views filled by circular linear interpolation.
* FBP:       system matrix + ramp filter from <data_dir>/My_AT_256.npz.
* metrics:   PSNR and SSIM over the full 256 x 256 images, averaged over the test slices.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

N_VIEWS, N_DET, IMG = 360, 357, 256
SLICES_PER_PATIENT, N_PATIENTS = 200, 10


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def load_split(data_dir, dataset, part, test_patient=10):
    """Return (sino (N, 360, 357), ct (N, 256, 256)) float32 for part = 'train' | 'test'.
    Patient k occupies slices [200(k-1), 200k) of My_data_<dataset>_256_2000.npz."""
    assert 1 <= test_patient <= N_PATIENTS
    d = np.load(os.path.join(data_dir, 'mymodel/My_data_%s_256_2000.npz' % dataset))
    sino, ct = d['f_img'], d['u_img']
    assert len(sino) == SLICES_PER_PATIENT * N_PATIENTS, 'expected 2000 patient-ordered slices'
    pid = np.arange(len(sino)) // SLICES_PER_PATIENT + 1
    sel = (pid == test_patient) if part == 'test' else (pid != test_patient)
    return sino[sel].astype(np.float32), ct[sel].astype(np.float32).reshape(-1, IMG, IMG)


def interp_views(s_sparse, c):
    """Circular linear interpolation of a sparse sinogram (B, 360/c, D) to (B, 360, D).

    View k*c+j (0 <= j < c) = (c-j)/c * s[k] + j/c * s[(k+1) mod n].
    """
    if c == 1:
        return s_sparse
    nxt = torch.roll(s_sparse, shifts=-1, dims=1)
    w = torch.arange(c, device=s_sparse.device, dtype=s_sparse.dtype) / c
    full = (1 - w)[None, None, :, None] * s_sparse[:, :, None, :] + w[None, None, :, None] * nxt[:, :, None, :]
    return full.reshape(s_sparse.shape[0], -1, s_sparse.shape[2])


# ----------------------------------------------------------------------------- FBP
class _SparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, AT, A):            # x: (n_sino, B) -> (n_img, B)
        ctx.A = A
        return torch.sparse.mm(AT, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.A, g.contiguous()), None, None


class FBP(nn.Module):
    """Filtered back-projection with the fan-beam system matrix of My_AT_256.npz
    (arr_0: (nnz, 2) [sinogram index, pixel index], arr_1: values, arr_2: shape (128520, 65536), arr_3: 713-tap filter)."""

    def __init__(self, at_path, device, share=None):
        super().__init__()
        if share is not None:   # second FBP layer with the same matrix
            self.A, self.AT = share.A, share.AT
            self.filter = nn.Parameter(share.filter.detach().clone())
            self.scale = nn.Parameter(torch.tensor([10.0], device=share.filter.device))
            self.bias = nn.Parameter(torch.tensor([0.0], device=share.filter.device))
            return
        raw = np.load(at_path)
        idx = torch.from_numpy(raw['arr_0'].astype(np.int64).T)
        val = torch.from_numpy(raw['arr_1'].astype(np.float32))
        shape = tuple(int(s) for s in raw['arr_2'])
        A = torch.sparse_coo_tensor(idx, val, shape).coalesce()        # duplicate entries are summed
        self.A = A.to(device).to_sparse_csr()                          # (n_sino, n_img), used in backward
        self.AT = A.t().coalesce().to(device).to_sparse_csr()          # (n_img, n_sino)
        self.filter = nn.Parameter(torch.from_numpy(raw['arr_3'].astype(np.float32).reshape(1, 1, -1)))
        self.scale = nn.Parameter(torch.tensor([10.0]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.to(device)

    def forward(self, sino):               # (B, 360, 357) -> (B, 1, 256, 256)
        with torch.autocast('cuda', enabled=False):
            sino = sino.float()
            B, V, D = sino.shape
            k = self.filter.shape[-1]
            x = F.conv1d(sino.reshape(B * V, 1, D), self.filter, padding=(k - 1) // 2)
            x = x.reshape(B, V * D).t().contiguous()
            img = _SparseMM.apply(x, self.AT, self.A).t().reshape(B, 1, IMG, IMG)
            return img * self.scale + self.bias


# ----------------------------------------------------------------------------- metrics
def batch_psnr(ref, out):
    """Mean PSNR over a batch; ref/out: (B, H, W)."""
    ref, out = ref.astype(np.float32), out.astype(np.float32)
    max_val = ref.max() - ref.min()
    mse = ((ref - out) ** 2).reshape(ref.shape[0], -1).mean(1)
    return float(np.mean(20 * np.log10(max_val) - 10 * np.log10(mse)))


def batch_ssim(ref, out):
    """Mean SSIM over a batch (scikit-image, default settings)."""
    from skimage.metrics import structural_similarity as s_ssim
    return float(np.mean([s_ssim(ref[i].astype(np.float32), out[i].astype(np.float32)) for i in range(ref.shape[0])]))


def count_params(model):
    """Trainable parameters excluding the FBP layers."""
    fbp_ids = {id(p) for m in model.modules() if isinstance(m, FBP) for p in m.parameters()}
    return sum(p.numel() for p in model.parameters() if id(p) not in fbp_ids)
