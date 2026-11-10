"""Generate the fan-beam FBP operator Data/My_AT_256.npz used by common.FBP.

Geometry: 256 x 256 image, rotation centre at pixel (127, 127) (0-based), source-to-centre distance 600.16 pixels,
equiangular detector with 357 channels spaced 0.1 degrees (central channel 178), 360 views spaced 1 degree.

* Back-projection matrix (128520 x 65536, row = view * 357 + detector, column = row-major pixel index):
  pixel-driven; for each view and pixel, the fan angle of the ray through the pixel is converted to a detector
  position, the pixel is distributed to the two neighbouring channels by linear interpolation, and weighted by 1/L^2
  (L = source-to-pixel distance).
* Filter (713 taps): equiangular ramp filter, h(n) = (D/2) (n a / sin(n a))^2 (sinc(n)/2 - sinc(n/2)^2/4),
  a = 0.1 degrees.

Output format (as read by common.FBP): arr_0 (nnz, 2) [row, column], arr_1 (nnz,) values,
arr_2 = (128520, 65536), arr_3 (713, 1) filter.

    python torch/make_fbp_matrix.py            # -> Data/My_AT_256.npz
"""
import argparse
import os

import numpy as np

N, N_VIEWS, N_DET = 256, 360, 357
CENTRE = 127.0
SOURCE_DIST = 600.16
DET_CENTRE = 178
DET_STEP_DEG = 0.1
VIEW_STEP_DEG = 1.0


def backprojection_matrix():
    r, c = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64), indexing='ij')
    r, c = r.ravel(), c.ravel()
    pix = np.arange(N * N)
    rows, cols, vals = [], [], []
    for v in range(N_VIEWS):
        b = np.deg2rad(v * VIEW_STEP_DEG)
        sx, sy = CENTRE + SOURCE_DIST * np.cos(b), CENTRE + SOURCE_DIST * np.sin(b)   # source (column, row)
        ux, uy = CENTRE - sx, CENTRE - sy                                             # source -> centre
        px, py = c - sx, r - sy                                                       # source -> pixel
        gamma = -np.degrees(np.arctan2(ux * py - uy * px, ux * px + uy * py))         # fan angle
        t = DET_CENTRE + gamma / DET_STEP_DEG
        w = 1.0 / (px * px + py * py)
        k = np.floor(t)
        f = t - k
        k = k.astype(np.int64)
        for kk, ww in ((k, (1 - f) * w), (k + 1, f * w)):
            m = (kk >= 0) & (kk < N_DET) & (ww != 0)
            rows.append(v * N_DET + kk[m]); cols.append(pix[m]); vals.append(ww[m])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def ramp_filter():
    n = np.arange(-(N_DET - 1), N_DET).astype(np.float64)
    a = np.deg2rad(DET_STEP_DEG)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(n == 0, 1.0, (n * a / np.sin(n * a)) ** 2)
    return (SOURCE_DIST / 2 * ratio * (0.5 * np.sinc(n) - 0.25 * np.sinc(n / 2) ** 2)).reshape(-1, 1)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(root, 'Data', 'My_AT_256.npz'))
    a = ap.parse_args()
    rows, cols, vals = backprojection_matrix()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, np.stack([rows, cols], 1).astype(np.int32), vals,
             np.array([N_VIEWS * N_DET, N * N], dtype=np.int64), ramp_filter())
    print('wrote %s  (nnz %d)' % (a.out, len(vals)))


if __name__ == '__main__':
    main()
