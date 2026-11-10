"""Evaluate a trained network on the test patient.

    python torch/test.py --c 4                       # <run dir>/parallel_p2/best.pt
    python torch/test.py --c 4 --ckpt path/to/ckpt.pt

Writes next to the checkpoint: metrics<tag>.csv (one row per batch), summary<tag>.json (means, parameters, timing)
and recon<tag>.npy (float32 (N, 256, 256)).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FBP, batch_psnr, batch_ssim, count_params, interp_views, load_split  # noqa: E402
from model import ParallelNet  # noqa: E402
from train import ROOT, run_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.join(ROOT, 'Data'))
    p.add_argument('--dataset', default='public')
    p.add_argument('--test_patient', type=int, default=10)
    p.add_argument('--c', type=int, default=4)
    p.add_argument('--p', type=int, default=2)
    p.add_argument('--out', default=os.path.join(ROOT, 'runs'))
    p.add_argument('--ckpt', default=None, help='default: <out>/<dataset>_patient<k>/S<c>/parallel_p<p>/best.pt')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    dev = torch.device('cuda')
    ckpt = a.ckpt or os.path.join(run_dir(a), 'parallel_p%d' % a.p, 'best.pt')
    out = os.path.dirname(os.path.abspath(ckpt))

    fbp = FBP(os.path.join(a.data_dir, 'My_AT_256.npz'), dev)
    model = ParallelNet(a.c, fbp, p=a.p).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    sino, ct = load_split(a.data_dir, a.dataset, 'test', a.test_patient)
    rows, recon = [], []
    with torch.no_grad():
        for b in range(0, len(sino), 2):
            s = torch.from_numpy(sino[b:b + 2]).to(dev)
            o = model(s[:, ::a.c])
            img = o['img'][:, 0].cpu().numpy()
            fbp_lin = fbp(interp_views(s[:, ::a.c], a.c))[:, 0].cpu().numpy()
            ref = ct[b:b + 2]
            miss = torch.ones(s.shape[1], dtype=torch.bool, device=dev); miss[::a.c] = False
            s_out = o['sino']
            rows.append([b // 2, batch_psnr(ref, fbp_lin), batch_psnr(ref, img), batch_ssim(ref, fbp_lin), batch_ssim(ref, img),
                         batch_psnr(s.cpu().numpy(), s_out.cpu().numpy()),
                         batch_psnr(s[:, miss].cpu().numpy(), s_out[:, miss].cpu().numpy())])
            recon.append(img)
        s1 = torch.from_numpy(sino[:1]).to(dev)[:, ::a.c]       # inference time: batch 1
        for _ in range(20):
            model(s1)
        torch.cuda.synchronize()
        ts = []
        for _ in range(100):
            t = time.perf_counter(); model(s1); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)

    with open(os.path.join(out, 'metrics%s.csv' % a.tag), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch', 'psnr_fbp_interp', 'psnr', 'ssim_fbp_interp', 'ssim', 'psnr_sino_full', 'psnr_sino_missing'])
        w.writerows(rows)
    np.save(os.path.join(out, 'recon%s.npy' % a.tag), np.concatenate(recon).astype(np.float32))
    r = np.array(rows)
    summ = {'dataset': a.dataset, 'test_patient': a.test_patient, 'S': a.c, 'p': a.p, 'ckpt': ckpt, 'n_test': len(sino),
            'psnr': r[:, 2].mean(), 'ssim_x100': 100 * r[:, 4].mean(),
            'psnr_fbp_interp': r[:, 1].mean(), 'ssim_fbp_interp_x100': 100 * r[:, 3].mean(),
            'psnr_sino_full': r[:, 5].mean(), 'psnr_sino_missing': r[:, 6].mean(),
            'params_M_excl_fbp': count_params(model) / 1e6, 'time_ms_per_slice_b1_median': 1e3 * float(np.median(ts))}
    json.dump(summ, open(os.path.join(out, 'summary%s.json' % a.tag), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
