"""Train the parallel dual-domain network.

Stage 1 pre-trains the two branches separately (sino: sinogram branch, 15 epochs; unet: image branch, 8 epochs);
stage 2 (joint, 2 epochs) fine-tunes the network with the fusion module. MSE losses, Adam lr 1e-3, batch 2, seed 42.
Two training slices are held out for validation; the checkpoint with the best validation PSNR is kept as best.pt.

    python torch/train.py --c 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FBP, batch_psnr, count_params, interp_views, load_split, seed_all  # noqa: E402
from model import ParallelNet, SinoTransformer, UNet, missing_views_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.join(ROOT, 'Data'))
    p.add_argument('--dataset', default='public')
    p.add_argument('--test_patient', type=int, default=10)
    p.add_argument('--c', type=int, default=4, help='sparsity factor S (360/S views)')
    p.add_argument('--stages', default='sino,unet,joint')
    p.add_argument('--p', type=int, default=2, help='number of Transformer encoders')
    p.add_argument('--ep_sino', type=int, default=15)
    p.add_argument('--ep_unet', type=int, default=8)
    p.add_argument('--ep_joint', type=int, default=2)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val_sz', type=int, default=2)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_iters', type=int, default=0, help='stop each stage after this many iterations (debug)')
    p.add_argument('--out', default=os.path.join(ROOT, 'runs'))
    return p.parse_args()


def run_dir(a):
    return os.path.join(a.out, '%s_patient%d' % (a.dataset, a.test_patient), 'S%d' % a.c)


class Run:
    def __init__(self, a):
        self.a = a
        seed_all(a.seed)
        self.dev = torch.device('cuda')
        sino, ct = load_split(a.data_dir, a.dataset, 'train', a.test_patient)
        ids = np.random.permutation(len(sino))
        self.tr, self.va = ids[a.val_sz:], ids[:a.val_sz]
        self.sino = torch.from_numpy(sino).to(self.dev)
        self.ct = torch.from_numpy(ct).to(self.dev)
        self.fbp = FBP(os.path.join(a.data_dir, 'My_AT_256.npz'), self.dev)
        for q in self.fbp.parameters():
            q.requires_grad_(False)
        self.base = run_dir(a)
        print('train slices %d (val %d), S=%d -> %s' % (len(self.tr), len(self.va), a.c, self.base))

    def dir(self, name):
        d = os.path.join(self.base, name)
        os.makedirs(d, exist_ok=True)
        return d

    def log(self, d, msg):
        print(msg, flush=True)
        with open(os.path.join(d, 'train_log.txt'), 'a') as f:
            f.write(msg + '\n')

    def loop(self, d, name, epochs, params, step_fn, val_fn, state_fn, epoch_hook=None):
        a = self.a
        json.dump(vars(a) | {'stage': name, 'epochs': epochs}, open(os.path.join(d, 'args.json'), 'w'), indent=1)
        opt = torch.optim.Adam(params, lr=a.lr)
        spe = len(self.tr) // a.batch

        def val():
            with torch.no_grad():
                return float(np.mean([batch_psnr(self.ct[self.va[b:b + 2]].cpu().numpy(),
                                                 val_fn(torch.from_numpy(self.va[b:b + 2]).to(self.dev)))
                                      for b in range(0, len(self.va), 2)]))

        best, it, t0 = -1e9, 0, time.time()
        for ep in range(epochs):
            if epoch_hook:
                epoch_hook(ep)
            perm = np.random.permutation(self.tr)
            run = 0.0
            for b in range(spe):
                idx = torch.from_numpy(perm[b * a.batch:(b + 1) * a.batch]).to(self.dev)
                loss = step_fn(idx)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run += loss.item(); it += 1
                if it % 200 == 0:
                    self.log(d, '[%s] ep %d it %d loss %.6g %.0fs' % (name, ep, it, run / 200, time.time() - t0))
                    run = 0.0
                if it % a.eval_every == 0:
                    v = val()
                    if v > best:
                        best = v
                        torch.save(state_fn(), os.path.join(d, 'best.pt'))
                        self.log(d, '[%s] it %d new best val PSNR %.3f' % (name, it, v))
                if a.max_iters and it >= a.max_iters:
                    break
            v = val()
            if v > best:
                best = v
                torch.save(state_fn(), os.path.join(d, 'best.pt'))
            torch.save(state_fn(), os.path.join(d, 'ep%d.pt' % ep))
            torch.save(state_fn(), os.path.join(d, 'last.pt'))
            self.log(d, '[%s] epoch %d done  val PSNR %.3f  best %.3f  %.1f min' % (name, ep, v, best, (time.time() - t0) / 60))
            if a.max_iters and it >= a.max_iters:
                break

    # ------------------------------------------------------------------ stage 1: sinogram branch
    def stage_sino(self):
        c, d = self.a.c, self.dir('sino_p%d' % self.a.p)
        m = SinoTransformer(c, layers=self.a.p).to(self.dev)
        self.log(d, 'sinogram branch p=%d params %.3fM' % (self.a.p, count_params(m) / 1e6))

        def step(idx):
            m.train()
            s = self.sino[idx]
            x, _ = m(s[:, ::c])
            return F.mse_loss(x, missing_views_label(s, c))

        def val(idx):
            m.eval()
            return self.fbp(m(self.sino[idx][:, ::c])[1])[:, 0].cpu().numpy()

        self.loop(d, 'sino', self.a.ep_sino, m.parameters(), step, val, m.state_dict)

    # ------------------------------------------------------------------ stage 1: image branch
    def stage_unet(self):
        c, d = self.a.c, self.dir('unet')
        u = UNet().to(self.dev)
        self.log(d, 'U-Net params %.3fM' % (count_params(u) / 1e6))
        with torch.no_grad():
            ct_in = torch.cat([self.fbp(interp_views(self.sino[b:b + 8][:, ::c], c)) for b in range(0, len(self.sino), 8)])

        def step(idx):
            return F.mse_loss(u(ct_in[idx]), self.ct[idx][:, None])

        self.loop(d, 'unet', self.a.ep_unet, u.parameters(), step, lambda idx: u(ct_in[idx])[:, 0].cpu().numpy(), u.state_dict)

    # ------------------------------------------------------------------ stage 2: joint fine-tuning
    def stage_joint(self):
        a, c = self.a, self.a.c
        d = self.dir('parallel_p%d' % a.p)
        net = ParallelNet(c, self.fbp, p=a.p).to(self.dev)
        net.sino.load_state_dict(torch.load(os.path.join(self.base, 'sino_p%d' % a.p, 'best.pt'), map_location=self.dev))
        net.unet.load_state_dict(torch.load(os.path.join(self.base, 'unet', 'best.pt'), map_location=self.dev))
        net.eval()
        for q in net.sino.parameters():
            q.requires_grad_(False)
        self.log(d, 'parallel net p=%d params (excl. FBP) %.3fM' % (a.p, count_params(net) / 1e6))
        with torch.no_grad():
            fbp_s = torch.cat([net.fbp_sino(net.sino(self.sino[b:b + 8][:, ::c])[1]) for b in range(0, len(self.sino), 8)])
        trainable = list(net.fusion.parameters()) + list(net.unet.parameters()) + list(net.fbp_ct.parameters())

        def hook(ep):
            on = ep >= 1
            for q in list(net.unet.parameters()) + list(net.fbp_ct.parameters()):
                q.requires_grad_(on)

        def fwd(idx):
            ct_out = net.unet(net.fbp_ct(interp_views(self.sino[idx][:, ::c], c)))
            return net.fusion(fbp_s[idx], ct_out)

        self.loop(d, 'joint', a.ep_joint, trainable, lambda idx: F.mse_loss(fwd(idx), self.ct[idx][:, None]),
                  lambda idx: fwd(idx)[:, 0].cpu().numpy(), net.state_dict, hook)


def main():
    a = get_args()
    r = Run(a)
    t0 = time.time()
    for s in a.stages.split(','):
        getattr(r, 'stage_' + s)()
    print('total training time %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
