# Attention-Based Dual-Domain Sparse-View CT Reconstruction

This paper introduces a new sparse-view CT reconstruction method that combines a lightweight convolutional
interpolation network with a multi-head attention mechanism. We integrate this network into the dual-domain parallel
network architecture, incorporating a dual-domain fusion module based on attention.

The method has been re-implemented in PyTorch (`torch/`). The original TensorFlow code is kept in `tf/`.

## Repository layout

| directory | content |
|---|---|
| [`torch/`](torch/README.md) | PyTorch re-implementation; step-by-step reproduction instructions in [`torch/README.md`](torch/README.md) |
| [`tf/`](tf/) | original TensorFlow code |

## Quick start (PyTorch)

```bash
pip install torch            # CUDA build for your system
pip install -r torch/requirements.txt
# put the data under Data/ (format: torch/README.md, section 2)
python torch/train.py --c 4  # S = 4: 90 of 360 views
python torch/test.py  --c 4
```

## TensorFlow code

```
conda create -n tf25_py38 python=3.8
pip install -r tf/requirements.txt
cd tf          # the scripts read ./Data/..., so place (or link) Data/ inside tf/
```
