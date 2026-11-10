# PyTorch implementation

PyTorch re-implementation of the parallel dual-domain network for sparse-view CT.

| file | content |
|---|---|
| `common.py` | data split, linear view interpolation, FBP, PSNR / SSIM |
| `model.py` | sinogram branch (FCN + Transformer encoders), image branch (U-Net), attention-based fusion module |
| `train.py` | two-stage training |
| `test.py` | evaluation on the test patient |
| `make_fbp_matrix.py` | generates the fan-beam FBP operator `Data/My_AT_256.npz` |

## 1. Environment

```bash
conda create -n svct_torch python=3.10 -y && conda activate svct_torch
pip install torch --index-url https://download.pytorch.org/whl/cu121   # choose the build for your CUDA
pip install -r torch/requirements.txt
```
A CUDA GPU with at least 8 GB of memory is required.

## 2. Data

Place the files under `Data/` in the repository root (or pass `--data_dir`):

```
Data/
  My_AT_256.npz                          fan-beam FBP operator (generate it, see below)
  mymodel/My_data_public_256_2000.npz    2000 slices, ordered by patient
```

Generate the FBP operator (a few seconds, about 750 MB):

```bash
python torch/make_fbp_matrix.py          # -> Data/My_AT_256.npz
```

* Source: chest scans of the 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge (TCIA), full-dose images,
  10 patients x 200 slices, 256 x 256, each slice normalised to max = 1.
* Sinograms: simulated with MATLAB `fanbeam` (source-to-isocentre distance 600 pixels, equiangular, 360 views over
  360 degrees, 357 detectors).
* `My_data_<dataset>_256_2000.npz`: `f_img` float32 (2000, 360, 357) sinograms; `u_img` float32 (2000, 256, 256) images.
  Patient k occupies slices `[200(k-1), 200k)`.
* `My_AT_256.npz`: pixel-driven fan-beam back-projection matrix with 1/L^2 weighting and linear interpolation between
  detector channels (`arr_0` (nnz, 2) [sinogram index, pixel index], `arr_1` values, `arr_2` = (128520, 65536)) and
  the 713-tap equiangular ramp filter (`arr_3`); geometry and formulas are documented in `make_fbp_matrix.py`.

## 3. Data split

Patients 1-9 (1800 slices) are used for training, with 2 random slices held out for validation; patient 10
(200 slices) is used for testing. Use `--test_patient k` to hold out another patient.

## 4. Train and test

Run from the repository root. `--c` is the sparsity factor S (360/S views are kept): 2, 4, 8 or 15.

```bash
python torch/train.py --c 4
python torch/test.py  --c 4

# all sparsity factors
for C in 2 4 8 15; do python torch/train.py --c $C && python torch/test.py --c $C; done
```

Training: Adam, learning rate 1e-3, batch 2, MSE losses, seed 42. The validation PSNR is checked every
100 iterations and the best checkpoint is kept as `best.pt` (every epoch is also saved as `ep<k>.pt`, plus `last.pt`).

| stage | `--stages` | epochs | content |
|---|---|---|---|
| 1 | `sino` | 15 | pre-train the sinogram-domain branch |
| 1 | `unet` | 8 | pre-train the image-domain branch |
| 2 | `joint` | 2 | fine-tune the parallel network with the fusion module |

Outputs go to `runs/<dataset>_patient<k>/S<c>/`:

```
sino_p2/        best.pt ep*.pt last.pt train_log.txt args.json
unet/           ...
parallel_p2/    ... + metrics.csv summary.json recon.npy   (written by test.py)
```

`summary.json` contains the PSNR / SSIM of the reconstruction and of FBP(linear interpolation), the sinogram PSNR,
the number of parameters and the inference time per slice.

Pretrained weights: `python torch/test.py --c 4 --ckpt torch/weights/S4.pt`

## 5. Metrics

PSNR and SSIM are computed over the full 256 x 256 images and averaged over the test slices. SSIM uses
`skimage.metrics.structural_similarity` (scikit-image 0.19.3) with default settings.
