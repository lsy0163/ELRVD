# Training Scripts

This directory contains reference training scripts used during the project. The main public-facing pipeline in this repository is inference and visualization; these scripts are included for reproducibility and reference.

## Script-to-stage mapping

The final model (RViDeNet-ECBAM, `inference/models/denoiser/model_epoch500.pth`) was trained with the following 3-stage strategy:

| Stage | Script | Description |
|---|---|---|
| 1. Pre-denoising pretraining | `train_predenoising.py` | Trains the pre-denoising module on synthetic noisy-clean pairs (SID clean RAW + Poisson-Gaussian noise). Frozen afterwards; used only to guide deformable alignment offsets. |
| 2. RViDeNet pretraining | `train_pretrain.py` | Pretrains the full network on synthetic RAW video (MOTChallenge sRGB → unprocessed RAW + synthetic noise). RAW reconstruction loss only. lr 1e-4 → 1e-5 after 20 epochs. |
| 3-1. CRVD fine-tuning | `train_finetune.py` | Fine-tunes on CRVD (real RAW, GBRG packing) to adapt from synthetic to real RAW noise. Layer-wise lr: backbone 1e-6, recon trunk / attention / output conv 1e-5. RAW reconstruction + temporal consistency loss. |
| 3-2. Self-recorded fine-tuning | `train_finetune_self_record.py` | Final fine-tuning on the self-captured 0.1 lux IMX327 dataset (RGGB packing) with `RViDeNet_ECBAM`. |
| (Optional) ISP module | `train_isp.py` | Trains the ISP module (`inference/models/isp/model_epoch770.pth`). It is loaded frozen by the fine-tuning scripts to compute an sRGB-domain loss; it is not used at inference time (visualization uses debayering instead). |

## Experimental variants

The remaining scripts are experimental variants explored during the project and are **not** part of the final pipeline (different frame counts, noise models, or architectures):

```text
train_1frame.py / train_pretrain_1frame.py        # single-frame input ablation
train_finetune_5frame.py / train_pretrain_5frame.py  # 5-frame input ablation
train_pretrain_gan_noise.py / train_predenoising_gan_noise.py  # GAN-based noise model experiments
train_finetune_VNT.py                             # VNT architecture experiment
train_finetune_4k.py                              # 4K-resolution fine-tuning experiment
train_pretrain_imx327.py                          # IMX327 noise-model pretraining experiment
train_pretrain_orginal.py                         # original RViDeNet pretraining (baseline)
```

## Before running

Check and update:

```text
1. dataset paths (scripts contain machine-specific absolute paths)
2. checkpoint save/load paths
3. GPU settings
4. batch size and patch size
5. black/white level and Bayer layout assumptions
```

For environment setup, see `../SETUP.md` and `../environment.yaml`.
