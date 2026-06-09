# Training Scripts

This directory contains reference training scripts used during the project.

The main public-facing pipeline in this repository is inference and visualization. These training scripts are included for reproducibility/reference, but some scripts may still require dataset-specific paths, checkpoint paths, or experiment settings to be adjusted before running on a new machine.

Common scripts:

```text
train_pretrain.py              # pretraining script
train_finetune.py              # finetuning script
train_finetune_4k.py           # 4K finetuning script
train_isp.py                   # ISP training script
train_predenoising.py          # pre-denoising training script
```

Before training, check and update:

```text
1. dataset paths
2. checkpoint save/load paths
3. GPU settings
4. batch size and patch size
5. black/white level assumptions
```

For environment setup, see `../SETUP.md` and `../environment.yaml`.
