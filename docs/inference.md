# Inference

Main entry point:

```bash
python inference.py --help
```

Important arguments:

```text
--input_dir       directory containing noisy RAW frames
--input_pattern   glob pattern for RAW input files
--output_dir      directory for denoised RAW output
--height          RAW height
--width           RAW width
--black_level     RAW black level for normalization
--white_level     RAW white level for normalization
--gpu_id          physical GPU id through CUDA_VISIBLE_DEVICES
--save_rgb        save denoised PNG visualization during inference
--vis_data        save noisy PNG visualization during inference
```

For reproducible raw-only inference, use:

```bash
--save_rgb False --vis_data False
```
