# Model Checkpoints

Place the model checkpoints in this directory structure:

```text
inference/
  models/
    denoiser/
      model_epoch500.pth
    isp/
      model_epoch770.pth
```

`inference.py` uses the denoiser checkpoint by default:

```bash
./inference/models/denoiser/model_epoch500.pth
```

You can override it with:

```bash
python inference.py --model_path /path/to/model_epoch500.pth ...
```
