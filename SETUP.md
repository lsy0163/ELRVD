# Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate ELRVD
```

If you already have a compatible PyTorch/CUDA environment, install only the Python package dependencies:

```bash
pip install -r requirements.txt
```

## DCNv2

`inference.py` imports `modules/DCNv2_latest`. Build the CUDA extension in your environment before running inference.

If you use a separately built DCNv2 directory, expose it with:

```bash
export ELRVD_DCNV2_PATH=/path/to/DCNv2_latest
```

## Checkpoints

Expected checkpoint paths:

```text
inference/models/denoiser/model_epoch500.pth
inference/models/isp/model_epoch770.pth
```

The denoiser path can be overridden:

```bash
python inference.py --model_path /path/to/model_epoch500.pth ...
```

## Data

Large input/output data should not be committed. Keep RAW/PNG/MP4 files outside the repository or under ignored directories such as:

```text
inference/data/
samples/
```
