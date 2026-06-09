# ELRVD Results: RAW Video Denoising with RViDeNet

This repository contains the inference, visualization, and reference training code used for RAW video denoising experiments with RViDeNet-style models.

The main pipeline is:

```text
noisy Bayer RAW frames -> RViDeNet inference -> denoised RAW frames -> debayer PNG -> MP4 visualization
```

## Repository Structure

```text
inference.py                  # main RViDeNet inference entry point
raw_to_debayer_png.py         # RAW Bayer to PNG visualization helper
models.py                     # model definitions
utils.py                      # tiled inference and utility functions
scripts/                      # inference, visualization, and video conversion scripts
train/                        # reference training scripts
modules/DCNv2_latest/         # DCNv2 CUDA extension source
inference/models/             # inference checkpoints and checkpoint README
docs/                         # data format and usage notes
```

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate ELRVD
```

Build DCNv2 after activating the environment:

```bash
cd modules/DCNv2_latest
bash make.sh
cd ../..
```

See [SETUP.md](SETUP.md) for details.

## Checkpoints

Expected checkpoint paths:

```text
inference/models/denoiser/model_epoch500.pth
inference/models/isp/model_epoch770.pth
```

You can override the denoiser checkpoint path:

```bash
python inference.py --model_path /path/to/model_epoch500.pth ...
```

## Data Format

Input RAW frames are expected as 16-bit Bayer RAW files. The frame size, black level, white level, and Bayer layout must match the dataset.

Example noisy input structure:

```text
ELRVD_raw/
  scene3_snake/
    noisy/
      noisy_frame_00000.raw
      noisy_frame_00001.raw
```

See [docs/data_format.md](docs/data_format.md).

## Inference

Single command example:

```bash
python inference.py   --input_dir /path/to/scene/noisy   --output_dir /path/to/scene/rvidenet   --height 1080   --width 1920   --black_level 240   --white_level 4095   --gpu_id 0   --save_rgb False   --vis_data False
```

Batch scripts are available under `scripts/`.

```bash
INPUT_ROOT=/path/to/ELRVD_raw GPU_ID=0 scripts/run_elrvd_rvidenet_inference.sh
INPUT_ROOT=/path/to/ReCRVD_raw GPU_ID=1 scripts/run_recrvd_rvidenet_inference.sh
```

## Visualization

Convert denoised RAW frames to PNG:

```bash
python raw_to_debayer_png.py   --input_dir /path/to/scene/rvidenet   --output_dir /path/to/scene_png/rvidenet   --height 1080   --width 1920   --black_level 240   --white_level 4095   --gain 3.0   --debayer_layout RGGB   --output_name_format frame
```

Convert PNG frames to MP4:

```bash
scripts/png_to_mp4.sh   --png_dir /path/to/png_frames   --output_mp4 /path/to/output.mp4   --fps 10
```

## Training

Reference training scripts are provided in [train/](train/). They may require dataset paths and experiment settings to be adjusted before use.

## Acknowledgements

This project builds on ELRVD/RViDeNet-style RAW video denoising code and DCNv2 modules. Please cite or acknowledge the original works when using this repository.
