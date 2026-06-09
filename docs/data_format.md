# Data Format

The inference pipeline expects a sequence of 16-bit Bayer RAW files.

## Required metadata

For each dataset or scene, set:

```text
height
width
black_level
white_level
debayer_layout: RGGB or GBRG
```

## Example: ELRVD-style scenes

```text
ELRVD_raw/
  scene3_snake/
    noisy/
      noisy_frame_00000.raw
      noisy_frame_00001.raw
    rvidenet/
      denoised_raw_frame_000000.raw
```

## Example: ReCRVD-style scenes

Only noisy sample index 0 is used by the provided ReCRVD script:

```text
ReCRVD_raw/
  Beauty/
    noisy/
      wb_noisy_1_0.raw
      wb_noisy_2_0.raw
    rvidenet/
      denoised_raw_frame_000000.raw
```
