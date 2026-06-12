# Data Format

The inference pipeline expects a sequence of 16-bit Bayer RAW files.

## Limitations

- 16-bit single-channel Bayer RAW only (read as raw `uint16`); no DNG/TIFF/8-bit/RGB.
- RGGB Bayer layout only. The model packs input as RGGB; `--debayer_layout` only changes PNG visualization, not the model input.
- Height and width must be even.
- Input must be at least ~2× `--patch_size` per side (default patch 256 → min ~512×512); reduce `--patch_size` for smaller frames.
- `height`, `width`, `black_level`, `white_level` are passed as arguments, not read from the file, and must match the data exactly.

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
