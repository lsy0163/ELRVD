from __future__ import division

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def debayer_visualize(raw_2d, debayer, device, black_level, white_level, gain):
    import torch

    raw = raw_2d.astype(np.float32)
    raw = np.maximum(raw - black_level, 0) / (white_level - black_level)
    raw = np.clip(raw * gain, 0, 1)

    raw_tensor = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        rgb = debayer(raw_tensor)
        rgb = torch.clamp(rgb, 0, 1)
        rgb = torch.pow(rgb, 1 / 2.2)
        rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Convert Bayer RAW files to PNG with Debayer5x5.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .raw files")
    parser.add_argument("--output_dir", required=True, help="Directory to write debayered .png files")
    parser.add_argument("--height", type=int, required=True, help="RAW image height")
    parser.add_argument("--width", type=int, required=True, help="RAW image width")
    parser.add_argument("--gain", type=float, default=1.0, help="Visualization gain before debayer")
    parser.add_argument("--black_level", type=float, default=200.0, help="RAW black level")
    parser.add_argument("--white_level", type=float, default=2**12 - 1, help="RAW white level")
    parser.add_argument("--gpu_id", default="", help="GPU id to use. Empty string forces CPU.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of frames to convert. 0 means all.")
    parser.add_argument("--pattern", default="*.raw", help="Glob pattern for input RAW files")
    parser.add_argument("--debayer_layout", choices=("RGGB", "GBRG"), default="RGGB", help="Bayer layout for debayer.")
    parser.add_argument(
        "--output_name_format",
        choices=("stem", "frame"),
        default="stem",
        help="Use input stem or frame_00000 style output names.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    from debayer import Debayer5x5, Layout

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(input_dir.glob(args.pattern))
    if not raw_files:
        raise FileNotFoundError(f"No .raw files found in {input_dir}")
    if args.limit > 0:
        raw_files = raw_files[: args.limit]

    expected_bytes = args.height * args.width * np.dtype(np.uint16).itemsize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debayer = Debayer5x5(layout=getattr(Layout, args.debayer_layout)).to(device).eval()

    print(
        "input={} output={} frames={} shape={}x{} gain={} black={} white={} layout={} cuda_available={} device={}".format(
            input_dir,
            output_dir,
            len(raw_files),
            args.width,
            args.height,
            args.gain,
            args.black_level,
            args.white_level,
            args.debayer_layout,
            torch.cuda.is_available(),
            device,
        ),
        flush=True,
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    for frame_idx, raw_path in enumerate(raw_files):
        size = raw_path.stat().st_size
        if size != expected_bytes:
            raise ValueError(f"{raw_path} size {size} != expected {expected_bytes}")

        raw = np.fromfile(raw_path, dtype=np.uint16).reshape(args.height, args.width)
        rgb = debayer_visualize(
            raw,
            debayer,
            device,
            black_level=args.black_level,
            white_level=args.white_level,
            gain=args.gain,
        )

        if args.output_name_format == "frame":
            output_name = f"frame_{frame_idx:05d}.png"
        else:
            output_name = f"{raw_path.stem}.png"
        output_path = output_dir / output_name
        ok = cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write {output_path}")

        if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or frame_idx + 1 == len(raw_files):
            print(f"converted {frame_idx + 1}/{len(raw_files)}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
