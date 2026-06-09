from __future__ import division

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_DIR / "inference/models/denoiser/model_epoch500.pth"
DEFAULT_BLACK_LEVEL = 240.0
DEFAULT_WHITE_LEVEL = float(2**12 - 1)
DEFAULT_PATCH_SIZE = 256
DEFAULT_PATCH_OVERLAP = 64


def add_dcnv2_paths():
    dcnv2_dirs = [REPO_DIR / "modules/DCNv2_latest"]
    external_dcnv2_dir = os.environ.get("ELRVD_DCNV2_PATH")
    if external_dcnv2_dir:
        dcnv2_dirs.append(Path(external_dcnv2_dir))
    for dcnv2_dir in dcnv2_dirs:
        dcnv2_dir = str(dcnv2_dir)
        if os.path.isdir(dcnv2_dir) and dcnv2_dir not in sys.path:
            sys.path.insert(0, dcnv2_dir)


def set_cuda_visible_devices_from_argv():
    gpu_id = "0"
    for idx, arg in enumerate(sys.argv):
        if arg == "--gpu_id" and idx + 1 < len(sys.argv):
            gpu_id = sys.argv[idx + 1]
            break
        if arg.startswith("--gpu_id="):
            gpu_id = arg.split("=", 1)[1]
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


add_dcnv2_paths()
set_cuda_visible_devices_from_argv()

import torch
from debayer import Debayer5x5, Layout
from utils import test_big_size_raw


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RViDeNet raw denoising inference.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Physical GPU id exposed via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--height", type=int, default=1080, help="Input raw height.")
    parser.add_argument("--width", type=int, default=1920, help="Input raw width.")
    parser.add_argument("--black_level", type=float, default=DEFAULT_BLACK_LEVEL, help="Raw black level for normalization.")
    parser.add_argument("--white_level", type=float, default=DEFAULT_WHITE_LEVEL, help="Raw white level for normalization.")
    parser.add_argument("--gain", type=float, default=1.0, help="Visualization gain. Only affects PNG outputs.")
    parser.add_argument("--input_dir", type=str, default="./inference/data/input/", help="Directory containing input .raw files.")
    parser.add_argument("--input_pattern", type=str, default="*.raw", help="Glob pattern for input raw files.")
    parser.add_argument("--output_dir", type=str, default="./inference/data/output/", help="Directory for denoised .raw files.")
    parser.add_argument("--rgb_output_dir", type=str, default="./inference/data/rgb_output/", help="Directory for denoised PNG outputs.")
    parser.add_argument("--noisy_rgb_dir", type=str, default="./inference/data/rgb_input/", help="Directory for noisy PNG outputs.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to denoiser model checkpoint.")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch height and width for tiled inference.")
    parser.add_argument("--patch_overlap", type=int, default=DEFAULT_PATCH_OVERLAP, help="Patch overlap for tiled inference.")
    parser.add_argument("--debayer_layout", choices=("RGGB", "GBRG"), default="RGGB", help="Bayer layout used for PNG visualization.")
    parser.add_argument("--vis_data", type=str2bool, default=False, help="Save input noisy PNG visualizations.")
    parser.add_argument("--save_rgb", type=str2bool, default=True, help="Save denoised PNG visualizations.")
    return parser.parse_args()


def prepare_cuda(args):
    print("requested gpu_id:", args.gpu_id, flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
    print("torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
    print("torch.cuda.device_count:", torch.cuda.device_count(), flush=True)
    print("raw shape:", "{}x{}".format(args.width, args.height), flush=True)
    print("black/white level:", "{}/{}".format(args.black_level, args.white_level), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check conda env, CUDA_VISIBLE_DEVICES, and DCNv2 CUDA build.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device is visible to PyTorch.")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print("torch cuda current_device:", torch.cuda.current_device(), flush=True)
    print("torch cuda device_name:", torch.cuda.get_device_name(0), flush=True)
    return device


def normalize_raw(raw, black_level, white_level):
    raw = raw.astype(np.float32)
    return np.maximum(raw - black_level, 0) / (white_level - black_level)


def pack_rggb_raw(raw, black_level, white_level):
    normalized = normalize_raw(raw, black_level, white_level)
    normalized = np.expand_dims(normalized, axis=2)
    height, width, _ = normalized.shape
    return np.concatenate(
        (
            normalized[0:height:2, 0:width:2, :],
            normalized[0:height:2, 1:width:2, :],
            normalized[1:height:2, 1:width:2, :],
            normalized[1:height:2, 0:width:2, :],
        ),
        axis=2,
    )


def depack_rggb_raw(packed_raw):
    height, width, _ = packed_raw.shape
    output = np.zeros((height * 2, width * 2), dtype=packed_raw.dtype)
    output[0 : height * 2 : 2, 0 : width * 2 : 2] = packed_raw[:, :, 0]
    output[0 : height * 2 : 2, 1 : width * 2 : 2] = packed_raw[:, :, 1]
    output[1 : height * 2 : 2, 1 : width * 2 : 2] = packed_raw[:, :, 2]
    output[1 : height * 2 : 2, 0 : width * 2 : 2] = packed_raw[:, :, 3]
    return output


def restore_raw(normalized_raw, black_level, white_level):
    restored = normalized_raw * (white_level - black_level) + black_level
    return np.clip(restored, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def read_raw(raw_path, height, width):
    raw = np.fromfile(raw_path, dtype=np.uint16)
    expected_pixels = height * width
    if raw.size != expected_pixels:
        raise ValueError("{} has {} pixels, expected {}".format(raw_path, raw.size, expected_pixels))
    return raw.reshape(height, width)


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def get_neighbor_paths(input_files, frame_idx):
    paths = []
    for offset in (-1, 0, 1):
        idx = min(max(frame_idx + offset, 0), len(input_files) - 1)
        paths.append(input_files[idx])
    return paths


def make_model_input(input_files, frame_idx, args):
    frame_list = []
    for raw_path in get_neighbor_paths(input_files, frame_idx):
        raw = read_raw(raw_path, args.height, args.width)
        packed = pack_rggb_raw(raw, args.black_level, args.white_level)
        frame_list.append(np.expand_dims(packed, axis=0))
    return np.concatenate(frame_list, axis=3)


def build_debayer(layout_name, device):
    layout = getattr(Layout, layout_name)
    return Debayer5x5(layout=layout).to(device).eval()


def debayer_visualize(raw_2d, debayer, device, black_level, white_level, gain):
    raw = normalize_raw(raw_2d, black_level, white_level)
    raw = np.clip(raw * gain, 0, 1)
    raw_tensor = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        rgb = debayer(raw_tensor)
        rgb = torch.clamp(rgb, 0, 1)
        rgb = torch.pow(rgb, 1 / 2.2)
        rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


def save_png(rgb, output_path):
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to write {}".format(output_path))


def run_inference(args):
    device = prepare_cuda(args)
    model = torch.load(os.path.expanduser(args.model_path)).to(device).eval()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_pattern)), key=natural_sort_key)
    if not input_files:
        raise FileNotFoundError("No raw files matched {} in {}".format(args.input_pattern, args.input_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_output_dir = Path(args.rgb_output_dir)
    noisy_rgb_dir = Path(args.noisy_rgb_dir)
    if args.save_rgb:
        rgb_output_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_data:
        noisy_rgb_dir.mkdir(parents=True, exist_ok=True)

    debayer = build_debayer(args.debayer_layout, device) if args.save_rgb or args.vis_data else None

    for frame_idx, input_file in enumerate(input_files):
        print("processing frame {}".format(frame_idx), flush=True)
        input_data = make_model_input(input_files, frame_idx, args)

        with torch.no_grad():
            test_result = test_big_size_raw(
                input_data,
                model,
                patch_h=args.patch_size,
                patch_w=args.patch_size,
                patch_h_overlap=args.patch_overlap,
                patch_w_overlap=args.patch_overlap,
            )

        test_result = depack_rggb_raw(test_result.squeeze())
        print("test_result shape:", test_result.shape, flush=True)

        denoised_raw = restore_raw(test_result, args.black_level, args.white_level)
        raw_output_path = output_dir / "denoised_raw_frame_{:06d}.raw".format(frame_idx)
        denoised_raw.tofile(raw_output_path)

        if args.save_rgb:
            denoised_rgb = debayer_visualize(
                denoised_raw,
                debayer,
                device,
                args.black_level,
                args.white_level,
                args.gain,
            )
            print("denoised_srgb min/max:", denoised_rgb.min(), denoised_rgb.max(), flush=True)
            save_png(denoised_rgb, rgb_output_dir / "frame_{:06d}_denoised_sRGB.png".format(frame_idx))

        if args.vis_data:
            noisy_raw = read_raw(input_file, args.height, args.width)
            noisy_rgb = debayer_visualize(
                noisy_raw,
                debayer,
                device,
                args.black_level,
                args.white_level,
                args.gain,
            )
            save_png(noisy_rgb, noisy_rgb_dir / "frame_{:06d}_noisy_sRGB.png".format(frame_idx))


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
