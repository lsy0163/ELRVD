#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ELRVD" ]]; then
  echo "Please run: conda activate ELRVD" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_SCRIPT="${REPO_DIR}/inference.py"
MODEL_PATH="${REPO_DIR}/inference/models/denoiser/model_epoch500.pth"

INPUT_ROOT="${INPUT_ROOT:-/database2/iyj0121/ReCRVD_sample/ReCRVD_raw}"

HEIGHT=1080
WIDTH=1920
GPU_ID="${GPU_ID:-1}"
GAIN=1.0
BLACK_LEVEL=240
WHITE_LEVEL=4095
PATCH_SIZE=256
PATCH_OVERLAP=64
DEBAYER_LAYOUT="GBRG"
SAVE_RGB=False
VIS_DATA=False
INPUT_PATTERN="wb_noisy_*_0.raw"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for scene_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${scene_dir}" ]] || continue

  scene="$(basename "${scene_dir}")"
  input_dir="${scene_dir}/noisy"
  output_dir="${scene_dir}/rvidenet"

  if [[ ! -d "${input_dir}" ]]; then
    echo "skip ${scene}: missing noisy directory: ${input_dir}" >&2
    continue
  fi

  mkdir -p "${output_dir}"

  echo "scene=${scene} shape=${WIDTH}x${HEIGHT} black=${BLACK_LEVEL} white=${WHITE_LEVEL}"
  echo "  input : ${input_dir}"
  echo "  pattern: ${INPUT_PATTERN}"
  echo "  output: ${output_dir}"

  "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    --gpu_id "${GPU_ID}" \
    --model_path "${MODEL_PATH}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --black_level "${BLACK_LEVEL}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --patch_size "${PATCH_SIZE}" \
    --patch_overlap "${PATCH_OVERLAP}" \
    --debayer_layout "${DEBAYER_LAYOUT}" \
    --input_dir "${input_dir}" \
    --input_pattern "${INPUT_PATTERN}" \
    --output_dir "${output_dir}" \
    --rgb_output_dir "${output_dir}/_rgb_unused" \
    --noisy_rgb_dir "${output_dir}/_noisy_rgb_unused" \
    --save_rgb "${SAVE_RGB}" \
    --vis_data "${VIS_DATA}"
done

echo "done"
