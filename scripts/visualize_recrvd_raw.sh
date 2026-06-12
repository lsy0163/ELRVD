#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ELRVD" ]]; then
  echo "Please run: conda activate ELRVD" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONVERTER="${REPO_DIR}/raw_to_debayer_png.py"

INPUT_ROOT="${INPUT_ROOT:-./data/ReCRVD_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data/ReCRVD_png}"

HEIGHT=1080
WIDTH=1920
BLACK_LEVEL=240
WHITE_LEVEL=4095
GAIN=1.0
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for scene_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${scene_dir}" ]] || continue

  scene="$(basename "${scene_dir}")"
  gt_input="${scene_dir}/gt"
  noisy_input="${scene_dir}/noisy"
  gt_output="${OUTPUT_ROOT}/${scene}/gt"
  noisy_output="${OUTPUT_ROOT}/${scene}/noisy"

  if [[ ! -d "${gt_input}" ]]; then
    echo "missing gt directory: ${gt_input}" >&2
    exit 1
  fi
  if [[ ! -d "${noisy_input}" ]]; then
    echo "missing noisy directory: ${noisy_input}" >&2
    exit 1
  fi

  mkdir -p "${gt_output}" "${noisy_output}"

  echo "scene=${scene} gt"
  echo "  input : ${gt_input}"
  echo "  output: ${gt_output}"
  "${PYTHON_BIN}" "${CONVERTER}" \
    --input_dir "${gt_input}" \
    --output_dir "${gt_output}" \
    --pattern "*.raw" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --black_level "${BLACK_LEVEL}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --gpu_id "${GPU_ID}"

  echo "scene=${scene} noisy sample index 0"
  echo "  input : ${noisy_input}"
  echo "  output: ${noisy_output}"
  "${PYTHON_BIN}" "${CONVERTER}" \
    --input_dir "${noisy_input}" \
    --output_dir "${noisy_output}" \
    --pattern "wb_noisy_*_0.raw" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --black_level "${BLACK_LEVEL}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --gpu_id "${GPU_ID}"
done

echo "done"
