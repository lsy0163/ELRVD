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

INPUT_ROOT="${INPUT_ROOT:-./data/ELRVD_raw}"

GPU_ID="${GPU_ID:-0}"
GAIN=3.0
BLACK_LEVEL_1080P=240
BLACK_LEVEL_4K=200
WHITE_LEVEL=4095
PATCH_SIZE=256
PATCH_OVERLAP=64
DEBAYER_LAYOUT="RGGB"
SAVE_RGB=False
VIS_DATA=False

SCENES_1080P=(
  "scene2_wood"
  "scene3_snake"
  "scene4_elephant"
  "scene7_giraffe"
  "scene8_pink"
  "scene10_whiteball"
)

SCENES_4K=(
  "vds_lab_scene1"
  "vds_lab_scene2"
)

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_scene() {
  local scene="$1"
  local height="$2"
  local width="$3"
  local black_level="$4"

  input_dir="${INPUT_ROOT}/${scene}/noisy"
  output_dir="${INPUT_ROOT}/${scene}/rvidenet"

  if [[ ! -d "${input_dir}" ]]; then
    echo "missing input directory: ${input_dir}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"

  echo "scene=${scene} shape=${width}x${height} black=${black_level} white=${WHITE_LEVEL}"
  echo "  input : ${input_dir}"
  echo "  output: ${output_dir}"

  "${PYTHON_BIN}" "${INFERENCE_SCRIPT}" \
    --gpu_id "${GPU_ID}" \
    --model_path "${MODEL_PATH}" \
    --height "${height}" \
    --width "${width}" \
    --black_level "${black_level}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --patch_size "${PATCH_SIZE}" \
    --patch_overlap "${PATCH_OVERLAP}" \
    --debayer_layout "${DEBAYER_LAYOUT}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --rgb_output_dir "${output_dir}/_rgb_unused" \
    --noisy_rgb_dir "${output_dir}/_noisy_rgb_unused" \
    --save_rgb "${SAVE_RGB}" \
    --vis_data "${VIS_DATA}"
}

for scene in "${SCENES_1080P[@]}"; do
  run_scene "${scene}" 1080 1920 "${BLACK_LEVEL_1080P}"
done

for scene in "${SCENES_4K[@]}"; do
  run_scene "${scene}" 2160 3840 "${BLACK_LEVEL_4K}"
done

echo "done"
