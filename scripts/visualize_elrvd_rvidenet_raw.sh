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

INPUT_ROOT="${INPUT_ROOT:-./data/ELRVD_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./data/ELRVD_png}"

GAIN=1.0
WHITE_LEVEL=4095
BLACK_LEVEL_1080P=240
BLACK_LEVEL_4K=200
GPU_ID="${GPU_ID:-0}"

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

convert_scene() {
  local scene="$1"
  local height="$2"
  local width="$3"
  local black_level="$4"

  local input_dir="${INPUT_ROOT}/${scene}/rvidenet"
  local output_dir="${OUTPUT_ROOT}/${scene}/rvidenet"

  if [[ ! -d "${input_dir}" ]]; then
    echo "missing input directory: ${input_dir}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"

  echo "scene=${scene} shape=${width}x${height} black=${black_level} white=${WHITE_LEVEL} gain=${GAIN}"
  echo "  input : ${input_dir}"
  echo "  output: ${output_dir}"

  "${PYTHON_BIN}" "${CONVERTER}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --height "${height}" \
    --width "${width}" \
    --black_level "${black_level}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --gpu_id "${GPU_ID}" \
    --output_name_format frame
}

# for scene in "${SCENES_1080P[@]}"; do
#   convert_scene "${scene}" 1080 1920 "${BLACK_LEVEL_1080P}"
# done

for scene in "${SCENES_4K[@]}"; do
  convert_scene "${scene}" 2160 3840 "${BLACK_LEVEL_4K}"
done

echo "done"
