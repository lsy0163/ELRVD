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

HEIGHT=1080
WIDTH=1920
BLACK_LEVEL=240
WHITE_LEVEL=4095
GAIN=3.0

SCENES=(
  "scene2_wood"
  "scene3_snake"
  "scene4_elephant"
  "scene7_giraffe"
  "scene8_pink"
  "scene10_whiteball"
)
SUBDIRS=("gt" "noisy")
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for scene in "${SCENES[@]}"; do
  scene_dir="${INPUT_ROOT}/${scene}"

  if [[ ! -d "${scene_dir}" ]]; then
    echo "missing scene directory: ${scene_dir}" >&2
    exit 1
  fi

  for subdir in "${SUBDIRS[@]}"; do
    input_dir="${scene_dir}/${subdir}"
    output_dir="${OUTPUT_ROOT}/${scene}/${subdir}"

    if [[ ! -d "${input_dir}" ]]; then
      echo "missing input directory: ${input_dir}" >&2
      exit 1
    fi

    mkdir -p "${output_dir}"

    echo "scene=${scene} subdir=${subdir} gain=${GAIN}"
    echo "  input : ${input_dir}"
    echo "  output: ${output_dir}"

    "${PYTHON_BIN}" "${CONVERTER}" \
      --input_dir "${input_dir}" \
      --output_dir "${output_dir}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --black_level "${BLACK_LEVEL}" \
      --white_level "${WHITE_LEVEL}" \
      --gain "${GAIN}" \
      --gpu_id "${GPU_ID}" \
      --output_name_format frame
  done
done

echo "done"
