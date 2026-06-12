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
GPU_ID="${GPU_ID:-1}"
DEBAYER_LAYOUT="GBRG"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for scene_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${scene_dir}" ]] || continue

  scene="$(basename "${scene_dir}")"
  input_dir="${scene_dir}/rvidenet"
  output_dir="${OUTPUT_ROOT}/${scene}/rvidenet"

  if [[ ! -d "${input_dir}" ]]; then
    echo "skip ${scene}: missing rvidenet directory: ${input_dir}" >&2
    continue
  fi

  mkdir -p "${output_dir}"

  echo "scene=${scene} rvidenet shape=${WIDTH}x${HEIGHT} black=${BLACK_LEVEL} white=${WHITE_LEVEL} gain=${GAIN} layout=${DEBAYER_LAYOUT}"
  echo "  input : ${input_dir}"
  echo "  output: ${output_dir}"

  "${PYTHON_BIN}" "${CONVERTER}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --pattern "*.raw" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --black_level "${BLACK_LEVEL}" \
    --white_level "${WHITE_LEVEL}" \
    --gain "${GAIN}" \
    --gpu_id "${GPU_ID}" \
    --debayer_layout "${DEBAYER_LAYOUT}" \
    --output_name_format frame
done

echo "done"
