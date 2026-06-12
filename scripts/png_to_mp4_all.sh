#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-./data/ELRVD_png}"
FPS="${FPS:-10}"
CRF="${CRF:-18}"
PRESET="${PRESET:-medium}"

SCENES=(
  "scene3_snake"
  "scene4_elephant"
  "scene7_giraffe"
  "scene8_pink"
  "scene10_whiteball"
)

METHODS=(
  "fastdvdnet"
  "gt"
  "noisy"
  "rvidenet"
)

VBM3D_SIGMAS=(
  "sigma_20"
  "sigma_30"
  "sigma_40"
  "sigma_50"
)

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "missing command: ${cmd}" >&2
    exit 1
  fi
}

make_video() {
  local png_dir="$1"
  local output_mp4="$2"
  local tmp_list

  if [[ ! -d "${png_dir}" ]]; then
    echo "skip missing directory: ${png_dir}" >&2
    return
  fi

  tmp_list="$(mktemp)"
  find "${png_dir}" -maxdepth 1 -type f -name "*.png" \
    | sort -V \
    | awk '{print "file '\''" $0 "'\''"}' \
    > "${tmp_list}"

  if [[ ! -s "${tmp_list}" ]]; then
    echo "skip no png files: ${png_dir}" >&2
    rm -f "${tmp_list}"
    return
  fi

  mkdir -p "$(dirname "${output_mp4}")"

  echo "input : ${png_dir}"
  echo "output: ${output_mp4}"

  ffmpeg -y \
    -r "${FPS}" \
    -f concat \
    -safe 0 \
    -i "${tmp_list}" \
    -vf "format=yuv420p" \
    -c:v libx264 \
    -crf "${CRF}" \
    -preset "${PRESET}" \
    "${output_mp4}"

  rm -f "${tmp_list}"
}

require_cmd ffmpeg
require_cmd find
require_cmd sort
require_cmd awk

for scene in "${SCENES[@]}"; do
  scene_dir="${ROOT}/${scene}"

  if [[ ! -d "${scene_dir}" ]]; then
    echo "skip missing scene: ${scene_dir}" >&2
    continue
  fi

  echo "scene=${scene}"

  for method in "${METHODS[@]}"; do
    png_dir="${scene_dir}/${method}"
    output_mp4="${png_dir}/video/${scene}_${method}.mp4"
    make_video "${png_dir}" "${output_mp4}"
  done

  for sigma in "${VBM3D_SIGMAS[@]}"; do
    png_dir="${scene_dir}/vbm3d/${sigma}/png"
    output_mp4="${png_dir}/video/${scene}_vbm3d_${sigma}.mp4"
    make_video "${png_dir}" "${output_mp4}"
  done
done

echo "done"
