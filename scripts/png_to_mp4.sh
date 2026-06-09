#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  png_to_mp4.sh --png_dir DIR --output_mp4 FILE [--fps 10] [--crf 18] [--preset medium]

Example:
  scripts/png_to_mp4.sh     --png_dir /path/to/frames     --output_mp4 /path/to/video/output.mp4     --fps 10
EOF
}

PNG_DIR=""
OUTPUT_MP4=""
FPS=10
CRF=18
PRESET="medium"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --png_dir)
      PNG_DIR="$2"
      shift 2
      ;;
    --output_mp4)
      OUTPUT_MP4="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --crf)
      CRF="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PNG_DIR}" || -z "${OUTPUT_MP4}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -d "${PNG_DIR}" ]]; then
  echo "missing png_dir: ${PNG_DIR}" >&2
  exit 1
fi

TMP_LIST="$(mktemp)"
find "${PNG_DIR}" -maxdepth 1 -type f -name "*.png"   | sort -V   | awk '{ printf "file '''%s'''\n", $0 }'   > "${TMP_LIST}"

if [[ ! -s "${TMP_LIST}" ]]; then
  echo "no png files found: ${PNG_DIR}" >&2
  rm -f "${TMP_LIST}"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_MP4}")"

ffmpeg -y   -r "${FPS}"   -f concat   -safe 0   -i "${TMP_LIST}"   -vf "format=yuv420p"   -c:v libx264   -crf "${CRF}"   -preset "${PRESET}"   "${OUTPUT_MP4}"

rm -f "${TMP_LIST}"
echo "saved: ${OUTPUT_MP4}"
