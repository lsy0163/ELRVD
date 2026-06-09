#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  copy_one_raw_per_frame.sh --input_root DIR --output_dir DIR [options]

Copies one RAW file from each frame directory into one output directory.

Expected default structure:
  INPUT_ROOT/scene4_frame0/long/dump_bayer_frame_00000.raw
  INPUT_ROOT/scene4_frame1/long/dump_bayer_frame_00000.raw
  ...

Options:
  --input_root DIR       Root directory containing frame directories. Required.
  --output_dir DIR       Destination directory. Required.
  --frame_glob GLOB      Frame directory glob relative to input_root. Default: "*_frame*"
  --raw_subdir DIR       RAW subdirectory inside each frame directory. Default: "long"
  --raw_glob GLOB        RAW glob inside raw_subdir. Default: "*.raw"
  --raw_index N          Zero-based index after sorting RAW files. Default: 0
  --output_prefix NAME   Output filename prefix. Default: "noisy_frame"
  --start_index N        First output frame number. Default: 0
  --sequential_names     Number outputs sequentially without preserving source frame index.
  --dry_run              Print planned copies without writing.
  -h, --help             Show this help.

Output naming:
  noisy_frame_00000.raw, noisy_frame_00001.raw, ...
EOF
}

INPUT_ROOT=""
OUTPUT_DIR=""
FRAME_GLOB="*_frame*"
RAW_SUBDIR="long"
RAW_GLOB="*.raw"
RAW_INDEX=0
OUTPUT_PREFIX="noisy_frame"
START_INDEX=0
DRY_RUN=0
PRESERVE_FRAME_INDEX=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --frame_glob)
      FRAME_GLOB="$2"
      shift 2
      ;;
    --raw_subdir)
      RAW_SUBDIR="$2"
      shift 2
      ;;
    --raw_glob)
      RAW_GLOB="$2"
      shift 2
      ;;
    --raw_index)
      RAW_INDEX="$2"
      shift 2
      ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --start_index)
      START_INDEX="$2"
      shift 2
      ;;
    --sequential_names)
      PRESERVE_FRAME_INDEX=0
      shift
      ;;
    --dry_run)
      DRY_RUN=1
      shift
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

if [[ -z "${INPUT_ROOT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "--input_root and --output_dir are required" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "missing input_root: ${INPUT_ROOT}" >&2
  exit 1
fi

if ! [[ "${RAW_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--raw_index must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "${START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "--start_index must be a non-negative integer" >&2
  exit 1
fi

mapfile -t FRAME_DIRS < <(find "${INPUT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${FRAME_GLOB}" | sort -V)

if [[ "${#FRAME_DIRS[@]}" -eq 0 ]]; then
  echo "no frame directories found: ${INPUT_ROOT}/${FRAME_GLOB}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_DIR}"
fi

copied=0
for frame_dir in "${FRAME_DIRS[@]}"; do
  raw_dir="${frame_dir}/${RAW_SUBDIR}"
  if [[ ! -d "${raw_dir}" ]]; then
    echo "skip missing raw dir: ${raw_dir}" >&2
    continue
  fi

  mapfile -t RAW_FILES < <(find "${raw_dir}" -maxdepth 1 -type f -name "${RAW_GLOB}" | sort -V)
  if [[ "${#RAW_FILES[@]}" -le "${RAW_INDEX}" ]]; then
    echo "skip not enough raw files: ${raw_dir} index=${RAW_INDEX}" >&2
    continue
  fi

  src="${RAW_FILES[${RAW_INDEX}]}"
  if [[ "${PRESERVE_FRAME_INDEX}" -eq 1 && "$(basename "${frame_dir}")" =~ frame([0-9]+)$ ]]; then
    frame_number="${BASH_REMATCH[1]}"
  else
    frame_number=$((START_INDEX + copied))
  fi
  dst="${OUTPUT_DIR}/${OUTPUT_PREFIX}_$(printf '%05d' "${frame_number}").raw"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "cp -a ${src} ${dst}"
  else
    cp -a "${src}" "${dst}"
    echo "copied ${copied}: ${src} -> ${dst}"
  fi
  copied=$((copied + 1))
done

echo "done copied=${copied} output=${OUTPUT_DIR}"
