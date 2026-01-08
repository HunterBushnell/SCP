#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: clean_old_logs.sh [--path DIR] [--keep-days N] [--force] [--status]

Cleans items in logs/old. Default is a dry run.

  --path DIR     Target directory (default: <repo>/logs/old)
  --keep-days N  Only remove items older than N days
  --force        Actually delete files/directories
  --status       Also clean per-run status files in logs/status
  -h, --help     Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/logs/old"
TARGET_DIR=""
KEEP_DAYS=""
DO_DELETE=0
DO_STATUS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            TARGET_DIR="${2:-}"
            shift 2
            ;;
        --keep-days)
            KEEP_DAYS="${2:-}"
            shift 2
            ;;
        --force|-f)
            DO_DELETE=1
            shift
            ;;
        --status)
            DO_STATUS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

TARGET_DIR="${TARGET_DIR:-$DEFAULT_DIR}"

if [[ -z "${TARGET_DIR}" || "${TARGET_DIR}" == "/" ]]; then
    echo "Refusing to operate on '${TARGET_DIR}'." >&2
    exit 2
fi
if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "Directory not found: ${TARGET_DIR}" >&2
    exit 1
fi

if [[ -n "${KEEP_DAYS}" && ! "${KEEP_DAYS}" =~ ^[0-9]+$ ]]; then
    echo "--keep-days must be a non-negative integer." >&2
    exit 2
fi

find_cmd=(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1)
if [[ -n "${KEEP_DAYS}" ]]; then
    find_cmd+=(-mtime +"${KEEP_DAYS}")
fi

mapfile -t items < <("${find_cmd[@]}")
status_dir=""
status_items=()
if [[ "${DO_STATUS}" == "1" ]]; then
    status_dir="$(cd "${TARGET_DIR}/.." && pwd)/status"
    if [[ -d "${status_dir}" ]]; then
        mapfile -t status_items < <(
            find "${status_dir}" -maxdepth 1 -type f -name "pvsst_*.status" \
                ! -name "pvsst_latest.status" ! -name "pvsst_primary.status"
        )
    fi
fi

if [[ "${DO_DELETE}" != "1" ]]; then
    if (( ${#items[@]} == 0 )); then
        echo "Nothing to clean in ${TARGET_DIR}"
    else
        echo "Dry run (no deletions). Would remove:"
        printf '%s\n' "${items[@]}"
        echo "Run with --force to delete."
    fi
    if [[ "${DO_STATUS}" == "1" ]]; then
        echo ""
        if [[ -z "${status_dir}" || ! -d "${status_dir}" ]]; then
            echo "Status directory not found: ${status_dir}"
        elif (( ${#status_items[@]} == 0 )); then
            echo "No per-run status files to clean in ${status_dir}"
        else
            echo "Status cleanup would remove:"
            printf '%s\n' "${status_items[@]}"
        fi
    fi
    exit 0
fi

if (( ${#items[@]} == 0 )); then
    echo "Nothing to clean in ${TARGET_DIR}"
else
    rm -rf -- "${items[@]}"
    echo "Removed ${#items[@]} items from ${TARGET_DIR}"
fi

if [[ "${DO_STATUS}" == "1" ]]; then
    if [[ -z "${status_dir}" || ! -d "${status_dir}" ]]; then
        echo "Status directory not found: ${status_dir}"
    elif (( ${#status_items[@]} == 0 )); then
        echo "No per-run status files to clean in ${status_dir}"
    else
        rm -f -- "${status_items[@]}"
        echo "Cleaned old status files in ${status_dir}"
    fi
fi
