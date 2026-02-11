#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATUS_FILE="${STATUS_FILE:-}"
STATUS_LATEST_FILE="${STATUS_LATEST_FILE:-}"
STATUS_PRIMARY_FILE="${STATUS_PRIMARY_FILE:-}"
STATUS_IS_PRIMARY="${STATUS_IS_PRIMARY:-1}"
PARTS_DIR="${PARTS_DIR:-}"
RUN_TAG="${RUN_TAG:-${ORIG_ARRAY_JOB_ID:-}}"

allow_update_file() {
    local file="$1"
    local state="$2"
    local latest_tag
    if [[ -f "${file}" ]]; then
        latest_tag=$(grep -m1 "^run_tag=" "${file}" 2>/dev/null | cut -d= -f2- || true)
        if [[ -n "${latest_tag:-}" && "${latest_tag}" == "${RUN_TAG:-}" ]]; then
            if grep -q "^state=ERROR" "${file}" && [[ "${state}" != "ERROR" ]]; then
                return 1
            fi
        fi
    fi
    return 0
}

write_status() {
    local state="$1"
    local msg="${2:-}"
    local now
    local update_latest=0
    local update_primary=0
    now=$(date -Iseconds 2>/dev/null || date)
    if [[ -z "${STATUS_FILE}" ]]; then
        return
    fi
    mkdir -p "$(dirname "${STATUS_FILE}")" 2>/dev/null || true
    {
        echo "state=${state}"
        echo "time=${now}"
        echo "run_tag=${RUN_TAG:-}"
        echo "job_name=pvsst"
        echo "job_id=${SLURM_JOB_ID:-}"
        echo "array_job_id=${ORIG_ARRAY_JOB_ID:-}"
        echo "array_task_id="
        echo "tune_dir=${TUNE_DIR:-}"
        echo "output_dir=${RESULTS_DIR:-}"
        echo "run_root=${RUN_ROOT:-}"
        echo "output_stem=${RUN_OUTPUT_STEM:-}"
        echo "message=${msg}"
    } > "${STATUS_FILE}" 2>/dev/null || true
    if [[ "${STATUS_IS_PRIMARY}" == "1" ]]; then
        update_latest=1
        update_primary=1
    elif [[ "${state}" == "ERROR" ]]; then
        update_latest=1
    fi
    if [[ "${update_latest}" == "1" && -n "${STATUS_LATEST_FILE}" && "${STATUS_LATEST_FILE}" != "${STATUS_FILE}" ]]; then
        if allow_update_file "${STATUS_LATEST_FILE}" "${state}"; then
            cp -f "${STATUS_FILE}" "${STATUS_LATEST_FILE}" 2>/dev/null || true
        fi
    fi
    if [[ "${update_primary}" == "1" && -n "${STATUS_PRIMARY_FILE}" && "${STATUS_PRIMARY_FILE}" != "${STATUS_FILE}" ]]; then
        if allow_update_file "${STATUS_PRIMARY_FILE}" "${state}"; then
            cp -f "${STATUS_FILE}" "${STATUS_PRIMARY_FILE}" 2>/dev/null || true
        fi
    fi
}

write_status "MERGING" "merge_start"

if python "${SCRIPT_DIR}/merge_array_results.py" "$@"; then
    write_status "SUCCESS" "merged"
    if [[ -n "${PARTS_DIR}" && -d "${PARTS_DIR}" ]]; then
        rm -rf "${PARTS_DIR}"
    fi
else
    rc=$?
    write_status "ERROR" "merge_exit=${rc}"
    exit "${rc}"
fi
