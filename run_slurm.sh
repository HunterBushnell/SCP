#!/bin/bash
# Simple SLURM wrapper for the SCP single-cell pipeline.
# Adjust SBATCH fields to your cluster defaults.

#SBATCH -J pvsst
#SBATCH -o logs/pvsst_%A_%a.out
#SBATCH -e logs/pvsst_%A_%a.err
#SBATCH -t 0-03:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# To overwrite output files, uncomment the following lines:
# #SBATCH -o logs/pvsst.out
# #SBATCH -e logs/pvsst.err

# Optional: load modules / activate env if needed
# module load python/3.10
# source ~/miniconda3/bin/activate <env-name>

set -euo pipefail

# Capture submit dir before changing cwd (SLURM writes logs relative to submit dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_SRC_DIR="${SUBMIT_DIR}/logs"

# For arrays, SLURM_JOB_ID is per-task; SLURM_ARRAY_JOB_ID is the shared batch id.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"

# Status tracking (per-job/task file written in logs/status/)
STATUS_DIR="${STATUS_DIR:-${LOG_SRC_DIR}/status}"
MANUAL_TAG=""
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    MANUAL_TAG="manual_$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    STATUS_TAG="${ARRAY_JOB_ID:-${MANUAL_TAG}}_${SLURM_ARRAY_TASK_ID}"
else
    STATUS_TAG="${ARRAY_JOB_ID:-${MANUAL_TAG}}_0"
fi
RUN_TAG="${ARRAY_JOB_ID:-${MANUAL_TAG}}"
STATUS_FILE="${STATUS_FILE:-${STATUS_DIR}/pvsst_${STATUS_TAG}.status}"
STATUS_LATEST_FILE="${STATUS_LATEST_FILE:-${STATUS_DIR}/pvsst_latest.status}"
STATUS_PRIMARY_FILE="${STATUS_PRIMARY_FILE:-${STATUS_DIR}/pvsst_primary.status}"
STATUS_IS_PRIMARY=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    STATUS_IS_PRIMARY=0
fi

# Clean old per-run status files (keep stable latest/primary files)
CLEAN_STATUS="${CLEAN_STATUS:-1}"
if [[ "${CLEAN_STATUS}" == "1" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
        if [[ -d "${STATUS_DIR}" ]]; then
            shopt -s nullglob
            for f in "${STATUS_DIR}"/pvsst_*.status; do
                base="$(basename "${f}")"
                case "${base}" in
                    "$(basename "${STATUS_LATEST_FILE}")") ;;
                    "$(basename "${STATUS_PRIMARY_FILE}")") ;;
                    *) rm -f "${f}" ;;
                esac
            done
            shopt -u nullglob
        fi
    fi
fi

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
    mkdir -p "${STATUS_DIR}" 2>/dev/null || true
    {
        echo "state=${state}"
        echo "time=${now}"
        echo "run_tag=${RUN_TAG:-}"
        echo "job_name=pvsst"
        echo "job_id=${SLURM_JOB_ID:-}"
        echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
        echo "array_task_id=${SLURM_ARRAY_TASK_ID:-}"
        echo "tune_dir=${TUNE_DIR:-}"
        echo "output_dir=${RESULTS_DIR:-}"
        echo "run_root=${RUN_ROOT:-}"
        echo "output_stem=${RUN_OUTPUT_STEM:-}"
        echo "message=${msg}"
    } > "${STATUS_FILE}" 2>/dev/null || true
    if [[ "${state}" != "MERGE_PENDING" ]]; then
        if [[ "${STATUS_IS_PRIMARY}" == "1" ]]; then
            update_latest=1
            update_primary=1
        elif [[ "${state}" == "ERROR" ]]; then
            update_latest=1
            update_primary=1
        fi
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

on_exit() {
    local rc=$?
    trap - EXIT
    if [[ "${rc}" -eq 0 ]]; then
        local merge_array="${MERGE_ARRAY:-1}"
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${merge_array}" == "1" ]]; then
            write_status "MERGE_PENDING" "waiting_for_merge"
        else
            write_status "SUCCESS" "completed"
        fi
    else
        write_status "ERROR" "exit_code=${rc}"
    fi
    exit "${rc}"
}
trap on_exit EXIT
write_status "RUNNING" "starting"

# Rotate old logs so the latest run is easy to find (keep current job logs in logs/)
# For job arrays, rotate only on task 0 to avoid races.
# Set ROTATE_LOGS=1 in the environment to re-enable moving logs to logs/old.
ROTATE_LOGS="${ROTATE_LOGS:-0}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    ROTATE_LOGS=0
fi

if [[ "${ROTATE_LOGS}" == "1" ]]; then
    mkdir -p "${LOG_SRC_DIR}/old"
    if ls "${LOG_SRC_DIR}"/pvsst_*.* 1> /dev/null 2>&1; then
        ts=$(date +%Y%m%d_%H%M%S)
        mkdir -p "${LOG_SRC_DIR}/old/${ts}"
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
                current_base="pvsst_${ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            else
                current_base="pvsst_${SLURM_JOB_ID}_0"
            fi
            shopt -s nullglob
            files_to_move=()
            for f in "${LOG_SRC_DIR}"/pvsst_*.out "${LOG_SRC_DIR}"/pvsst_*.err; do
                case "$f" in
                    *"${current_base}."*) ;; # keep current job logs
                    *) files_to_move+=("$f") ;;
                esac
            done
            if (( ${#files_to_move[@]} )); then
                mv "${files_to_move[@]}" "${LOG_SRC_DIR}/old/${ts}/"
            fi
            shopt -u nullglob
        else
            # If not running under SLURM, move everything
            mv "${LOG_SRC_DIR}"/pvsst_*.* "${LOG_SRC_DIR}/old/${ts}/"
        fi
    fi
fi

# Tune selection:
#   - Set TUNE_DIR to use an explicit path, OR
#   - Set CELL and TUNE (defaults below)
CELL=${CELL:-SST}
TUNE=${TUNE:-seg_tuned}
TUNE_DIR=${TUNE_DIR:-/home/hrbncv/SCP/cells/${CELL}/tunes/${TUNE}}

OUTPUT_DIR=${OUTPUT_DIR:-${TUNE_DIR}/output_data}
MODE=${MODE:-}              # leave empty to auto-pick based on n_trials
N_TRIALS=${N_TRIALS:-}      # optional override
SEED=${SEED:-}              # optional
BASE_SEED=${BASE_SEED:-}    # optional, used with arrays
TOTAL_TRIALS=${TOTAL_TRIALS:-}  # optional, split across array tasks
TASKS=${TASKS:-}                # optional, fallback if SLURM_ARRAY_TASK_COUNT missing
MERGE_ARRAY=${MERGE_ARRAY:-1}
MERGED_STEM=${MERGED_STEM:-}
MERGE_PATTERN=${MERGE_PATTERN:-}
BATCH_STEM=${BATCH_STEM:-}
ICLAMP=${ICLAMP:-}
FORCE_SAVE=${FORCE_SAVE:-1}
SNAPSHOT=${SNAPSHOT:-}

RUN_ROOT="${OUTPUT_DIR}"
RESULTS_DIR="${OUTPUT_DIR}"
PARTS_DIR=""
RUN_OUTPUT_STEM="${OUTPUT_STEM:-}"

# Read save/output stem from sim_config.json (if enabled)
SAVE_ENABLED=0
SAVE_STEM=""
SIM_CFG_PATH="${TUNE_DIR}/cell_configs/sim_config.json"
if [[ ! -f "${SIM_CFG_PATH}" ]]; then
    SIM_CFG_PATH="${TUNE_DIR}/sim_config.json"
fi
if [[ -f "${SIM_CFG_PATH}" ]]; then
    read -r SAVE_ENABLED SAVE_STEM < <(SIM_CFG_PATH="$SIM_CFG_PATH" python - <<'PY'
import json
import os
from pathlib import Path
cfg_path = Path(os.environ.get("SIM_CFG_PATH", ""))
cfg = json.loads(cfg_path.read_text())
save_raw = cfg.get('save', cfg.get('output'))
save_output = cfg.get('save_output', None)
enabled = None
stem = None
if isinstance(save_raw, (list, tuple)):
    if len(save_raw) >= 1:
        enabled = bool(save_raw[0])
    if len(save_raw) >= 2:
        stem = save_raw[1]
elif isinstance(save_raw, dict):
    enabled = save_raw.get('enabled')
    stem = save_raw.get('path') or save_raw.get('stem') or save_raw.get('name')
else:
    if save_raw not in (None, '', False):
        enabled = True
        stem = save_raw
    else:
        stem = None
if enabled is None:
    enabled = True if save_output is None else bool(save_output)
stem = '' if stem in (None, '', False) else str(stem)
print(f"{int(enabled)} {stem}")
PY
)
fi
USE_CONFIG_STEM=0
if [[ "${SAVE_ENABLED}" == "1" && -n "${SAVE_STEM}" ]]; then
    USE_CONFIG_STEM=1
fi

cd "$TUNE_DIR"

# Build mechanisms if not already compiled (compile inside modfiles/)
if [[ ! -f "${TUNE_DIR}/modfiles/x86_64/.libs/libnrnmech.so" && ! -f "${TUNE_DIR}/modfiles/x86_64/libnrnmech.so" ]]; then
    echo "Compiling NEURON mechanisms with nrnivmodl in modfiles/..."
    if [[ ! -d "${TUNE_DIR}/modfiles" ]]; then
        echo "Missing modfiles directory in ${TUNE_DIR}" >&2
        exit 1
    fi
    (cd "${TUNE_DIR}/modfiles" && nrnivmodl)
    if [[ ! -f "${TUNE_DIR}/modfiles/x86_64/.libs/libnrnmech.so" && ! -f "${TUNE_DIR}/modfiles/x86_64/libnrnmech.so" ]]; then
        echo "Mechanism build failed: libnrnmech.so not found after nrnivmodl." >&2
        exit 1
    fi
fi

# If running as a job array, default to single-trial tasks and unique outputs
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "${MERGE_ARRAY}" == "1" ]]; then
        if [[ -z "${BATCH_STEM}" ]]; then
            if [[ "${USE_CONFIG_STEM}" == "1" ]]; then
                BATCH_STEM="${SAVE_STEM}"
            else
                BATCH_STEM="slurm_${ARRAY_JOB_ID}"
            fi
        fi
        RUN_ROOT="${OUTPUT_DIR}/${BATCH_STEM}"
        PARTS_DIR="${RUN_ROOT}/parts"
        RESULTS_DIR="${PARTS_DIR}"
    else
        if [[ -z "${BATCH_STEM}" ]]; then
            if [[ "${USE_CONFIG_STEM}" == "1" ]]; then
                BATCH_STEM="${SAVE_STEM}_${SLURM_ARRAY_TASK_ID}"
            else
                BATCH_STEM="slurm_${ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            fi
        fi
        RUN_ROOT="${OUTPUT_DIR}/${BATCH_STEM}"
        RESULTS_DIR="${RUN_ROOT}"
    fi

    if [[ -n "${TOTAL_TRIALS}" ]]; then
        task_count="${SLURM_ARRAY_TASK_COUNT:-${TASKS:-}}"
        if [[ -z "${task_count}" ]]; then
            task_count=1
        fi
        base=$((TOTAL_TRIALS / task_count))
        rem=$((TOTAL_TRIALS % task_count))
        if (( SLURM_ARRAY_TASK_ID < rem )); then
            N_TRIALS=$((base + 1))
        else
            N_TRIALS=$((base))
        fi
        if (( N_TRIALS <= 0 )); then
            echo "Task ${SLURM_ARRAY_TASK_ID}: no trials assigned, exiting."
            exit 0
        fi
        TRIAL_OFFSET=$((base * SLURM_ARRAY_TASK_ID + (SLURM_ARRAY_TASK_ID < rem ? SLURM_ARRAY_TASK_ID : rem)))
    elif [[ -z "${N_TRIALS}" ]]; then
        N_TRIALS=1
    fi
    if [[ -z "${TRIAL_OFFSET:-}" && -n "${N_TRIALS}" ]]; then
        TRIAL_OFFSET=$((N_TRIALS * SLURM_ARRAY_TASK_ID))
    fi
    if [[ "${MERGE_ARRAY}" == "1" ]]; then
        if [[ -z "${RUN_OUTPUT_STEM}" ]]; then
            if [[ "${USE_CONFIG_STEM}" == "1" ]]; then
                RUN_OUTPUT_STEM="${SAVE_STEM}_${SLURM_ARRAY_TASK_ID}"
            else
                RUN_OUTPUT_STEM="slurm_${ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            fi
        else
            RUN_OUTPUT_STEM="${RUN_OUTPUT_STEM}_${SLURM_ARRAY_TASK_ID}"
        fi
    else
        RUN_OUTPUT_STEM="results"
    fi
    if [[ -z "${SEED}" && -n "${BASE_SEED}" ]]; then
        SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
    fi
fi

# For non-array runs, ensure a stable output stem for log placement
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${RUN_OUTPUT_STEM}" ]]; then
    if [[ "${USE_CONFIG_STEM}" == "1" ]]; then
        RUN_OUTPUT_STEM="${SAVE_STEM}"
    else
        RUN_OUTPUT_STEM="slurm_${SLURM_JOB_ID}"
    fi
fi

export STATUS_FILE STATUS_LATEST_FILE STATUS_PRIMARY_FILE STATUS_IS_PRIMARY RUN_TAG

CMD=(python /home/hrbncv/SCP/run_pipeline.py
    --tune-dir "$TUNE_DIR"
    --output-dir "$RESULTS_DIR")

write_status "RUNNING" "run_pipeline"

if [[ -n "${MODE}" ]]; then
    CMD+=(--mode "$MODE")
fi
if [[ -n "${N_TRIALS}" ]]; then
    CMD+=(--n-trials "$N_TRIALS")
fi
if [[ -n "${RUN_OUTPUT_STEM}" ]]; then
    CMD+=(--output-stem "$RUN_OUTPUT_STEM")
fi
if [[ -n "${SEED}" ]]; then
    CMD+=(--seed "$SEED")
fi
if [[ -n "${TRIAL_OFFSET:-}" ]]; then
    CMD+=(--trial-offset "$TRIAL_OFFSET")
fi
if [[ -n "${ICLAMP}" && "${ICLAMP}" != "0" ]]; then
    CMD+=(--iclamp)
fi
if [[ -n "${SNAPSHOT}" && "${SNAPSHOT}" != "0" ]]; then
    CMD+=(--snapshot)
fi
if [[ "${FORCE_SAVE}" == "1" ]]; then
    CMD+=(--force-save)
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# Move this task's logs into the run folder
if [[ -n "${RUN_OUTPUT_STEM}" && -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        log_dir="${RUN_ROOT}/logs"
    else
        log_dir="${RESULTS_DIR}/${RUN_OUTPUT_STEM}/logs"
    fi
    mkdir -p "${log_dir}"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        log_base="pvsst_${ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    else
        log_base="pvsst_${SLURM_JOB_ID}_0"
    fi
    for ext in out err; do
        if [[ -f "${LOG_SRC_DIR}/${log_base}.${ext}" ]]; then
            cp -f "${LOG_SRC_DIR}/${log_base}.${ext}" "${log_dir}/"
        fi
    done
fi

# Auto-merge array outputs into a single multi result (default on)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" == "0" && "${MERGE_ARRAY}" == "1" ]]; then
    MERGED_STEM=${MERGED_STEM:-"results"}
    mkdir -p "${RUN_ROOT}/logs"
    MERGE_CMD=(/home/hrbncv/SCP/scripts/merge_array_results_with_status.sh
        --input-dir "$PARTS_DIR"
        --output-dir "$RUN_ROOT"
        --job-id "$ARRAY_JOB_ID"
        --output-stem "$MERGED_STEM")
    if [[ -n "${MERGE_PATTERN}" ]]; then
        MERGE_CMD+=(--pattern "$MERGE_PATTERN")
    fi
    echo "Submitting merge job: ${MERGE_CMD[*]}"
    ORIG_ARRAY_JOB_ID="${ARRAY_JOB_ID}"
    STATUS_IS_PRIMARY=1
    export ORIG_ARRAY_JOB_ID STATUS_FILE STATUS_LATEST_FILE STATUS_PRIMARY_FILE STATUS_IS_PRIMARY PARTS_DIR RUN_ROOT RESULTS_DIR RUN_OUTPUT_STEM TUNE_DIR RUN_TAG
    sbatch --dependency=afterok:${ARRAY_JOB_ID} \
        --output "${RUN_ROOT}/logs/merge.out" \
        --error "${RUN_ROOT}/logs/merge.err" \
        --wrap "${MERGE_CMD[*]}"
fi
