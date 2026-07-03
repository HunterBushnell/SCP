from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _status_paths() -> tuple[Optional[str], Optional[str], Optional[str], bool, str]:
    status_file = os.environ.get("STATUS_FILE") or os.environ.get("SCP_STATUS_FILE")
    latest_file = os.environ.get("STATUS_LATEST_FILE") or os.environ.get("SCP_STATUS_LATEST_FILE")
    primary_file = os.environ.get("STATUS_PRIMARY_FILE") or os.environ.get(
        "SCP_STATUS_PRIMARY_FILE"
    )
    is_primary = os.environ.get("STATUS_IS_PRIMARY", "0")
    is_primary = str(is_primary).strip() in ("1", "true", "yes", "on")
    run_tag = os.environ.get("RUN_TAG", "")
    return status_file, latest_file, primary_file, is_primary, run_tag


def _read_status_file(path: str) -> tuple[Dict[str, str], list[str]]:
    data: Dict[str, str] = {}
    order: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in data:
                    order.append(key)
                data[key] = value.strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return data, order


def _write_status_file(path: str, data: Dict[str, str], order: list[str]) -> None:
    if not path:
        return
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(Path(path).with_suffix(".tmp"))
    with open(tmp_path, "w", encoding="utf-8") as f:
        for key in order:
            if key in data:
                f.write(f"{key}={data[key]}\n")
        for key in sorted(k for k in data.keys() if k not in order):
            f.write(f"{key}={data[key]}\n")
    os.replace(tmp_path, path)


def _allows_update_file(path: str, run_tag: str, new_state: str) -> bool:
    if not path or not os.path.isfile(path):
        return True
    latest_data, _ = _read_status_file(path)
    if latest_data.get("state") == "ERROR" and latest_data.get("run_tag") == run_tag:
        return new_state == "ERROR"
    return True


def _update_status_progress(
    status_file: Optional[str],
    latest_file: Optional[str],
    primary_file: Optional[str],
    is_primary: bool,
    run_tag: str,
    *,
    trial_idx: int,
    n_trials: int,
    trial_offset: int,
) -> None:
    if not status_file:
        return
    data, order = _read_status_file(status_file)
    if run_tag and "run_tag" not in data:
        data["run_tag"] = run_tag
        order.append("run_tag")
    n_trials = max(1, int(n_trials))
    done = max(0, min(int(trial_idx) + 1, n_trials))
    pct = int(round(100.0 * float(done) / float(n_trials)))
    now = datetime.now().isoformat()
    data.update(
        {
            "state": "RUNNING",
            "time": now,
            "trial_idx": str(int(trial_idx)),
            "trial_num": str(int(done)),
            "n_trials": str(int(n_trials)),
            "trial_offset": str(int(trial_offset)),
            "trial_percent": str(int(pct)),
            "message": f"trial {done}/{n_trials}",
        }
    )
    if "state" not in order:
        order.append("state")
    if "time" not in order:
        order.append("time")
    _write_status_file(status_file, data, order)

    if is_primary and primary_file and primary_file != status_file:
        if _allows_update_file(primary_file, run_tag, "RUNNING"):
            primary_data, primary_order = _read_status_file(primary_file)
            if run_tag and "run_tag" not in primary_data:
                primary_data["run_tag"] = run_tag
                primary_order.append("run_tag")
            primary_data.update(data)
            if "state" not in primary_order:
                primary_order.append("state")
            if "time" not in primary_order:
                primary_order.append("time")
            _write_status_file(primary_file, primary_data, primary_order)

    if is_primary and latest_file and latest_file != status_file:
        if _allows_update_file(latest_file, run_tag, "RUNNING"):
            latest_data, latest_order = _read_status_file(latest_file)
            if run_tag and "run_tag" not in latest_data:
                latest_data["run_tag"] = run_tag
                latest_order.append("run_tag")
            latest_data.update(data)
            if "state" not in latest_order:
                latest_order.append("state")
            if "time" not in latest_order:
                latest_order.append("time")
            _write_status_file(latest_file, latest_data, latest_order)


def build_trial_callback(session: Any):
    if session.iclamp_enabled or session.sim_cfg is None:
        return None

    status_file, latest_file, primary_file, is_primary, run_tag = _status_paths()
    n_trials = int(session.sim_cfg.get("n_trials", 1) or 1)
    trial_offset = int(session.sim_cfg.get("trial_offset", 0) or 0)

    def trial_callback(info: Dict[str, Any]) -> None:
        idx = info.get("trial_idx", 0)
        try:
            idx = int(idx)
        except Exception:
            idx = 0
        _update_status_progress(
            status_file,
            latest_file,
            primary_file,
            is_primary,
            run_tag,
            trial_idx=idx,
            n_trials=n_trials,
            trial_offset=trial_offset,
        )

    return trial_callback
