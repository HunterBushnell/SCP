#!/usr/bin/env python3
import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import curses

DEFAULT_STATUS_DIR = os.path.join(os.path.expanduser("~"), "SCP", "logs", "status")
DEFAULT_LATEST_FILE = os.path.join(DEFAULT_STATUS_DIR, "pvsst_latest.status")
DEFAULT_PRIMARY_FILE = os.path.join(DEFAULT_STATUS_DIR, "pvsst_primary.status")

FIELD_ORDER = [
    "state",
    "time",
    "age",
    "run_tag",
    "job_id",
    "array_job_id",
    "array_task_id",
    "tune_dir",
    "output_dir",
    "run_root",
    "output_stem",
    "message",
]

STATE_COLORS = {
    "SUCCESS": "green",
    "RUNNING": "yellow",
    "MERGING": "yellow",
    "MERGE_PENDING": "yellow",
    "ERROR": "red",
    "MISSING": "cyan",
    "UNKNOWN": "cyan",
}


def _progress_line(status):
    try:
        n_trials = int(status.get("n_trials", 0))
    except Exception:
        return None
    if n_trials <= 0:
        return None
    trial_num = status.get("trial_num")
    trial_idx = status.get("trial_idx")
    try:
        if trial_num is not None:
            done = int(trial_num)
        elif trial_idx is not None:
            done = int(trial_idx) + 1
        else:
            done = 0
    except Exception:
        done = 0
    done = max(0, min(done, n_trials))
    try:
        pct = int(status.get("trial_percent", ""))
    except Exception:
        pct = int(round(100.0 * float(done) / float(max(1, n_trials))))
    bar_w = 20
    filled = int(round(bar_w * float(pct) / 100.0))
    filled = max(0, min(filled, bar_w))
    bar = "[" + ("#" * filled) + ("-" * (bar_w - filled)) + "]"
    offset = status.get("trial_offset", "")
    offset_str = ""
    if offset not in ("", None, "0"):
        offset_str = " offset {}".format(offset)
    return "progress: {}/{} ({}%) {}{}".format(done, n_trials, pct, bar, offset_str)


def parse_status(path):
    if not os.path.isfile(path):
        return {"state": "MISSING", "message": "status file not found"}
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    except Exception as exc:
        return {"state": "UNKNOWN", "message": "read error: {}".format(exc)}
    if "state" not in data:
        data["state"] = "UNKNOWN"
    return data


def parse_time(raw):
    if not raw:
        return None
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def format_age(dt):
    if dt is None:
        return None
    try:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
    except Exception:
        return None
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return "{}h{:02d}m".format(hrs, mins)
    if mins > 0:
        return "{}m{:02d}s".format(mins, sec)
    return "{}s".format(sec)


def tail_file(path, n_lines):
    if not path or not os.path.isfile(path):
        return []
    lines = deque(maxlen=n_lines)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
    except Exception:
        return []
    return list(lines)


def pick_log_file(status, status_file):
    candidates = []
    run_root = status.get("run_root")
    output_dir = status.get("output_dir")
    output_stem = status.get("output_stem")
    if run_root:
        candidates.append(os.path.join(run_root, "logs"))
    if output_dir and output_stem:
        candidates.append(os.path.join(output_dir, output_stem, "logs"))
    status_dir = os.path.dirname(status_file)
    if status_dir:
        candidates.append(os.path.dirname(status_dir))

    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        merge_err = os.path.join(d, "merge.err")
        if os.path.isfile(merge_err):
            return merge_err
        err_files = [
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".err")
        ]
        if err_files:
            err_files.sort(key=lambda p: os.path.getmtime(p))
            return err_files[-1]
    return None


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    colors = {
        "red": curses.COLOR_RED,
        "green": curses.COLOR_GREEN,
        "yellow": curses.COLOR_YELLOW,
        "cyan": curses.COLOR_CYAN,
        "white": curses.COLOR_WHITE,
    }
    pairs = {}
    pair_id = 1
    for name, color in colors.items():
        curses.init_pair(pair_id, color, -1)
        pairs[name] = curses.color_pair(pair_id)
        pair_id += 1
    return pairs


def add_line(stdscr, row, text, color=0):
    height, width = stdscr.getmaxyx()
    if row >= height:
        return row
    if text is None:
        text = ""
    if len(text) > width - 1:
        text = text[: max(0, width - 1)]
    try:
        stdscr.addstr(row, 0, text, color)
    except curses.error:
        pass
    return row + 1


def render_panel(stdscr, args):
    if not args.no_color and curses.has_colors():
        color_pairs = init_colors()
    else:
        color_pairs = {}

    stdscr.nodelay(False)
    stdscr.timeout(int(args.interval * 1000))
    curses.curs_set(0)

    last_state = None
    while True:
        status = parse_status(args.status_file)
        time_raw = status.get("time")
        age = format_age(parse_time(time_raw))
        if age:
            status["age"] = age

        state = status.get("state", "UNKNOWN")
        state = state.upper()
        status["state"] = state
        state_color = color_pairs.get(STATE_COLORS.get(state, "white"), 0)

        if args.beep and state in args.beep_on and state != last_state:
            try:
                curses.beep()
            except Exception:
                sys.stdout.write("\a")
                sys.stdout.flush()

        stdscr.erase()
        row = 0
        header = "SCP Status Panel  (q to quit)  {}".format(time.strftime("%Y-%m-%d %H:%M:%S"))
        row = add_line(stdscr, row, header)
        row = add_line(stdscr, row, "state: {}".format(state), state_color)
        progress = _progress_line(status)
        if progress:
            row = add_line(stdscr, row, progress)
        row = add_line(stdscr, row, "")

        displayed = set()
        if progress:
            displayed.update(
                {"trial_idx", "trial_num", "n_trials", "trial_percent", "trial_offset"}
            )
        for key in FIELD_ORDER:
            if key in ("state",):
                continue
            if key in status:
                displayed.add(key)
                row = add_line(stdscr, row, "{}: {}".format(key, status.get(key, "")))

        extras = sorted(k for k in status.keys() if k not in displayed and k not in ("state",))
        for key in extras:
            row = add_line(stdscr, row, "{}: {}".format(key, status.get(key, "")))

        if not args.no_tail:
            row = add_line(stdscr, row, "")
            log_path = pick_log_file(status, args.status_file)
            if log_path:
                row = add_line(
                    stdscr,
                    row,
                    "tail: {} (last {} lines)".format(log_path, args.tail_lines),
                )
                for line in tail_file(log_path, args.tail_lines):
                    row = add_line(stdscr, row, "  " + line)
            else:
                row = add_line(stdscr, row, "tail: no .err log found")

        stdscr.refresh()
        last_state = state

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break


def main():
    parser = argparse.ArgumentParser(description="Live status panel for SCP runs.")
    parser.add_argument(
        "--status-file",
        default=os.environ.get("SCP_STATUS_FILE", ""),
        help="Path to status file (default: primary if present, else latest).",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--tail-lines", type=int, default=12, help="Lines of .err log to show.")
    parser.add_argument("--no-tail", action="store_true", help="Disable log tail.")
    parser.add_argument("--beep", action="store_true", help="Bell on state change.")
    parser.add_argument(
        "--beep-on",
        default="SUCCESS,ERROR",
        help="Comma-separated states that trigger a bell (default: %(default)s).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable color output.")
    args = parser.parse_args()

    if not args.status_file:
        if os.path.isfile(DEFAULT_PRIMARY_FILE):
            args.status_file = DEFAULT_PRIMARY_FILE
        else:
            args.status_file = DEFAULT_LATEST_FILE
    args.beep_on = {s.strip().upper() for s in args.beep_on.split(",") if s.strip()}
    try:
        curses.wrapper(render_panel, args)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
