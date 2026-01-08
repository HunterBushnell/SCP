#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

STATE_TO_CODE = {
    "RUNNING": "R",
    "SUCCESS": "S",
    "ERROR": "E",
    "MERGE_PENDING": "P",
    "MERGING": "G",
}

DEFAULT_REMOTE_PATH = "/home/hrbncv/SCP/logs/status/pvsst_latest.status"


def run_ssh(ssh_bin, ssh_opts, host, remote_path, timeout):
    remote_cmd = "cat {}".format(shlex.quote(remote_path))
    cmd = [ssh_bin] + shlex.split(ssh_opts) + [host, remote_cmd]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        return None, err or "ssh failed"
    return proc.stdout, None


def parse_state(raw):
    if raw is None:
        return None
    match = re.search(r"^state=(.+)$", raw, re.M)
    if not match:
        return None
    return match.group(1).strip().upper()


def list_ports():
    try:
        import serial.tools.list_ports as list_ports
    except Exception as exc:
        print("pyserial not available: {}".format(exc), file=sys.stderr)
        return 1
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return 0
    for p in ports:
        desc = p.description or ""
        manu = p.manufacturer or ""
        print("{}\t{}\t{}".format(p.device, desc, manu))
    return 0


def auto_port():
    try:
        import serial.tools.list_ports as list_ports
    except Exception:
        return None
    ports = list(list_ports.comports())
    candidates = []
    for p in ports:
        label = " ".join([p.device, p.description or "", p.manufacturer or ""]).lower()
        if "micro:bit" in label or "microbit" in label or "mbed" in label or "daplink" in label:
            candidates.append(p.device)
    if len(candidates) == 1:
        return candidates[0]
    return None


def open_serial(port, baud):
    try:
        import serial
    except Exception as exc:
        raise RuntimeError("pyserial not available: {}".format(exc))
    return serial.Serial(port, baudrate=baud, timeout=1)


def main():
    parser = argparse.ArgumentParser(description="Micro:bit status bridge for SCP.")
    parser.add_argument("--host", default=os.environ.get("SCP_STATUS_HOST", ""))
    parser.add_argument("--path", default=os.environ.get("SCP_STATUS_PATH", DEFAULT_REMOTE_PATH))
    parser.add_argument("--port", default=os.environ.get("MICROBIT_PORT", ""))
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--ssh-bin", default=os.environ.get("SSH_BIN", "ssh"))
    parser.add_argument("--ssh-opts", default=os.environ.get("SSH_OPTS", "-o BatchMode=yes"))
    parser.add_argument("--ssh-timeout", type=float, default=10.0)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--print", dest="print_state", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-keep-last", action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        return list_ports()

    if not args.host:
        print("Missing --host (e.g. user@server).", file=sys.stderr)
        return 2

    port = args.port or auto_port()
    if not port and not args.print_state:
        print("Missing --port and no micro:bit port auto-detected.", file=sys.stderr)
        print("Use --list-ports to see available ports.", file=sys.stderr)
        return 2

    ser = None
    if port:
        try:
            ser = open_serial(port, args.baud)
        except Exception as exc:
            print("Serial open failed: {}".format(exc), file=sys.stderr)
            return 2

    last_code = None
    last_state = None

    while True:
        raw, err = run_ssh(args.ssh_bin, args.ssh_opts, args.host, args.path, args.ssh_timeout)
        state = parse_state(raw)
        if state is None:
            keep_last = not args.no_keep_last
            if not keep_last or last_code is None:
                code = "?"
            else:
                code = last_code
        else:
            code = STATE_TO_CODE.get(state, "?")

        if args.print_state and state != last_state:
            print("{} -> {}".format(state or "UNKNOWN", code))

        if ser and code != last_code:
            try:
                ser.write((code + "\n").encode("ascii", "ignore"))
            except Exception as exc:
                print("Serial write failed: {}".format(exc), file=sys.stderr)
                return 2

        last_code = code
        last_state = state

        if args.once:
            break

        time.sleep(args.poll)

    if ser:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
