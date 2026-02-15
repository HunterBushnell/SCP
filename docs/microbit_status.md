# Micro:bit Status Output

This connects a micro:bit (on your local machine) to the SCP status file on the server and shows state changes without chasing log filenames.

## Overview
The flow is:
1) `run_slurm.sh` updates `logs/status/pvsst_latest.status` on the server.
2) A local Python bridge polls that file over SSH.
3) The bridge sends one letter over USB serial to the micro:bit.
4) The micro:bit shows an icon for the current state.

States sent:
- `R` = RUNNING
- `S` = SUCCESS (arrays only after merge completes)
- `E` = ERROR
- `P` = MERGE_PENDING
- `G` = MERGING
- `?` = unknown

## Micro:bit (MakeCode)
Create a new MakeCode project and switch to the JavaScript view, then paste:

```typescript
serial.setBaudRate(BaudRate.BaudRate115200)
let state = "?"

serial.onDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    const s = serial.readUntil(serial.delimiters(Delimiters.NewLine))
    if (s.length > 0) {
        state = s.charAt(s.length - 1)
    }
})

basic.forever(function () {
    if (state == "R") {
        basic.showIcon(IconNames.ArrowNorth)
    } else if (state == "S") {
        basic.showIcon(IconNames.Yes)
    } else if (state == "E") {
        basic.showIcon(IconNames.No)
    } else if (state == "P") {
        basic.showIcon(IconNames.Asleep)
    } else if (state == "G") {
        basic.showIcon(IconNames.Target)
    } else {
        basic.showString("?")
    }
    basic.pause(250)
})
```

Download the .hex from MakeCode and copy it to the micro:bit drive.

## Local bridge (Windows)
The bridge script lives at `scripts/microbit_status_bridge.py` in this repo. It runs on your local machine.

1) Install Python 3 and pyserial:
```bash
pip install pyserial
```

2) Run the bridge:
```bash
cd <repo_root>
python scripts/microbit_status_bridge.py --host user@server \
  --path /path/to/repo/logs/status/pvsst_latest.status \
  --port COM5 --poll 5 --print
```

If you are unsure which port is the micro:bit:
```bash
cd <repo_root>
python scripts/microbit_status_bridge.py --list-ports
```

Notes:
- Use `COM6` if your other micro:bit is on that port.
- Make sure `ssh` works from your local machine to the server (keys recommended).
- If you submit jobs from a different directory, update `--path` to that submit-dir status file.
