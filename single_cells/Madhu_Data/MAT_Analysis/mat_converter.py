#!/usr/bin/env python3
"""
mat_ingest.py — Inspect and convert MATLAB .mat files (classic ≤v7.2 and v7.3/HDF5)
to convenient Python formats: .npz (arrays), .pkl (full Python structure),
and optionally .csv (tidy when shapes line up, or timeseries with a chosen time key).

Usage examples
--------------
# Auto convert one file to npz+pkl (+csv when feasible)
python mat_ingest.py path/to/file.mat --formats npz pkl csv

# Batch convert all .mat under a folder
python mat_ingest.py path/to/folder --formats npz pkl

# Just list inventory (no export)
python mat_ingest.py path/to/file.mat --list

# Export only selected variables
python mat_ingest.py file.mat --vars Time Voltage --formats npz csv

# Force timeseries CSV using a chosen time column
python mat_ingest.py file.mat --formats csv --timeseries --time-key Time

# Guided (interactive) mode
python mat_ingest.py path/to/file.mat --guided
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import scipy.io as sio
import h5py
import pandas as pd


# ---------- detectors & inventories ----------
def is_v73(path: Union[str, Path]) -> bool:
    p = Path(path)
    try:
        return h5py.is_hdf5(p)
    except Exception:
        return False


def classic_inventory(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    return sio.whosmat(str(path))


def v73_inventory(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append((name, obj.shape, str(obj.dtype)))
        f.visititems(walk)
    return out


# ---------- classic (≤v7.2) loaders ----------
def _is_mat_struct(x) -> bool:
    return getattr(x, "__class__", type(x)).__name__ == "mat_struct"


def _mat_to_py(obj: Any) -> Any:
    if _is_mat_struct(obj):
        return {k: _mat_to_py(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_mat_to_py(x) for x in obj.ravel()]
    return obj  # numeric arrays, strings, etc.


def load_classic(path: Path) -> Dict[str, Any]:
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {k: _mat_to_py(v) for k, v in mat.items() if not k.startswith("__")}


# ---------- v7.3 (HDF5) loaders ----------
def _load_v73_dataset(f: h5py.File, name: str) -> Any:
    d = f[name]
    if d.dtype.kind in ("S", "O", "U"):
        try:
            return d.asstr()[...]
        except Exception:
            return d[...]
    return d[...]


def load_v73(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for name in f:
            obj = f[name]
            if isinstance(obj, h5py.Dataset):
                out[name] = _load_v73_dataset(f, name)
            else:
                # flatten one level of datasets inside this group
                group: Dict[str, Any] = {}
                def collect(n, o):
                    if isinstance(o, h5py.Dataset):
                        key = n.split("/")[-1]
                        group[key] = _load_v73_dataset(f, n)
                obj.visititems(collect)
                out[name] = group
    return out


# ---------- dataframe helpers ----------
def try_make_dataframe(py: Dict[str, Any]) -> Union[pd.DataFrame, None]:
    """Heuristic: gather 1-D arrays with equal length into one DataFrame."""
    cols = {k: np.asarray(v).ravel() for k, v in py.items()
            if isinstance(v, np.ndarray) and v.ndim == 1}
    if not cols:
        return None
    lengths = {k: len(v) for k, v in cols.items()}
    L = max(lengths.values())
    keys = [k for k, ln in lengths.items() if ln == L]
    if not keys:
        return None
    df = pd.DataFrame({k: cols[k] for k in keys})
    # put time-like column first if present
    for cand in ("t", "time", "Time", "c001_Time", "T"):
        if cand in df.columns:
            df = df[[cand] + [c for c in df.columns if c != cand]]
            break
    return df


def to_timeseries_df(py: Dict[str, Any], time_key: str) -> Union[pd.DataFrame, None]:
    """Build a tidy timeseries DataFrame using a chosen time column."""
    if time_key not in py:
        return None
    t = np.asarray(py[time_key]).ravel()
    cols: Dict[str, np.ndarray] = {}
    for k, v in py.items():
        if k == time_key:
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.size == t.size:
            cols[k] = arr
        elif arr.ndim == 2 and arr.shape[0] == t.size:
            for j in range(arr.shape[1]):
                cols[f"{k}_{j}"] = arr[:, j]
    if not cols:
        return None
    df = pd.DataFrame({"t": t, **cols})
    return df


# ---------- saving ----------
def save_outputs(py: Dict[str, Any], out_dir: Path, stem: str,
                 formats: Iterable[str] = ("npz", "pkl", "csv"),
                 timeseries: bool = False, time_key: str | None = None) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if "npz" in formats:
        arrays = {k: v for k, v in py.items() if isinstance(v, np.ndarray)}
        if arrays:
            p = out_dir / f"{stem}.npz"
            np.savez_compressed(p, **arrays)
            paths["npz"] = p

    if "pkl" in formats:
        import pickle
        p = out_dir / f"{stem}.pkl"
        with open(p, "wb") as f:
            pickle.dump(py, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths["pkl"] = p

    if "csv" in formats:
        df = None
        if timeseries and time_key:
            df = to_timeseries_df(py, time_key)
        if df is None:
            df = try_make_dataframe(py)
        if df is not None:
            p = out_dir / f"{stem}.csv"
            df.to_csv(p, index=False)
            paths["csv"] = p

    return paths


# ---------- processing ----------
def process_one(path: Path, out_root: Path, formats: Iterable[str],
                select_vars: List[str] | None = None,
                timeseries: bool = False, time_key: str | None = None,
                list_only: bool = False) -> Dict[str, Path]:
    v73 = is_v73(path)
    inv = v73_inventory(path) if v73 else classic_inventory(path)
    kind = "v7.3 (HDF5)" if v73 else "classic (≤v7.2)"
    print(f"\n[{path.name}] Detected {kind}. {len(inv)} items found.")
    for name, shape, dtype in inv[:10]:
        print(f"  - {name:30s} shape={shape} dtype={dtype}")
    if len(inv) > 10:
        print("  ...")

    if list_only:
        return {}

    py = load_v73(path) if v73 else load_classic(path)
    if select_vars:
        missing = [k for k in select_vars if k not in py]
        if missing:
            print(f"  ! Skipping missing vars: {missing}")
        py = {k: py[k] for k in select_vars if k in py}

    out_dir = out_root / path.stem
    paths = save_outputs(py, out_dir, path.stem, formats, timeseries, time_key)
    if paths:
        print("  Wrote:", {k: str(v) for k, v in paths.items()})
    else:
        print("  Nothing written (no matching data/format).")
    return paths


def iter_mat_paths(inputs: List[Path]) -> Iterable[Path]:
    for p in inputs:
        p = p.expanduser()
        if p.is_dir():
            yield from sorted(p.glob("*.mat"))
        elif p.is_file() and p.suffix.lower() == ".mat":
            yield p


# ---------- guided mode ----------
def guided(path: Path, out_root: Path):
    v73 = is_v73(path)
    inv = v73_inventory(path) if v73 else classic_inventory(path)
    kind = "v7.3 (HDF5)" if v73 else "classic (≤v7.2)"
    print(f"\n[{path}] Detected {kind}. Items:")
    for name, shape, dtype in inv:
        print(f"  - {name:30s} shape={shape} dtype={dtype}")

    py = load_v73(path) if v73 else load_classic(path)
    keys = list(py.keys())
    print("\nVariables:", keys)

    vars_in = input("\nEnter variables to export (comma-separated) or press Enter for ALL: ").strip()
    select_vars = [s.strip() for s in vars_in.split(",") if s.strip()] if vars_in else None

    fmts_in = input("Formats (choose any of: npz,pkl,csv) [default: npz,pkl]: ").strip() or "npz,pkl"
    fmts = [s.strip() for s in fmts_in.split(",") if s.strip()]

    ts = False
    tkey = None
    if "csv" in fmts:
        yn = input("Try timeseries CSV with a chosen time key? [y/N]: ").strip().lower()
        if yn == "y":
            tkey = input("  Time key (e.g., Time, t, c001_Time): ").strip()
            ts = True

    print("\nConverting...")
    if select_vars:
        py = {k: py[k] for k in select_vars if k in py}
    out_dir = out_root / path.stem
    paths = save_outputs(py, out_dir, path.stem, fmts, ts, tkey)
    print("Done. Wrote:", {k: str(v) for k, v in paths.items()})


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Inspect/convert MATLAB .mat files to NPZ/PKL/CSV.")
    ap.add_argument("inputs", nargs="+", help=".mat files or folders containing .mat")
    ap.add_argument("--out-root", default="processed", help="Output root directory (default: processed)")
    ap.add_argument("--formats", nargs="+", choices=("npz", "pkl", "csv"),
                    default=["npz", "pkl"], help="Formats to write")
    ap.add_argument("--vars", nargs="+", help="Subset of variable names to export")
    ap.add_argument("--timeseries", action="store_true",
                    help="When writing CSV, force timeseries layout using --time-key")
    ap.add_argument("--time-key", help="Time variable name for --timeseries CSV")
    ap.add_argument("--list", action="store_true", help="Only list inventory; no export")
    ap.add_argument("--guided", action="store_true", help="Interactive mode for a single file")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    if args.guided:
        # guided mode expects exactly one input file
        first = next(iter_mat_paths([Path(args.inputs[0])]), None)
        if first is None:
            print("No .mat file found for guided mode.")
            return
        guided(first, out_root)
        return

    # auto / batch mode
    files = list(iter_mat_paths([Path(p) for p in args.inputs]))
    if not files:
        print("No .mat files found.")
        return

    for p in files:
        process_one(
            p,
            out_root,
            formats=args.formats,
            select_vars=args.vars,
            timeseries=args.timeseries,
            time_key=args.time_key,
            list_only=args.list,
        )


if __name__ == "__main__":
    main()
