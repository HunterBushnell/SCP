"""Fresh-process worker for the compact pipeline's input/synapse preview."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence


def build_input_preview(
    tune_dir: str | Path,
    *,
    seed: Optional[int] = None,
    trial_idx: int = 0,
) -> dict[str, Any]:
    """Prepare current configs and sample preview-only synapse records."""

    from modules.analysis import analysis
    from modules.simulation import SimulationOptions, SimulationSession

    options = SimulationOptions(
        n_trials=1,
        seed=None if seed is None else int(seed),
        sim_overrides={"iclamp": {"enabled": False}},
        iclamp=False,
        force_save=False,
    )
    session = SimulationSession.from_tune(tune_dir, options=options).prepare()
    syn_state = session.preview_synapses(trial_idx=int(trial_idx))
    records = syn_state.get("records", {}) or {}
    summary = analysis.summarize_synapse_records(records, geom=session.geom)
    preview_state = {
        "records": records,
        "preview_only": True,
        "summary": summary,
    }

    print("Input/synapse preview prepared in a fresh process.")
    for key, value in session.summary().items():
        print(f"  {key}: {value}")
    if records:
        print("  synapse groups:", ", ".join(records))
    else:
        print("  synapse groups: none")

    return {
        "syn_state": preview_state,
        "summary": summary,
        "session_summary": session.summary(),
        "trial_idx": int(trial_idx),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trial-idx", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.trial_idx < 0:
        raise ValueError("trial_idx must be non-negative.")
    payload = build_input_preview(
        args.tune_dir,
        seed=args.seed,
        trial_idx=args.trial_idx,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle)
    print("Preview data serialized for notebook display.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
