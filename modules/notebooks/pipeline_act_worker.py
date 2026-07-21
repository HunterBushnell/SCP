"""Fresh-process worker used by the compact ACT notebook interface."""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from pathlib import Path
from typing import Any, Mapping


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(request: Mapping[str, Any]) -> dict[str, Any]:
    from modules.tuning import (
        act_config_fingerprint,
        act_output_status,
        collect_act_predictions,
        ensure_act_on_syspath,
        evaluate_act_predictions,
        load_act_active_config,
        load_act_module_metrics,
        run_act_active_modules,
    )

    action = str(request["action"])
    config_path = Path(request["config_path"]).expanduser().resolve()
    cfg = load_act_active_config(config_path)
    print(f"SCP ACT worker: {action}", flush=True)
    print(f"ACT workspace: {cfg['workspace']}", flush=True)
    if action == "probe":
        ensure_act_on_syspath(repo_root=Path(cfg["repo_root"]))
        from modules.tuning.act_active import _import_act_api, _import_workspace_builder

        importlib.import_module("sklearn")
        importlib.import_module("timeout_decorator")
        _import_act_api(cfg)
        builder = _import_workspace_builder(cfg)
        cell = builder()
        soma = getattr(cell, "soma", None)
        if soma is None:
            raise AttributeError("The registered ACT loader did not expose a soma section.")
        sections = list(soma) if not callable(soma) else [soma]
        if not sections:
            raise AttributeError("The registered ACT loader exposed an empty soma section list.")
        sections[0](0.5)
        print("ACT availability and fresh-process cell construction: ready", flush=True)
        return {
            "available": True,
            "message": "ACT is available and the registered loader builds in a fresh process.",
        }
    if action == "run":
        requested = request.get("modules", "all")
        print(f"Running ACT module selection: {requested}", flush=True)
        modules = run_act_active_modules(
            config_path,
            modules=requested,
            n_cpus=request.get("n_cpus"),
            overwrite=bool(request.get("overwrite", False)),
        )
        print("ACT optimization complete; loading proposals and metrics", flush=True)
        return {
            "modules": modules,
            "predictions": collect_act_predictions(config_path),
            "metrics": load_act_module_metrics(config_path),
            "output_status": act_output_status(config_path),
            "config_fingerprint": act_config_fingerprint(config_path),
        }
    if action == "evaluate":
        print("Evaluating merged ACT predictions with a temporary FI simulation", flush=True)
        result = evaluate_act_predictions(
            config_path,
            predictions=request.get("predictions"),
            n_cpus=request.get("n_cpus"),
            overwrite=bool(request.get("overwrite", False)),
        )
        result["config_path"] = str(config_path)
        print("ACT prediction evaluation complete", flush=True)
        return result
    raise ValueError(f"Unknown ACT worker action: {action!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        payload = _run(request)
    except BaseException:
        traceback.print_exc()
        raise
    _write_result(Path(args.result), payload)


if __name__ == "__main__":
    main()
