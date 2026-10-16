"""Compatibility entry point; supports existing flags including --project."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Bootstrap only old checkout scripts; dependency users import scp_pipeline.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from scp_pipeline.cli import setup as _implementation
except ModuleNotFoundError as exc:
    if exc.name != 'scp_pipeline':
        raise
    import modules  # checkout-only canonical namespace bootstrap
    from scp_pipeline.cli import setup as _implementation

_implementation.REPO_ROOT = ROOT
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
