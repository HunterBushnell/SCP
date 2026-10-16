"""Explicit, hash-checked aliases for relocated repository data files.

Only missing paths declared in cells/DATA_PATH_ALIASES.json are redirected.
Existing files and arbitrary external paths always retain their normal meaning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def resolve_data_alias(path, repo_root):
    root = Path(repo_root).resolve()
    source = Path(path).expanduser()
    source = source if source.is_absolute() else root / source
    if source.exists():
        return source.resolve()
    try:
        relative = source.resolve().relative_to(root).as_posix()
    except ValueError:
        return source
    manifest = root / 'cells/DATA_PATH_ALIASES.json'
    if not manifest.is_file():
        return source
    aliases = json.loads(manifest.read_text(encoding='utf-8')).get('aliases', {})
    entry = aliases.get(relative)
    if entry is None:
        return source
    raw = Path(entry['path'])
    target = (root / raw).resolve()
    if raw.is_absolute() or '..' in raw.parts or not target.is_relative_to(root):
        raise ValueError(f'Unsafe data relocation for {relative!r}: {entry["path"]!r}')
    if not target.is_file():
        raise FileNotFoundError(f'Relocated data {relative!r} is missing at {target}. Restore the declared data copy.')
    expected = entry.get('sha256')
    if not expected or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Relocated data hash mismatch for {relative!r} at {target}. '
                         'Do not replay historical inputs using a changed dataset; select the intended data explicitly.')
    return target
