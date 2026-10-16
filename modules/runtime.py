"""Code identity; never use this directory as an implicit project workspace."""
from importlib import metadata
from pathlib import Path
import re


def code_root():
    return Path(__file__).resolve().parent.parent


def software_provenance():
    try:
        version = metadata.version("scp-pipeline")
    except metadata.PackageNotFoundError:
        version = "0.2.0"
    result = {"name": "scp-pipeline", "version": version}
    root = code_root()
    try:
        git = root / '.git'
        if git.is_file():
            git = (root / git.read_text().strip().removeprefix('gitdir: ')).resolve()
        head = (git / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head[5:]
            target = (git / ref).resolve()
            if git.resolve() not in target.parents:
                return result
            if target.is_file():
                head = target.read_text().strip()
            else:
                head = next((line.split()[0] for line in (git / 'packed-refs').read_text().splitlines()
                             if line.endswith(' ' + ref)), '')
        if re.fullmatch(r'[0-9a-f]{40,64}', head):
            result['revision'] = head
    except (OSError, ValueError):
        pass  # Optional checkout provenance must not prevent saving a run.
    return result
