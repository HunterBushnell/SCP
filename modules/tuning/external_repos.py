"""Safe, action-triggered provisioning for optional external repositories.

SCP treats ACT and BMTool as read-only external dependencies.  This module
locates an existing checkout or installs a fresh checkout only when a caller
explicitly invokes a feature that needs it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ExternalRepoSpec:
    """Description of one optional source checkout."""

    display_name: str
    directory_name: str
    package_name: str
    marker_rel: Path
    repo_url: str
    path_env_vars: tuple[str, ...]
    target_dir_env: str
    repo_url_env: str
    repo_branch_env: str
    auto_clone_env: str
    canonical_path_env: str
    token_env_vars: tuple[str, ...] = ()


def env_flag(name: str, *, default: bool) -> bool:
    """Read a conventional boolean environment variable."""

    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSE_VALUES


def external_repo_candidates(
    spec: ExternalRepoSpec,
    *,
    repo_root: Path,
    extra_candidates: Sequence[str | Path] = (),
) -> list[Path]:
    """Return de-duplicated checkout candidates in resolution order."""

    candidates = [Path(value).expanduser() for value in extra_candidates]
    for variable in spec.path_env_vars:
        raw = os.environ.get(variable)
        if raw:
            candidates.append(Path(raw).expanduser())

    root = Path(repo_root).expanduser().resolve()
    cwd = Path.cwd()
    candidates.extend(
        [
            root.parent / "mods" / spec.directory_name,
            root / "mods" / spec.directory_name,
            Path.home() / "mods" / spec.directory_name,
            (cwd / ".." / "mods" / spec.directory_name).resolve(),
            (cwd / "mods" / spec.directory_name).resolve(),
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def looks_like_external_repo(path: Path, spec: ExternalRepoSpec) -> bool:
    """Return whether ``path`` contains the expected import marker."""

    return (Path(path) / spec.marker_rel).is_file()


def resolve_external_repo_checkout(
    spec: ExternalRepoSpec,
    *,
    repo_root: Path,
    extra_candidates: Sequence[str | Path] = (),
) -> Path:
    """Resolve an existing checkout without importing or installing anything."""

    for candidate in external_repo_candidates(
        spec,
        repo_root=repo_root,
        extra_candidates=extra_candidates,
    ):
        if looks_like_external_repo(candidate, spec):
            return candidate

    expected = ", ".join(spec.path_env_vars)
    raise FileNotFoundError(
        f"{spec.display_name} repo not found. Set one of {expected} or place "
        f"a checkout at ../mods/{spec.directory_name} relative to SCP."
    )


def ensure_external_repo_checkout(
    spec: ExternalRepoSpec,
    *,
    repo_root: Path,
    auto_clone: Optional[bool] = None,
    prepend: bool = False,
    target_dir: Optional[str | Path] = None,
    repo_url: Optional[str] = None,
) -> Path:
    """Resolve or safely clone an optional repository and add it to ``sys.path``.

    Calling this function represents an explicit request to use the external
    tool, so missing checkouts are installed by default.  Set the tool-specific
    ``SCP_AUTO_CLONE_*`` flag to ``0`` or pass ``auto_clone=False`` to opt out.
    """

    root = Path(repo_root).expanduser().resolve()
    if auto_clone is None:
        auto_clone = env_flag(spec.auto_clone_env, default=True)

    try:
        checkout = resolve_external_repo_checkout(spec, repo_root=root)
    except FileNotFoundError as missing_error:
        if not auto_clone:
            raise FileNotFoundError(
                f"{missing_error} Automatic installation is disabled by "
                f"{spec.auto_clone_env}=0."
            ) from missing_error
        checkout = _clone_external_repo(
            spec,
            repo_root=root,
            target_dir=target_dir,
            repo_url=repo_url,
        )

    checkout = checkout.resolve()
    _assert_import_location_is_compatible(spec, checkout)
    os.environ[spec.canonical_path_env] = str(checkout)
    path_string = str(checkout)
    if path_string not in sys.path:
        if prepend:
            sys.path.insert(0, path_string)
        else:
            sys.path.append(path_string)
    return checkout


def _clone_external_repo(
    spec: ExternalRepoSpec,
    *,
    repo_root: Path,
    target_dir: Optional[str | Path],
    repo_url: Optional[str],
) -> Path:
    configured_target = target_dir or os.environ.get(spec.target_dir_env)
    destination = Path(
        configured_target
        or (Path(repo_root).resolve().parent / "mods" / spec.directory_name)
    ).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and looks_like_external_repo(destination, spec):
        return destination
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileNotFoundError(
            f"{spec.target_dir_env} exists but is not a valid "
            f"{spec.display_name} checkout: {destination}. Move it aside or set "
            f"{spec.target_dir_env} to an empty installation location."
        )

    resolved_url = (
        repo_url
        or os.environ.get(spec.repo_url_env)
        or spec.repo_url
    )
    branch = os.environ.get(spec.repo_branch_env, "").strip()
    clone_url, secrets = _authenticated_clone_url(
        str(resolved_url),
        spec.token_env_vars,
    )
    temp_path = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.directory_name}-clone-",
            dir=str(destination.parent),
        )
    )
    command = ["git", "clone", "--depth", "1"]
    if branch:
        command.extend(["--branch", branch])
    command.extend([clone_url, str(temp_path)])

    print(
        f"{spec.display_name} was not found; installing a fresh copy at "
        f"{destination} ..."
    )
    try:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while cloning {spec.display_name} from "
                f"{resolved_url}. Check network access, clone it manually, or "
                f"configure one of {', '.join(spec.path_env_vars)}."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not run Git to install {spec.display_name}: {exc}. "
                "Install Git, clone the repository manually, or configure "
                f"one of {', '.join(spec.path_env_vars)}."
            ) from exc
        if completed.returncode:
            detail = _sanitize_clone_error(
                completed.stderr or completed.stdout,
                secrets=secrets,
                clone_url=clone_url,
                public_url=str(resolved_url),
            )
            raise RuntimeError(
                f"Could not clone {spec.display_name} from {resolved_url}. "
                f"{detail or 'Git exited without an error message.'} "
                f"Clone it manually or set one of {', '.join(spec.path_env_vars)}."
            )
        if not looks_like_external_repo(temp_path, spec):
            raise FileNotFoundError(
                f"The downloaded {spec.display_name} checkout did not contain "
                f"{spec.marker_rel}: {temp_path}"
            )

        if destination.exists():
            if looks_like_external_repo(destination, spec):
                return destination
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"{destination} became non-empty while "
                    f"{spec.display_name} was being installed."
                )
            destination.rmdir()
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    print(f"{spec.display_name} ready: {destination}")
    return destination


def _authenticated_clone_url(
    repo_url: str,
    token_env_vars: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    token = next(
        (
            os.environ.get(variable)
            for variable in token_env_vars
            if os.environ.get(variable)
        ),
        None,
    )
    if token and repo_url.startswith("https://") and "@" not in repo_url:
        return repo_url.replace("https://", f"https://{token}@", 1), (token,)
    return repo_url, ()


def _sanitize_clone_error(
    message: str,
    *,
    secrets: Sequence[str],
    clone_url: str,
    public_url: str,
) -> str:
    sanitized = str(message).strip().replace(clone_url, public_url)
    for secret in secrets:
        sanitized = sanitized.replace(secret, "***")
    return " ".join(sanitized.split())


def _assert_import_location_is_compatible(
    spec: ExternalRepoSpec,
    checkout: Path,
) -> None:
    module = sys.modules.get(spec.package_name)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        return
    imported_path = Path(module_file).resolve()
    try:
        imported_path.relative_to(checkout)
    except ValueError:
        raise RuntimeError(
            f"{spec.display_name} is already imported from {imported_path}, but "
            f"SCP resolved {checkout}. Restart the kernel before switching "
            f"{spec.display_name} checkouts."
        ) from None
