"""OpenRouter credential selection for Forge Bench.

Credentials are never included in status output or experiment metadata. The
dedicated Forge Bench key file takes precedence so an unrelated shell or
Hermes credential cannot silently replace the benchmark key.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CredentialSelection:
    value: str | None
    source: str | None


def benchmark_key_path() -> Path:
    return Path.home() / ".config" / "forge-bench" / "openrouter.env"


def _key_from_export_file(path: Path) -> str | None:
    """Read the single literal export line written by store_openrouter_key."""
    if not path.is_file():
        return None

    found: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)", stripped)
        if not match:
            raise ValueError(
                f"{path} has an unsupported entry on line {line_number}; "
                "expected a literal OPENROUTER_API_KEY assignment"
            )
        # shlex decodes the quoted form emitted by the helper but does not
        # evaluate shell substitutions or source arbitrary shell code.
        parts = shlex.split(match.group(1), comments=False, posix=True)
        if len(parts) != 1 or not parts[0] or re.search(r"\$\(|`", parts[0]):
            raise ValueError(
                f"{path} has an invalid OPENROUTER_API_KEY value on line {line_number}"
            )
        found.append(parts[0])

    if len(found) > 1:
        raise ValueError(f"{path} contains more than one OPENROUTER_API_KEY assignment")
    return found[0] if found else None


def select_openrouter_credential() -> CredentialSelection:
    """Select the intended key without exposing it.

    The dedicated Forge Bench key is authoritative when present. Otherwise,
    preserve the existing shell-environment behavior. If neither is available,
    Hermes can still load credentials from the copied Hermes home files.
    """
    key_path = benchmark_key_path()
    key = _key_from_export_file(key_path)
    if key:
        return CredentialSelection(key, "forge_bench_key_file")

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return CredentialSelection(key, "shell_environment")

    return CredentialSelection(None, None)


def available_hermes_credential_files(home: Path) -> list[str]:
    """Return names (never contents) of legacy Hermes credential files."""
    return [name for name in (".env", "auth.json") if (home / name).is_file()]
