#!/usr/bin/env python3
"""Store the OpenRouter key Forge Bench loads for the current user."""

from __future__ import annotations

import getpass
import shlex
from pathlib import Path


def main() -> int:
    config_dir = Path.home() / ".config" / "forge-bench"
    key_file = config_dir / "openrouter.env"

    key = getpass.getpass("OpenRouter API key: ").strip()
    if not key:
        print("No key entered; nothing saved.")
        return 1

    confirmation = getpass.getpass("Confirm OpenRouter API key: ").strip()
    if key != confirmation:
        print("The keys did not match; nothing saved.")
        return 1

    config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_dir.chmod(0o700)
    key_file.write_text(
        f"export OPENROUTER_API_KEY={shlex.quote(key)}\n",
        encoding="utf-8",
    )
    key_file.chmod(0o600)
    print(f"Saved key to {key_file} with owner-only permissions.")
    print("Forge Bench will load this key automatically; verify with:")
    print("  uv run forge-bench --check-credentials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
