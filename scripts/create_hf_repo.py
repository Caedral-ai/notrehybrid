#!/usr/bin/env python3
"""Create private Hub repo notre-checkpoints. Requires huggingface-cli login."""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from huggingface_hub import HfApi, HfHubHTTPError, whoami
    except ImportError:
        print("Install huggingface_hub first (see setup_kaggle.sh).", file=sys.stderr)
        return 1

    repo_id = os.environ.get("HF_CHECKPOINT_REPO", "notre-checkpoints")
    try:
        info = whoami()
    except Exception as exc:  # noqa: BLE001 — login missing is the expected failure
        print("Not logged in. Run: huggingface-cli login", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    name = info.get("name") or info.get("fullname") or "?"
    api = HfApi()
    try:
        url = api.create_repo(
            repo_id,
            private=True,
            repo_type="model",
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        print(f"Could not create {repo_id}: {exc}", file=sys.stderr)
        return 1

    print(f"logged in as {name}")
    print(f"private model repo ready: {url}")
    print("Keep it private until Gate C. Do not publish weights from Week 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
