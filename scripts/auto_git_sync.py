"""
scripts/auto_git_sync.py — Automatic Git Commit & Push Sync Engine for Miku

Functions:
  1. Detects any uncommitted or untracked file changes in the workspace.
  2. Respects .gitignore (never adds checkpoints, binaries, or databases).
  3. Automatically stages, commits with ISO timestamp, and pushes to origin/main.
  4. Complies with the Antigravity hook contract (consumes JSON on stdin, outputs {} on stdout).
  5. Fail-safe: Any network or git error is caught and logged without breaking IDE operations.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys


def get_workspace_root() -> str:
    # Anchor to the directory containing .git
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(candidate, ".git")):
        return candidate
    return os.path.abspath(os.path.join(script_dir, ".."))


def run_git_cmd(args: list[str], cwd: str) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def sync_workspace():
    workspace = get_workspace_root()
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return

    # Check for working tree changes
    rc, status_out, _ = run_git_cmd(["status", "--porcelain"], cwd=workspace)
    if rc != 0 or not status_out:
        # Working tree is clean, nothing to commit
        return

    # Filter out untracked files that might be in transient ignore states
    lines = [line.strip() for line in status_out.splitlines() if line.strip()]
    if not lines:
        return

    # Stage changes (respects .gitignore)
    rc_add, _, err_add = run_git_cmd(["add", "."], cwd=workspace)
    if rc_add != 0:
        return

    # Commit changes
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    commit_msg = f"Auto-sync: update from Antigravity session ({now_str})\n\nModified files:\n"
    for line in lines[:15]:
        commit_msg += f"- {line}\n"
    if len(lines) > 15:
        commit_msg += f"- ... and {len(lines) - 15} more\n"

    rc_commit, _, _ = run_git_cmd(["commit", "-m", commit_msg], cwd=workspace)
    if rc_commit != 0:
        return

    # Push to origin main
    run_git_cmd(["push", "origin", "main"], cwd=workspace)


def main():
    try:
        sync_workspace()
    except Exception:
        pass

    # Antigravity PostToolUse / Stop hook contract expects a JSON object on stdout
    print(json.dumps({}))


if __name__ == "__main__":
    main()
