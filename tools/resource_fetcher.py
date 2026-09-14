"""
Smart Resource Fetcher & Download State Monitor for Miku OS Assistant.

Integrates:
  1. WinGet CLI package management (search, metadata resolution, non-interactive install).
  2. Browser Download State Monitor tracking .crdownload and .part temporary files.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.screen_inspector import check_autonomous_authorization


# ---------------------------------------------------------------------------
# 1. WinGet CLI Wrapper
# ---------------------------------------------------------------------------

PACKAGE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\.\-]+$")


def sanitize_package_id(package_id: str) -> Tuple[bool, str]:
    """
    Validates package identifier against strict alphanumeric, dot, and dash pattern
    to prevent command injection into subprocess arguments.
    """
    cleaned = package_id.strip()
    if not cleaned:
        return False, "Package ID cannot be empty."
    if len(cleaned) > 128:
        return False, f"Package ID exceeds maximum length of 128 characters: {len(cleaned)}"
    if not PACKAGE_ID_REGEX.match(cleaned):
        return False, f"Invalid characters in Package ID '{cleaned}'. Only [a-zA-Z0-9_.-] allowed."
    return True, cleaned


def parse_winget_search_table(output: str) -> List[Dict[str, str]]:
    """
    Parses fixed-width columnar output from 'winget search'.
    Extracts name, id, version, match, and source.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    # Find the header row by looking for words like Name, Id, Version
    header_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"\bName\b", line, re.IGNORECASE) and re.search(r"\bId\b", line, re.IGNORECASE):
            header_idx = i
            break

    if header_idx == -1 or header_idx + 2 > len(lines):
        return []

    header_line = lines[header_idx]
    # Find column start positions
    matches = list(re.finditer(r"\S+", header_line))
    if not matches:
        return []

    col_spans = []
    for i in range(len(matches)):
        col_name = matches[i].group(0).lower()
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else None
        col_spans.append((col_name, start, end))

    # Next line is usually the dash divider '-------'
    start_row = header_idx + 1
    if set(lines[start_row].strip()) <= {"-", " "}:
        start_row += 1

    records: List[Dict[str, str]] = []
    for line in lines[start_row:]:
        if not line.strip():
            continue
        row: Dict[str, str] = {}
        for col_name, start, end in col_spans:
            val = line[start:end].strip() if start < len(line) else ""
            row[col_name] = val
        if row.get("id"):
            records.append(row)

    return records


def winget_search_package(query: str, timeout: float = 30.0) -> List[Dict[str, str]]:
    """
    Searches for packages matching query via winget CLI.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    cmd = ["winget", "search", cleaned_query, "--accept-source-agreements"]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
        )
        if res.returncode != 0 and not res.stdout:
            print(f"[!] [WINGET SEARCH ERROR] Exit code {res.returncode}: {res.stderr.strip()}", flush=True)
            return []
        return parse_winget_search_table(res.stdout)
    except Exception as exc:
        print(f"[!] [WINGET SEARCH EXCEPTION] {exc}", flush=True)
        return []


def winget_show_package(package_id: str, timeout: float = 30.0) -> Dict[str, str]:
    """
    Queries detailed package metadata using 'winget show'.
    """
    valid, sanitized = sanitize_package_id(package_id)
    if not valid:
        return {"error": sanitized}

    cmd = ["winget", "show", "--id", sanitized, "--exact", "--accept-source-agreements"]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
        )
        if res.returncode != 0:
            return {"error": res.stderr.strip() or f"Package '{sanitized}' not found."}

        details: Dict[str, str] = {"id": sanitized}
        current_key: Optional[str] = None
        for line in res.stdout.splitlines():
            if ":" in line and not line.startswith(" "):
                parts = line.split(":", 1)
                k = parts[0].strip().lower().replace(" ", "_")
                v = parts[1].strip()
                details[k] = v
                current_key = k
            elif current_key and line.startswith("  "):
                details[current_key] += " " + line.strip()
        return details
    except Exception as exc:
        return {"error": str(exc)}


def request_live_human_install_confirmation(package_id: str) -> bool:
    """
    Strict confirmation check before executing a live package installation.
    If MIKU_AUTONOMOUS_MODE=True and MIKU_LIVE_EXECUTION=True, permits execution non-blockingly.
    Otherwise requires interactive console (sys.stdin.isatty()) and typing 'CONFIRM INSTALL'.
    """
    if check_autonomous_authorization("INSTALL"):
        return True

    if not sys.stdin or not sys.stdin.isatty():
        return False

    print("\n" + "=" * 64, flush=True)
    print(" [!] CAUTION: APPLICATION INSTALLATION REQUESTED", flush=True)
    print(f" Target Package ID: {package_id}", flush=True)
    print(" Command: winget install --silent", flush=True)
    print("=" * 64, flush=True)

    try:
        token = input("Type 'CONFIRM INSTALL' to proceed with installation: ").strip()
        return token == "CONFIRM INSTALL"
    except (EOFError, KeyboardInterrupt):
        return False


def winget_install_package(
    package_id: str,
    silent: bool = True,
    force_dry_run: Optional[bool] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Executes non-interactive application installation via winget CLI.

    Flags enforced for non-interactive execution:
      --silent
      --accept-package-agreements
      --accept-source-agreements

    Parameters:
      package_id: Unique package identifier (e.g. 'Git.Git').
      silent: If True, suppresses all UI dialogs (--silent).
      force_dry_run: If True, simulates without executing.
      timeout: Max seconds to await installation completion.
    """
    valid, sanitized_id = sanitize_package_id(package_id)
    if not valid:
        return {
            "success": False,
            "status": "INVALID_PACKAGE_ID",
            "package_id": package_id,
            "message": sanitized_id,
        }

    live_exec = os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes")

    if force_dry_run is True or (force_dry_run is None and not live_exec):
        print(f"[*] [SIMULATED INSTALL] Would install '{sanitized_id}' non-interactively via winget.", flush=True)
        return {
            "success": True,
            "status": "SIMULATED_INSTALL",
            "package_id": sanitized_id,
            "message": f"[SIMULATED INSTALL] Package '{sanitized_id}' validated and ready for installation.",
        }

    # Verify authorization gate
    authorized = request_live_human_install_confirmation(sanitized_id)
    if not authorized:
        print(f"[!] [SAFETY GATE REJECTION] Installation aborted: Not authorized for '{sanitized_id}'.", flush=True)
        return {
            "success": False,
            "status": "ABORT_UNAUTHORIZED",
            "package_id": sanitized_id,
            "message": f"Installation unconfirmed or rejected for '{sanitized_id}'.",
        }

    cmd = [
        "winget",
        "install",
        "--id",
        sanitized_id,
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    if silent:
        cmd.append("--silent")

    print(f"[*] [WINGET INSTALL] Executing: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
        )

        if proc.returncode == 0:
            print(f"[+] [WINGET INSTALL SUCCESS] Successfully installed package '{sanitized_id}'.", flush=True)
            return {
                "success": True,
                "status": "INSTALL_SUCCESS",
                "package_id": sanitized_id,
                "return_code": proc.returncode,
                "output": proc.stdout.strip(),
            }
        else:
            print(f"[!] [WINGET INSTALL FAILED] Exit code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}", flush=True)
            return {
                "success": False,
                "status": "INSTALL_FAILED",
                "package_id": sanitized_id,
                "return_code": proc.returncode,
                "error": proc.stderr.strip() or proc.stdout.strip(),
            }

    except Exception as exc:
        print(f"[!] [WINGET INSTALL EXCEPTION] {exc}", flush=True)
        return {
            "success": False,
            "status": "EXCEPTION",
            "package_id": sanitized_id,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# 2. Browser Download State Monitor
# ---------------------------------------------------------------------------

class BrowserDownloadMonitor:
    """
    Monitors a download directory for active browser download temporary files:
      - .crdownload (Chromium, Google Chrome, Microsoft Edge)
      - .part (Mozilla Firefox)

    Triggers terminal system event notifications and optional callbacks
    when the temporary extension resolves, signaling download completion.
    """

    TEMP_EXTENSIONS = (".crdownload", ".part")

    def __init__(
        self,
        download_dir: Optional[str] = None,
        poll_interval: float = 1.0,
        on_download_complete: Optional[Callable[[str, int], None]] = None,
    ):
        if download_dir:
            self.download_dir = Path(download_dir).resolve()
        else:
            self.download_dir = Path.home() / "Downloads"

        self.poll_interval = max(0.1, poll_interval)
        self.on_download_complete = on_download_complete

        # Internal tracking: {temp_path_str: {"target_path": str, "detected_at": float}}
        self._active_downloads: Dict[str, Dict[str, Any]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @staticmethod
    def resolve_target_filename(temp_path: Path) -> Path:
        """
        Derives target finished file path from the temporary filename.
        e.g. 'installer.exe.crdownload' -> 'installer.exe'
        """
        temp_name = temp_path.name
        for ext in BrowserDownloadMonitor.TEMP_EXTENSIONS:
            if temp_name.endswith(ext):
                target_name = temp_name[:-len(ext)]
                return temp_path.parent / target_name
        return temp_path

    def poll_once(self) -> List[Dict[str, Any]]:
        """
        Performs a single inspection pass over the target directory.
        Returns a list of completed downloads detected in this pass.
        """
        completed: List[Dict[str, Any]] = []

        if not self.download_dir.exists() or not self.download_dir.is_dir():
            return completed

        with self._lock:
            # 1. Scan directory for current temp files
            current_temp_files = set()
            try:
                for entry in self.download_dir.iterdir():
                    if entry.is_file() and any(entry.name.endswith(ext) for ext in self.TEMP_EXTENSIONS):
                        entry_str = str(entry.resolve())
                        current_temp_files.add(entry_str)
                        if entry_str not in self._active_downloads:
                            target = self.resolve_target_filename(entry)
                            self._active_downloads[entry_str] = {
                                "target_path": str(target),
                                "detected_at": time.time(),
                            }
                            print(f"[*] [DOWNLOAD DETECTED] Active download in progress: '{entry.name}'", flush=True)
            except Exception as scan_err:
                print(f"[!] [DOWNLOAD MONITOR ERROR] Failed directory scan: {scan_err}", flush=True)

            # 2. Check previously tracked active downloads
            untracked = []
            for temp_path_str, info in self._active_downloads.items():
                if temp_path_str not in current_temp_files:
                    # Temp file disappeared! Check if target completed file exists
                    target_path = Path(info["target_path"])
                    if target_path.exists() and target_path.is_file():
                        try:
                            file_size = target_path.stat().st_size
                        except Exception:
                            file_size = 0

                        event_data = {
                            "filename": target_path.name,
                            "path": str(target_path),
                            "size_bytes": file_size,
                            "timestamp": time.time(),
                        }
                        completed.append(event_data)
                        print(
                            f"[*] [DOWNLOAD COMPLETE] Successfully downloaded: {target_path.name} "
                            f"({file_size} bytes) -> {target_path}",
                            flush=True,
                        )

                        if self.on_download_complete:
                            try:
                                self.on_download_complete(str(target_path), file_size)
                            except Exception as cb_err:
                                print(f"[!] [DOWNLOAD CALLBACK ERROR] {cb_err}", flush=True)
                    untracked.append(temp_path_str)

            for removed in untracked:
                del self._active_downloads[removed]

        return completed

    def _run_loop(self):
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.poll_interval)

    def start(self):
        """Starts background monitoring in a daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MikuDownloadMonitorThread")
        self._thread.start()
        print(f"[*] [DOWNLOAD MONITOR STARTED] Polling '{self.download_dir}' every {self.poll_interval}s.", flush=True)

    def stop(self):
        """Stops the background monitoring thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        print("[*] [DOWNLOAD MONITOR STOPPED]", flush=True)

    async def run_async(self, stop_event: Optional[asyncio.Event] = None):
        """
        Runs the monitoring loop within an asyncio coroutine.
        """
        print(f"[*] [ASYNC DOWNLOAD MONITOR STARTED] Polling '{self.download_dir}'.", flush=True)
        try:
            while stop_event is None or not stop_event.is_set():
                self.poll_once()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass
        finally:
            print("[*] [ASYNC DOWNLOAD MONITOR STOPPED]", flush=True)
