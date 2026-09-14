"""
tools/resource_fetcher.py — Smart Resource Fetcher & Entity Classifier (Phase 2)

Features:
  1. Heuristic & NLP Resource Entity Classifier:
     Classifies user download requests into APPLICATION, DATASET, MEDIA, or DOCUMENT.
  2. Routing Dispatcher:
     - APPLICATION: Queries Windows Package Manager (winget) first via subprocess.
     - DATASET / DOCUMENT / MEDIA: Formats targeted browser search queries.
  3. Download Completion Monitor:
     Monitors OS Downloads folder using os.path.getmtime and handles in-progress
     download temporary files (.crdownload, .part) until completion.
"""

from __future__ import annotations

from enum import Enum
import os
import re
import subprocess
import time
import urllib.parse
import webbrowser
from typing import Any, Callable, Dict, List, Optional, Tuple


# ==============================================================================
# PERMANENT SAFETY CIRCUIT BREAKER (DEFENSE IN DEPTH)
# ==============================================================================
# REAL RESOURCE FETCHING / PACKAGE INSTALL / BROWSER LAUNCH IS HARD-CODED TO FALSE.
# Automated installation or external downloading is strictly prohibited by default.
REAL_RESOURCE_FETCH_ENABLED: bool = False


class ResourceType(str, Enum):
    APPLICATION = "APPLICATION"
    DATASET = "DATASET"
    MEDIA = "MEDIA"
    DOCUMENT = "DOCUMENT"
    UNKNOWN = "UNKNOWN"


# Heuristic keyword and pattern maps
APP_PATTERNS = [
    r"\b(?:install|download|get|setup)\s+(?:app|application|software|tool|program|editor|ide|package)\b",
    r"\b(?:app|application|software|ide|editor|tool|program|installer|driver)\b",
    r"\.(?:exe|msi|dmg|deb|rpm|appx|msix)\b",
    r"\b(?:vscode|chrome|firefox|brave|edge|git|python|node|docker|zoom|slack|discord|spotify|steam|blender|vlc|notepad\+\+|7zip|obs|postman|wireshark|powertoys)\b",
]

DATASET_PATTERNS = [
    r"\b(?:dataset|data\s*set|corpus|tabular\s*data|benchmarks?|training\s*data|database\s*dump|records)\b",
    r"\.(?:csv|tsv|jsonl|parquet|arrow|feather|sqlite|db|arff)\b",
    r"\b(?:huggingface|kaggle|openimages|imagenet|common\s*crawl|coco\s*dataset)\b",
]

MEDIA_PATTERNS = [
    r"\b(?:video|audio|movie|song|soundtrack|album|track|podcast|clip|stream|photo|picture|wallpaper)\b",
    r"\.(?:mp4|mkv|avi|mov|mp3|flac|wav|aac|ogg|png|jpg|jpeg|gif|webp|svg)\b",
    r"\b(?:youtube|spotify|soundcloud|vimeo|netflix|tiktok)\b",
]

DOCUMENT_PATTERNS = [
    r"\b(?:document|pdf|paper|article|thesis|dissertation|report|whitepaper|guidelines?|manual|specification|cheatsheet|book|textbook|slides|presentation)\b",
    r"\.(?:pdf|docx?|xlsx?|pptx?|epub|mobi|txt|md|rtf)\b",
    r"\b(?:arxiv|biorxiv|pubmed|ieee|researchgate)\b",
]


def classify_resource_request(query: str) -> Dict[str, Any]:
    """
    Classifies a user query into a resource category (APPLICATION, DATASET, MEDIA, DOCUMENT)
    and extracts the cleaned target entity name.
    """
    q = query.strip()
    q_low = q.lower()

    scores = {
        ResourceType.APPLICATION: 0,
        ResourceType.DATASET: 0,
        ResourceType.MEDIA: 0,
        ResourceType.DOCUMENT: 0,
    }

    # Match patterns
    for pat in APP_PATTERNS:
        if re.search(pat, q_low):
            scores[ResourceType.APPLICATION] += 2

    for pat in DATASET_PATTERNS:
        if re.search(pat, q_low):
            scores[ResourceType.DATASET] += 2

    for pat in MEDIA_PATTERNS:
        if re.search(pat, q_low):
            scores[ResourceType.MEDIA] += 2

    for pat in DOCUMENT_PATTERNS:
        if re.search(pat, q_low):
            scores[ResourceType.DOCUMENT] += 2

    # Intent verb cues
    if re.search(r"\b(?:install|setup)\b", q_low):
        scores[ResourceType.APPLICATION] += 3
    elif re.search(r"\b(?:read|cite)\b", q_low):
        scores[ResourceType.DOCUMENT] += 2
    elif re.search(r"\b(?:play|listen|watch)\b", q_low):
        scores[ResourceType.MEDIA] += 2

    # Determine highest scoring category
    best_category = ResourceType.UNKNOWN
    best_score = 0
    for cat, score in scores.items():
        if score > best_score:
            best_score = score
            best_category = cat

    confidence = min(1.0, best_score / 4.0) if best_score > 0 else 0.0

    # Extract target entity: strip command verbs
    cleaned = q
    lead_verbs = r"^(?:please\s+)?(?:download|install|fetch|get|search\s+for|find|locate|pull|grab)\s+"
    cleaned = re.sub(lead_verbs, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip category keywords from entity
    cat_terms = r"^(?:the\s+)?(?:dataset|app|application|software|paper|document|video|song|media|tool|package|installer)\s+(?:called|named|for)?\s*"
    cleaned = re.sub(cat_terms, "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        cleaned = q

    return {
        "category": best_category,
        "confidence": confidence,
        "target_entity": cleaned,
        "scores": {k.value: v for k, v in scores.items()},
        "original_query": query,
    }


def search_winget(package_name: str, runner: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Queries the Windows Package Manager (winget) CLI for matching applications.
    """
    cmd = ["winget", "search", package_name]
    try:
        if runner:
            return runner(cmd)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=True)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        packages = []
        lines = stdout.strip().splitlines()
        # Parse table lines if output contains standard separator
        header_idx = -1
        for idx, line in enumerate(lines):
            if "---" in line:
                header_idx = idx
                break

        if header_idx != -1 and header_idx + 1 < len(lines):
            for row in lines[header_idx + 1:]:
                row_stripped = row.strip()
                if row_stripped:
                    parts = re.split(r"\s{2,}", row_stripped)
                    if len(parts) >= 2:
                        packages.append({
                            "name": parts[0],
                            "id": parts[1],
                            "version": parts[2] if len(parts) > 2 else "",
                            "source": parts[3] if len(parts) > 3 else "",
                        })

        found = len(packages) > 0
        return {
            "status": "SUCCESS" if found else "NO_MATCH",
            "found": found,
            "package_name": package_name,
            "packages": packages,
            "raw_output": stdout,
            "error": stderr if proc.returncode != 0 else None,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "found": False,
            "package_name": package_name,
            "packages": [],
            "error": str(e),
        }


def format_search_url_for_resource(category: ResourceType, entity: str) -> Tuple[str, str]:
    """
    Builds a targeted search URL and display query for browser search.
    """
    if category == ResourceType.DATASET:
        query_str = f"{entity} dataset download"
    elif category == ResourceType.DOCUMENT:
        query_str = f"{entity} filetype:pdf download"
    elif category == ResourceType.MEDIA:
        query_str = f"{entity} download media"
    else:
        query_str = f"{entity} download"

    url = f"https://www.google.com/search?q={urllib.parse.quote(query_str)}"
    return url, query_str


def dispatch_resource_fetch(
    query: str,
    dry_run: bool = True,
    winget_runner: Optional[Callable] = None,
    browser_opener: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Smart Resource Fetcher Dispatcher:
    - APPLICATION -> Queries winget first. If winget succeeds, provides installation candidate.
                     If winget finds nothing, falls back to browser search.
    - DATASET / DOCUMENT / MEDIA -> Formats targeted search query for default browser.
    """
    clf = classify_resource_request(query)
    category = clf["category"]
    entity = clf["target_entity"]

    if not dry_run and not REAL_RESOURCE_FETCH_ENABLED:
        return {
            "status": "CIRCUIT_BREAKER_BLOCKED",
            "executed": False,
            "dry_run": True,
            "category": category.value if hasattr(category, "value") else str(category),
            "entity": entity,
            "error": "REAL_RESOURCE_FETCH_DISABLED: Real resource fetching/installation is permanently disabled (REAL_RESOURCE_FETCH_ENABLED=False).",
            "output": f"[CIRCUIT BREAKER BLOCKED] Real resource downloading or installation is permanently disabled (REAL_RESOURCE_FETCH_ENABLED=False). Target: '{entity}'.",
        }

    if category == ResourceType.APPLICATION:
        if dry_run:
            return {
                "status": "SIMULATED_APP_FETCH",
                "executed": False,
                "dry_run": True,
                "category": category.value,
                "entity": entity,
                "action": "winget_search",
                "command": f"winget search {entity}",
                "output": (
                    f"[SIMULATION: APP FETCH] Target '{entity}' classified as APPLICATION. "
                    f"Would query Windows Package Manager: `winget search {entity}`."
                ),
            }

        # Real execution: query winget
        wg_result = search_winget(entity, runner=winget_runner)
        if wg_result.get("found"):
            top_pkg = wg_result["packages"][0]
            return {
                "status": "WINGET_MATCH_FOUND",
                "executed": True,
                "dry_run": False,
                "category": category.value,
                "entity": entity,
                "packages": wg_result["packages"],
                "recommended_id": top_pkg["id"],
                "recommended_command": f"winget install --id {top_pkg['id']} -e",
                "output": (
                    f"[WINGET FOUND] Found package '{top_pkg['name']}' (Id: {top_pkg['id']}, Version: {top_pkg['version']}). "
                    f"Recommended install command: `winget install --id {top_pkg['id']} -e`."
                ),
            }
        else:
            # Fallback to browser search if not available in winget
            url, search_q = format_search_url_for_resource(category, entity)
            if browser_opener:
                browser_opener(url)
            else:
                webbrowser.open(url)

            return {
                "status": "WINGET_NOT_FOUND_BROWSER_FALLBACK",
                "executed": True,
                "dry_run": False,
                "category": category.value,
                "entity": entity,
                "url": url,
                "search_query": search_q,
                "output": (
                    f"[RESOURCE ROUTER] Package '{entity}' not indexed in winget. "
                    f"Opened browser search query: \"{search_q}\"."
                ),
            }

    else:
        # DATASET, DOCUMENT, MEDIA, or UNKNOWN
        url, search_q = format_search_url_for_resource(category, entity)
        if dry_run:
            return {
                "status": "SIMULATED_BROWSER_SEARCH",
                "executed": False,
                "dry_run": True,
                "category": category.value,
                "entity": entity,
                "action": "browser_search",
                "url": url,
                "search_query": search_q,
                "output": (
                    f"[SIMULATION: RESOURCE SEARCH] Target '{entity}' classified as {category.value}. "
                    f"Would open default browser to: \"{search_q}\" ({url})."
                ),
            }

        if browser_opener:
            browser_opener(url)
        else:
            webbrowser.open(url)

        return {
            "status": "BROWSER_SEARCH_OPENED",
            "executed": True,
            "dry_run": False,
            "category": category.value,
            "entity": entity,
            "url": url,
            "search_query": search_q,
            "output": f"[RESOURCE ROUTER] Launched browser search for {category.value}: \"{search_q}\".",
        }


def monitor_download_completion(
    downloads_dir: Optional[str] = None,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Monitors the OS Downloads folder for completed downloads.
    Tracks .crdownload (Chrome/Edge) and .part (Firefox) temporary files
    and verifies completion when the temporary extension is stripped and file size stabilizes.
    """
    folder = downloads_dir or os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "Downloads")
    if not os.path.exists(folder):
        return {
            "status": "DOWNLOADS_DIR_NOT_FOUND",
            "executed": False,
            "folder": folder,
            "error": f"Downloads directory does not exist: {folder}",
        }

    t0 = start_time if start_time is not None else time.time()
    deadline = t0 + timeout

    tracked_partials: Dict[str, float] = {}  # partial_path -> initial_mtime

    while time.time() < deadline:
        try:
            entries = os.listdir(folder)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

        current_partials = []
        completed_candidates = []

        for entry in entries:
            full_path = os.path.join(folder, entry)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue

            # Only examine items touched after or shortly before start_time
            if mtime < (t0 - 2.0):
                continue

            low = entry.lower()
            if low.endswith(".crdownload") or low.endswith(".part"):
                current_partials.append(full_path)
                tracked_partials[full_path] = mtime
            else:
                completed_candidates.append((full_path, mtime))

        # Check if a previously tracked partial file completed (its base name now exists without .crdownload/.part)
        for partial, _ in list(tracked_partials.items()):
            if partial not in current_partials:
                # The partial file disappeared! Let's check for target base file
                base_name = re.sub(r"\.(?:crdownload|part)$", "", partial, flags=re.IGNORECASE)
                if os.path.exists(base_name):
                    try:
                        size = os.path.getsize(base_name)
                        return {
                            "status": "DOWNLOAD_COMPLETED",
                            "file_path": base_name,
                            "file_name": os.path.basename(base_name),
                            "size_bytes": size,
                            "duration_sec": round(time.time() - t0, 2),
                            "from_partial": True,
                        }
                    except OSError:
                        pass

        # If no partials are currently running, check if a newly created complete file appeared
        if not current_partials and completed_candidates:
            # Sort by mtime descending
            completed_candidates.sort(key=lambda x: x[1], reverse=True)
            newest_file, newest_mtime = completed_candidates[0]
            if newest_mtime >= (t0 - 1.0):
                try:
                    s1 = os.path.getsize(newest_file)
                    time.sleep(poll_interval)
                    s2 = os.path.getsize(newest_file)
                    # If size is stable and non-zero
                    if s1 == s2 and s1 > 0:
                        return {
                            "status": "DOWNLOAD_COMPLETED",
                            "file_path": newest_file,
                            "file_name": os.path.basename(newest_file),
                            "size_bytes": s2,
                            "duration_sec": round(time.time() - t0, 2),
                            "from_partial": False,
                        }
                except OSError:
                    pass

        time.sleep(poll_interval)

    # If timed out but partials are still downloading
    if current_partials:
        return {
            "status": "DOWNLOAD_IN_PROGRESS",
            "active_partials": [os.path.basename(p) for p in current_partials],
            "duration_sec": round(time.time() - t0, 2),
            "timed_out": True,
        }

    return {
        "status": "DOWNLOAD_TIMEOUT",
        "duration_sec": round(time.time() - t0, 2),
        "timed_out": True,
    }
