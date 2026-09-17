"""
Wi-Fi Management Module for Windows via netsh.
"""

import subprocess
import re
from typing import List, Dict, Any


class WiFiManager:
    @staticmethod
    def get_current_connection() -> Dict[str, Any]:
        """Check active Wi-Fi connection and SSID."""
        try:
            res = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True,
                text=True,
                timeout=5
            )
            ssid_match = re.search(r"^\s*SSID\s*:\s*(.+)$", res.stdout, re.MULTILINE)
            state_match = re.search(r"^\s*State\s*:\s*(.+)$", res.stdout, re.MULTILINE)
            
            connected = state_match and "connected" in state_match.group(1).lower()
            ssid = ssid_match.group(1).strip() if ssid_match else "Disconnected"
            
            return {
                "connected": bool(connected),
                "ssid": ssid,
                "raw": res.stdout
            }
        except Exception as e:
            return {"connected": False, "ssid": "Unknown", "error": str(e)}

    @staticmethod
    def scan_networks() -> List[str]:
        """Scan for available Wi-Fi networks in range."""
        try:
            res = subprocess.run(
                ["netsh", "wlan", "show", "networks"],
                capture_output=True,
                text=True,
                timeout=10
            )
            ssids = []
            for match in re.finditer(r"^\s*SSID\s+\d+\s*:\s*(.+)$", res.stdout, re.MULTILINE):
                name = match.group(1).strip()
                if name and name not in ssids:
                    ssids.append(name)
            return ssids
        except Exception:
            return []
