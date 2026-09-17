"""
Bluetooth Device Discovery & Connection Module for Windows.
Zero external cloud dependencies.
"""

import subprocess
import json
from typing import List, Dict, Any


class BluetoothManager:
    @staticmethod
    def scan_devices() -> List[Dict[str, Any]]:
        """Query Bluetooth devices via PowerShell Get-PnpDevice."""
        ps_cmd = """
        Get-PnpDevice -Class Bluetooth -Status OK | 
        Select-Object FriendlyName, InstanceId, Status, Present | 
        ConvertTo-Json -Compress
        """
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            if not res.stdout.strip():
                return []
            
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            
            devices = []
            for item in data:
                devices.append({
                    "name": item.get("FriendlyName", "Unknown Device"),
                    "id": item.get("InstanceId", ""),
                    "status": item.get("Status", "Unknown"),
                    "present": item.get("Present", False)
                })
            return devices
        except Exception:
            return []
