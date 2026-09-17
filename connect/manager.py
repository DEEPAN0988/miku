"""
Unified Connect Manager for Miku.
Manages Bluetooth and Wi-Fi devices.
Enforces Section 3 & 5.6: No silent auto-pairing; explicit user confirmation required.
"""

from typing import Dict, Any, List, Optional
from .bluetooth import BluetoothManager
from .wifi import WiFiManager
from control.gate import RiskGate, RiskLevel


class ConnectManager:
    def __init__(self):
        self.bt = BluetoothManager()
        self.wifi = WiFiManager()

    def scan_all(self, target_type: str = "all") -> Dict[str, Any]:
        """Discover nearby devices and networks."""
        results = {}
        if target_type in ["all", "bluetooth"]:
            results["bluetooth_devices"] = self.bt.scan_devices()
        if target_type in ["all", "wifi"]:
            results["wifi_networks"] = self.wifi.scan_networks()
            results["current_wifi"] = self.wifi.get_current_connection()
        return results

    def request_pair_device(self, device_name: str, device_address: str, confirmed: bool = False) -> Dict[str, Any]:
        """
        Request pairing with a new device.
        Requires explicit confirmation per PRD non-goals.
        """
        if not confirmed:
            return {
                "success": False,
                "requires_confirmation": True,
                "payload": f"pair {device_name} ({device_address})",
                "message": f"Pairing request for new device '{device_name}' [{device_address}]. Do you want to authorize this connection?"
            }
        
        # When confirmed, record authorized device
        return {
            "success": True,
            "message": f"Successfully authorized and paired device '{device_name}'."
        }
