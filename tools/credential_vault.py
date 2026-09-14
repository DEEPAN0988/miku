"""
tools/credential_vault.py — Local Windows Credential Vault Integration (Phase 3)

SAFETY & PRIVACY SPECIFICATION:
  1. NATIVE WINDOWS CREDENTIAL MANAGER:
     Uses win32cred to read and write credentials securely into Windows Credential Manager.
  2. SECURE MEMORY WRAPPER (ZERO LOG LEAKAGE):
     Secrets are wrapped in VaultSecret objects whose __repr__ and __str__ strictly display
     '***REDACTED***'. Plaintext is never emitted to console logs, exception tracebacks,
     or chat contexts, and is only accessed via .get_secret() inside active tool memory.
  3. API SPECIFICATION:
     - Storage via win32cred.CredWrite(Credential, 0)
     - Retrieval via win32cred.CredRead(TargetName, Type, Flags)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
import win32cred


class VaultSecret:
    """
    Secure in-memory secret container.
    Protects sensitive credentials from being leaked into string conversions,
    chat responses, dictionary formatting, or logging pipelines.
    """

    def __init__(self, target_name: str, username: str, secret: str):
        self._target_name = target_name
        self._username = username
        self._secret = secret

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def username(self) -> str:
        return self._username

    def get_secret(self) -> str:
        """Returns the unredacted credential value strictly for in-memory tool execution."""
        return self._secret

    def to_safe_dict(self) -> Dict[str, str]:
        """Returns a non-sensitive dictionary safe for logging and agent status reporting."""
        return {
            "target_name": self._target_name,
            "username": self._username,
            "secret": "***REDACTED***",
        }

    def __repr__(self) -> str:
        return f"<VaultSecret target='{self._target_name}' user='{self._username}' secret='***REDACTED***'>"

    def __str__(self) -> str:
        return f"<VaultSecret target='{self._target_name}' user='{self._username}' secret='***REDACTED***'>"


def store_credential(
    target_name: str,
    username: str,
    password: str,
    cred_type: int = win32cred.CRED_TYPE_GENERIC,
    persist: int = win32cred.CRED_PERSIST_LOCAL_MACHINE,
) -> Dict[str, Any]:
    """
    Stores credentials in the Windows Credential Manager using win32cred.CredWrite.
    Passes Flags=0 (CRED_PRESERVE_CREDENTIAL_BLOB).
    """
    if not target_name or not username:
        return {
            "status": "ERROR",
            "executed": False,
            "error": "TargetName and UserName cannot be empty.",
        }

    credential_dict = {
        "Type": cred_type,
        "TargetName": target_name,
        "UserName": username,
        "CredentialBlob": password,
        "Persist": persist,
    }

    try:
        # Explicitly pass Flags=0 (CRED_PRESERVE_CREDENTIAL_BLOB)
        win32cred.CredWrite(credential_dict, 0)
        return {
            "status": "SUCCESS",
            "executed": True,
            "target_name": target_name,
            "username": username,
            "output": f"[VAULT] Successfully secured credentials for target '{target_name}' in Windows Credential Manager.",
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "executed": False,
            "target_name": target_name,
            "error": str(e),
            "output": f"[VAULT ERROR] Failed to store credential for target '{target_name}': {e}",
        }


def retrieve_credential(
    target_name: str,
    cred_type: int = win32cred.CRED_TYPE_GENERIC,
    flags: int = 0,
) -> Optional[VaultSecret]:
    """
    Retrieves credentials from Windows Credential Manager using win32cred.CredRead.
    Returns a VaultSecret object that obscures plaintext in logs.
    """
    try:
        cred = win32cred.CredRead(target_name, cred_type, flags)
        username = cred.get("UserName", "")
        raw_blob = cred.get("CredentialBlob", b"")

        if isinstance(raw_blob, bytes):
            try:
                secret_str = raw_blob.decode("utf-16-le")
            except Exception:
                secret_str = raw_blob.decode("utf-8", errors="replace")
        else:
            secret_str = str(raw_blob)

        return VaultSecret(target_name=target_name, username=username, secret=secret_str)
    except Exception:
        return None


def delete_credential(
    target_name: str,
    cred_type: int = win32cred.CRED_TYPE_GENERIC,
    flags: int = 0,
) -> bool:
    """
    Deletes credentials from Windows Credential Manager.
    """
    try:
        win32cred.CredDelete(target_name, cred_type, flags)
        return True
    except Exception:
        return False


class CredentialVault:
    """
    Secure client interface for desktop agent tools requiring credentials.
    """

    @staticmethod
    def get(target_name: str) -> Optional[VaultSecret]:
        return retrieve_credential(target_name)

    @staticmethod
    def put(target_name: str, username: str, password: str, persist: int = win32cred.CRED_PERSIST_LOCAL_MACHINE) -> Dict[str, Any]:
        return store_credential(target_name, username, password, persist=persist)

    @staticmethod
    def remove(target_name: str) -> bool:
        return delete_credential(target_name)
