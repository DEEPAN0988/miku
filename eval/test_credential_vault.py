"""
eval/test_credential_vault.py — Unit Tests for Windows Credential Vault Integration
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import win32cred

sys.path.insert(0, os.path.abspath("."))

from tools.credential_vault import (
    REAL_CREDENTIAL_WRITE_ENABLED,
    CredentialVault,
    VaultSecret,
    delete_credential,
    retrieve_credential,
    store_credential,
)


class TestCredentialVault(unittest.TestCase):

    def setUp(self):
        self.test_target = "Miku_Unit_Test_Target_Secure"
        self.test_user = "test_agent_user"
        self.test_password = "P@ssw0rd_Very_Secret_987!"
        self.mock_vault = {}

    def _mock_cred_write(self, cred_dict, flags):
        self.mock_vault[cred_dict["TargetName"]] = {
            "TargetName": cred_dict["TargetName"],
            "UserName": cred_dict["UserName"],
            "CredentialBlob": cred_dict["CredentialBlob"],
            "Type": cred_dict.get("Type", win32cred.CRED_TYPE_GENERIC),
        }
        return None

    def _mock_cred_read(self, target_name, cred_type, flags):
        if target_name in self.mock_vault:
            return self.mock_vault[target_name]
        raise ValueError("Credential not found")

    def _mock_cred_delete(self, target_name, cred_type, flags):
        if target_name in self.mock_vault:
            del self.mock_vault[target_name]
            return True
        raise ValueError("Credential not found")

    def test_circuit_breaker_invariant_fails_closed(self):
        """Constitutional Safety Invariant: REAL_CREDENTIAL_WRITE_ENABLED must be False by default."""
        self.assertFalse(
            REAL_CREDENTIAL_WRITE_ENABLED,
            "CRITICAL SAFETY VIOLATION: REAL_CREDENTIAL_WRITE_ENABLED must be False by default."
        )
        res = store_credential(self.test_target, self.test_user, self.test_password)
        self.assertEqual(res["status"], "CIRCUIT_BREAKER_BLOCKED")
        self.assertFalse(res["executed"])

        deleted = delete_credential(self.test_target)
        self.assertFalse(deleted)

    def test_store_and_retrieve_credential_mocked(self):
        with patch("tools.credential_vault.REAL_CREDENTIAL_WRITE_ENABLED", True):
            with patch("win32cred.CredWrite", side_effect=self._mock_cred_write):
                with patch("win32cred.CredRead", side_effect=self._mock_cred_read):
                    # Store
                    res = store_credential(
                        target_name=self.test_target,
                        username=self.test_user,
                        password=self.test_password,
                        persist=win32cred.CRED_PERSIST_SESSION,
                    )
                    self.assertEqual(res["status"], "SUCCESS")
                    self.assertTrue(res["executed"])

                    # Retrieve
                    secret = retrieve_credential(self.test_target)
                    self.assertIsNotNone(secret)
                    self.assertIsInstance(secret, VaultSecret)
                    self.assertEqual(secret.target_name, self.test_target)
                    self.assertEqual(secret.username, self.test_user)
                    self.assertEqual(secret.get_secret(), self.test_password)

    def test_vault_secret_obscuration_in_logs_and_strings(self):
        """Verify password is never exposed in str, repr, or safe dict."""
        secret = VaultSecret(self.test_target, self.test_user, self.test_password)

        self.assertNotIn(self.test_password, repr(secret))
        self.assertNotIn(self.test_password, str(secret))
        self.assertIn("***REDACTED***", repr(secret))
        self.assertIn("***REDACTED***", str(secret))

        safe_dict = secret.to_safe_dict()
        self.assertEqual(safe_dict["secret"], "***REDACTED***")
        self.assertNotIn(self.test_password, str(safe_dict))

        # But memory access succeeds
        self.assertEqual(secret.get_secret(), self.test_password)

    def test_delete_credential_mocked(self):
        with patch("tools.credential_vault.REAL_CREDENTIAL_WRITE_ENABLED", True):
            with patch("win32cred.CredWrite", side_effect=self._mock_cred_write):
                with patch("win32cred.CredRead", side_effect=self._mock_cred_read):
                    with patch("win32cred.CredDelete", side_effect=self._mock_cred_delete):
                        store_credential(
                            target_name=self.test_target,
                            username=self.test_user,
                            password=self.test_password,
                            persist=win32cred.CRED_PERSIST_SESSION,
                        )
                        self.assertIsNotNone(retrieve_credential(self.test_target))

                        deleted = delete_credential(self.test_target)
                        self.assertTrue(deleted)

                        self.assertIsNone(retrieve_credential(self.test_target))

    def test_credential_vault_class_wrapper_mocked(self):
        with patch("tools.credential_vault.REAL_CREDENTIAL_WRITE_ENABLED", True):
            with patch("win32cred.CredWrite", side_effect=self._mock_cred_write):
                with patch("win32cred.CredRead", side_effect=self._mock_cred_read):
                    with patch("win32cred.CredDelete", side_effect=self._mock_cred_delete):
                        CredentialVault.put(self.test_target, self.test_user, self.test_password, persist=win32cred.CRED_PERSIST_SESSION)
                        vault_sec = CredentialVault.get(self.test_target)
                        self.assertIsNotNone(vault_sec)
                        self.assertEqual(vault_sec.get_secret(), self.test_password)

                        CredentialVault.remove(self.test_target)
                        self.assertIsNone(CredentialVault.get(self.test_target))


if __name__ == "__main__":
    unittest.main()
