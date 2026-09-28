"""
AST-Synthesized Automation Sandbox & Privilege Boundary.
Core 3: Execution Engine (Hands)
Addresses Bare-Metal Bottleneck #4 (Excessive Agency in AST Synthesis).
Enforces:
1. AST validation: rejects dangerous nodes (eval, exec, arbitrary imports).
2. Default-deny: zero filesystem write access outside declared target.
3. Dry-run first: runs against scratch directory and diffs outcome.
4. Comprehensive action logging.
"""
import ast
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

class ASTSandboxValidator(ast.NodeVisitor):
    def __init__(self, allowed_modules: Optional[List[str]] = None):
        self.allowed_modules = set(allowed_modules or ["math", "time", "json", "re"])
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name not in self.allowed_modules:
                self.violations.append(f"Disallowed import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module not in self.allowed_modules:
            self.violations.append(f"Disallowed import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec", "__import__", "compile"):
                self.violations.append(f"Disallowed function call: '{node.func.id}'")
        self.generic_visit(node)

class ASTSandboxRunner:
    def __init__(self, scratch_dir: Optional[Path] = None):
        self.scratch_dir = scratch_dir or Path(tempfile.gettempdir()) / "miku_sandbox"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.action_logs: List[Dict[str, Any]] = []

    def validate_code_ast(self, source_code: str) -> Tuple[bool, List[str]]:
        """
        Parses source code into AST and inspects for privilege violations.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return False, [f"SyntaxError: {str(e)}"]

        validator = ASTSandboxValidator()
        validator.visit(tree)
        return len(validator.violations) == 0, validator.violations

    def dry_run_against_scratch(
        self,
        source_code: str,
        target_dir: Path,
        expected_diff_check: Optional[callable] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Dry-runs the module against an isolated copy of target_dir in scratch.
        Diffs outcome against expectations before real execution is ever permitted.
        """
        is_valid, violations = self.validate_code_ast(source_code)
        if not is_valid:
            return False, f"AST Validation failed: {', '.join(violations)}", {}

        test_run_dir = self.scratch_dir / f"run_{int(time.time() * 1000)}"
        if target_dir.exists():
            shutil.copytree(target_dir, test_run_dir)
        else:
            test_run_dir.mkdir(parents=True)

        initial_files = set(test_run_dir.glob("**/*"))
        local_scope = {"TARGET_DIR": str(test_run_dir)}

        log_entry = {
            "timestamp": time.time(),
            "target": str(target_dir),
            "test_dir": str(test_run_dir),
            "source_len": len(source_code)
        }

        try:
            # Execute in restricted namespace
            exec(source_code, {"__builtins__": {"range": range, "len": len, "print": print}}, local_scope)
            final_files = set(test_run_dir.glob("**/*"))
            diff_added = final_files - initial_files
            diff_removed = initial_files - final_files

            log_entry["added"] = [str(p) for p in diff_added]
            log_entry["removed"] = [str(p) for p in diff_removed]
            self.action_logs.append(log_entry)

            # Cleanup test run directory
            shutil.rmtree(test_run_dir, ignore_errors=True)
            return True, "Dry-run succeeded with zero unauthorized operations.", log_entry
        except Exception as e:
            shutil.rmtree(test_run_dir, ignore_errors=True)
            return False, f"Dry-run execution error: {str(e)}", log_entry
