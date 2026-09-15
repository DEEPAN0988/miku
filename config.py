"""
config.py — Configuration for Desktop Automation Orchestration Layer

Defines strict safety limits, execution thresholds, and toggle settings for
the human confirmation gate (FastConfirm).
"""

from dataclasses import dataclass

@dataclass
class OrchestratorConfig:
    # STRICT SAFETY GATE: Must default to strictly True (ON)
    CONFIRMATION_GATE_ENABLED: bool = True
    
    # Timeout in seconds for user fast-confirm input before auto-halting
    CONFIRMATION_TIMEOUT_SECONDS: float = 30.0
    
    # Maximum allowed execution steps per agent run (prevents runaway loops)
    MAX_EXECUTION_STEPS_PER_RUN: int = 10
    
    # Enable manual edit mode during fast-confirm prompt
    ALLOW_MANUAL_EDIT: bool = True
    
    # OS UIAutomation tree traversal timeout in seconds
    UI_INSPECTION_TIMEOUT: float = 5.0
    
    # STRICT SAFETY CONSTRAINT: No unsupervised retries allowed!
    ENABLE_AUTO_RETRIES: bool = False


# Default active configuration instance
default_config = OrchestratorConfig()
