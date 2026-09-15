"""
llm_bridge.py — Replaced by cli_macro_engine.py (NO AI, NO API)

This module has been replaced with cli_macro_engine.py which uses a
deterministic RegexCommandParser in accordance with project constraints.
"""

from cli_macro_engine import RegexCommandParser, interactive_macro_terminal

__all__ = ["RegexCommandParser", "interactive_macro_terminal"]

if __name__ == "__main__":
    import asyncio
    import sys
    if sys.platform != "win32":
        print("This requires Windows UIAutomation.")
        sys.exit(1)
    asyncio.run(interactive_macro_terminal())
