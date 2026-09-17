"""
MIKU SOVEREIGN MASTER TERMINAL CLI (Unified Interface)
Integrates Daemon, Dual-System Entropy Routing, Formal Logic Prover,
Causal Do-Calculus, Neural Architecture Search, and Category Theory Functors.
"""

import sys
import asyncio
import torch
import torch.nn.functional as F

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import Sovereign Subsystems
try:
    from miku_daemon import SovereignMikuDaemon
    from miku_kernel import upgrade_daemon_cognitive_routing
    from miku_prover import verify_and_prove
    from miku_causal import OS_CausalGraph, patch_causal_world_model
    from miku_category import (
        build_filesystem_category,
        build_webdom_category,
        StructurePreservingFunctor,
        fit_functor,
        transfer_heuristic,
    )
    from miku_system_life import SystemLifeOSBridge
except ImportError as e:
    print(f"[Terminal Boot Error] Failed to import subsystem: {e}")
    sys.exit(1)


async def main_terminal():
    # ANSI Color Codes for Solo Leveling Sci-Fi HUD Aesthetic
    PURPLE = "\033[38;5;129m"
    CYAN = "\033[38;5;51m"
    GREEN = "\033[38;5;46m"
    YELLOW = "\033[38;5;226m"
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"

    print(f"""{PURPLE}
╔══════════════════════════════════════════════════════════════════════╗
║             MIKU SOVEREIGN AGI ARCHITECTURE: MASTER CLI              ║
║         Phase I-V Integrated | Air-Gapped Edge Hardware Node         ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
    """)

    print(f"{CYAN}[Terminal Boot]{RESET} Initializing sovereign AGI daemon...")
    daemon = SovereignMikuDaemon()

    print(f"{CYAN}[Terminal Boot]{RESET} Activating Shannon Entropy Dual-System Router...")
    entropy_engine = upgrade_daemon_cognitive_routing(daemon, entropy_threshold=1.5)

    print(f"{CYAN}[Terminal Boot]{RESET} Loading Structural Causal Model (SCM)...")
    causal_graph = OS_CausalGraph(node_names=["CPU", "RAM", "Network", "ScriptExecution", "MikuReward"])
    patch_causal_world_model(daemon, causal_graph=causal_graph, ate_threshold=0.0)

    print(f"{CYAN}[Terminal Boot]{RESET} Mounting Category Theory Knowledge Domains...")
    cat_fs = build_filesystem_category(dim=64)
    cat_dom = build_webdom_category(dim=64)
    known_categories = {"FileSystem": cat_fs, "WebDOM": cat_dom}

    print(f"{GREEN}✓ Miku Sovereign Terminal online and fully synchronized.{RESET}\n")
    print(f"Type your prompt or task below. Type 'status' for Hunter stats, or 'exit' to shutdown.\n")

    while True:
        try:
            user_input = input(f"{CYAN}Miku-CLI >{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{PURPLE}[Shutdown]{RESET} Sovereign node powering down.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print(f"{PURPLE}[Shutdown]{RESET} Preserving experience buffers and closing kernel.")
            break
        if user_input.lower() == "status":
            SystemLifeOSBridge.render_hunter_status()
            continue

        print(f"{YELLOW}[Processing]{RESET} Evaluating cognitive turn...")
        start_time = asyncio.get_event_loop().time()

        # Execute cognitive step through the entropy router and causal world model
        try:
            result = await daemon.cognitive_step(user_input)
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000

            print(f"\n{GREEN}--- [Miku Cognitive Output] ---{RESET}")
            print(f"  • Mode:          {result.get('cognitive_mode', 'SYSTEM_1_FAST')}")
            print(f"  • Entropy:       {result.get('entropy_bits', 0.0):.2f} bits")
            print(f"  • Turn Latency:  {elapsed:.2f} ms")
            print(f"  • Action Type:   {result.get('action_type', 'text')}")
            print(f"  • Response:\n{result.get('response', '[No Output]')}\n")

        except Exception as e:
            print(f"\n{ORANGE}[Execution Error]{RESET} {e}\n")


if __name__ == "__main__":
    asyncio.run(main_terminal())
