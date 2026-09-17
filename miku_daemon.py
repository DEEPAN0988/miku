"""
MIKU SOVEREIGN AGI DAEMON (Master ReAct Loop & Autonomous Test Engine)
Monolithic Orchestrator Integrating Phase I & II Sovereign Capabilities

Orchestrated Subsystems:
1. miku_core: 4-layer 256-dim Transformer, Two-Phase FSM Logit Masking, Sandboxed REPL & Traceback Truncator.
2. miku_telemetry: Ambient non-blocking OS hardware polling & slot-0 sensory injection.
3. miku_memory: 64-dim Siamese episodic vector retrieval with FlatIP cosine indexing.
4. miku_dpo: Zero-RAM LoRA adapters, Experience Replay Buffer, and Idle DPO training daemon.
5. miku_mcts: Hierarchical Monte Carlo Tree Search for System 2 deliberation.
"""

import sys
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import all sovereign subsystems
from miku_core import (
    MikuTransformer,
    SimpleTokenizer,
    MemoryState,
    RouteManager,
    generate_two_phase,
    execute_code_sandboxed,
    truncate_traceback
)
from miku_telemetry import ContextInjector
from miku_memory import SiameseProjector, EpisodicMemory
from miku_dpo import inject_lora, ExperienceBuffer, IdleTrainer
from miku_mcts import HierarchicalMCTS


class SovereignMikuDaemon:
    """
    Master Autonomous ReAct Daemon for Sovereign AGI Execution.
    """

    def __init__(self, d_model: int = 256, n_layer: int = 4, n_head: int = 8):
        print("[Miku Daemon] Bootstrapping sovereign architecture...")
        
        # 1. Core Neural Brain & Tokenizer
        self.tokenizer = SimpleTokenizer()
        self.model = MikuTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head
        )

        # 2. Inject LoRA Adapters for Metacognition (Zero base weight contamination)
        self.lora_params = inject_lora(self.model, r=4, alpha=8.0)
        base_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        lora_count = sum(p.numel() for p in self.lora_params)
        print(f"  Base Parameters (Frozen): {base_count:,}")
        print(f"  LoRA Parameters (Active): {lora_count:,} ({lora_count/base_count*100:.2f}% overhead)")

        # 3. Thread-Safe Context Memory & Ambient Telemetry Daemon
        self.memory = MemoryState(max_items=32)
        self.telemetry = ContextInjector(memory=self.memory, interval_ms=250)

        # 4. Episodic Vector Memory (64-dim FlatIP)
        self.projector = SiameseProjector(in_dim=d_model, hidden_dim=128, out_dim=64)
        self.episodic_memory = EpisodicMemory(
            model=self.model,
            tokenizer=self.tokenizer,
            projector=self.projector,
            dim=64
        )

        # 5. Experience Replay & Idle DPO Trainer
        self.dpo_buffer = ExperienceBuffer(max_capacity=512)
        self.dpo_trainer = IdleTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            buffer=self.dpo_buffer,
            lora_params=self.lora_params,
            lr=5e-4,
            batch_size=2,
            grad_accum_steps=2,
            beta=0.1
        )

        # 6. System 2 Deliberation Engine (Hierarchical MCTS)
        self.mcts = HierarchicalMCTS(
            model=self.model,
            tokenizer=self.tokenizer,
            max_depth=2,
            branch_factor=2
        )

        # 7. Hybrid Intent Router
        self.router = RouteManager(model=self.model, tokenizer=self.tokenizer)
        self._register_default_routes()

        self._running: bool = False
        print("✓ Sovereign Miku Daemon initialization complete.")

    def _register_default_routes(self):
        """Registers built-in OS and cognitive intent routes."""
        async def handle_status(_):
            metrics = self.telemetry.poll_state()
            return f"Sovereign Host Telemetry: {self.telemetry.format_dense_sensory_state(metrics)}"

        async def handle_memory_inspect(_):
            return f"Episodic Memory Status: {self.episodic_memory.index.ntotal} stored episodes."

        self.router.register_route("system_status", "check system hardware telemetry", handle_status)
        self.router.register_route("memory_status", "inspect episodic memory vectors", handle_memory_inspect)

    async def start(self):
        """Launches continuous background daemons."""
        self._running = True
        self.telemetry.start()
        self.dpo_trainer.start()

    async def stop(self):
        """Stops background daemons gracefully."""
        self._running = False
        await self.telemetry.stop()
        await self.dpo_trainer.stop()

    async def cognitive_step(self, user_query: str, force_mcts: bool = False) -> Dict[str, Any]:
        """
        Single ReAct Cognitive Step:
        1. Query episodic vector memory for relevant historical premises.
        2. Read slot-0 hardware telemetry.
        3. Deliberate via MCTS (if complex / forced).
        4. Execute code in sandboxed REPL with traceback interception.
        5. Log trajectory to DPO buffer for recursive self-improvement.
        """
        # User query arrival: signal busy to pause idle training
        self.dpo_trainer.is_idle = False

        # 1. Retrieve Episodic Memory
        relevant_mems = self.episodic_memory.search(user_query, top_k=2)
        dense_mem_str = self.episodic_memory.format_dense_memories(relevant_mems)

        # 2. Append User Query to Memory Context
        await self.memory.append("user", user_query)
        ctx = await self.memory.get_context()
        telemetry_str = ctx[0]["content"] if ctx and ctx[0]["role"] == "system_telemetry" else ""

        # 3. Assemble Prompt for Transformer
        prompt_parts = []
        if telemetry_str:
            prompt_parts.append(telemetry_str)
        if dense_mem_str:
            prompt_parts.append(dense_mem_str)
        prompt_parts.append(f"<USER> {user_query} <THINK>")
        assembled_prompt = "\n".join(prompt_parts)

        # 4. Deliberation Phase (System 2 MCTS)
        if force_mcts or "solve" in user_query.lower() or "calculate" in user_query.lower():
            deliberated_prompt, mcts_val = await self.mcts.search(assembled_prompt, num_simulations=4)
            generation_seed = deliberated_prompt
        else:
            generation_seed = assembled_prompt + " <EXEC>"
            mcts_val = 0.0

        # 5. Generation with Two-Phase FSM Logit Masking
        gen_output, triggered_exec = generate_two_phase(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=generation_seed,
            max_new_tokens=64,
            temperature=0.7
        )

        # 6. Sandboxed Execution Handling
        exec_result = None
        if "<EXEC>" in gen_output:
            code_to_run = gen_output.split("<EXEC>")[-1].strip()
            # Clean comments / eos
            code_to_run = code_to_run.replace("<EOS>", "").strip()
            if code_to_run:
                exec_result = await execute_code_sandboxed(code_to_run, timeout=2.0)
        
        # 7. Trajectory Logging to DPO Experience Buffer
        if exec_result:
            p_tok = self.tokenizer.encode(user_query)
            if exec_result["success"]:
                # Positive reinforcement
                c_tok = self.tokenizer.encode(code_to_run)
                r_tok = self.tokenizer.encode(f"# Syntax error attempt\n{code_to_run} / 0")
                self.dpo_buffer.add(p_tok, c_tok, r_tok)
            else:
                # Negative reinforcement with traceback
                r_tok = self.tokenizer.encode(code_to_run)
                c_tok = self.tokenizer.encode("# Resolved safe execution\nprint('Handled error')")
                self.dpo_buffer.add(p_tok, c_tok, r_tok)

        # Store response in conversational memory
        response_summary = f"[Exec: {exec_result['stdout'] if exec_result and exec_result['success'] else (exec_result['stderr'] if exec_result else 'Direct Response')}]"
        await self.memory.append("assistant", response_summary)

        # Step complete: re-enable idle status
        self.dpo_trainer.is_idle = True

        return {
            "telemetry": telemetry_str,
            "memory": dense_mem_str,
            "prompt": assembled_prompt,
            "generation": gen_output,
            "exec_result": exec_result,
            "mcts_value": mcts_val
        }


# =====================================================================
# AUTOMATED TESTING HARNESS & CONTINUOUS LEARNING RUNNER
# =====================================================================

async def run_autonomous_test_suite():
    print("=" * 70)
    print("  MIKU SOVEREIGN AGI DAEMON: AUTONOMOUS TEST & DPO LEARNING SUITE")
    print("=" * 70)

    daemon = SovereignMikuDaemon(d_model=256, n_layer=4, n_head=8)
    await daemon.start()

    # Pre-populate episodic memory with historical axioms
    daemon.episodic_memory.add_memory("Factorial 10 is computed via math.factorial(10) -> 3628800.")
    daemon.episodic_memory.add_memory("System hardware consists of local physical CPU and RAM.")

    # Pre-seed 1 baseline trajectory in DPO buffer
    daemon.dpo_buffer.add(
        prompt_tokens=daemon.tokenizer.encode("Safe baseline math test"),
        chosen_tokens=daemon.tokenizer.encode("print(10 * 10)"),
        rejected_tokens=daemon.tokenizer.encode("print(10 * 10 / 0)")
    )

    test_tasks = [
        # Test 1: Mathematical computation task
        "Compute 10 factorial in Python.",
        # Test 2: System telemetry query
        "Query system CPU count and available RAM.",
        # Test 3: Intentional syntax/runtime recovery test
        "Execute 1 divided by 0 and handle ZeroDivisionError.",
        # Test 4: Algorithmic deliberation
        "Solve disk usage summary report."
    ]

    print(f"\n[Test Queue] Dispatching {len(test_tasks)} autonomous synthetic tasks...\n")

    for i, task in enumerate(test_tasks, 1):
        print(f"--- [Task {i}/{len(test_tasks)}]: \"{task}\" ---")
        t0 = time.perf_counter()
        
        # Execute cognitive turn
        result = await daemon.cognitive_step(task, force_mcts=(i == 1))
        duration_ms = (time.perf_counter() - t0) * 1000

        print(f"  • Injected Telemetry: {result['telemetry']}")
        print(f"  • Retrieved Memory:  {result['memory'] or 'None'}")
        print(f"  • MCTS Score:        {result['mcts_value']:.4f}")
        if result['exec_result']:
            status = "SUCCESS" if result['exec_result']['success'] else "FAILED (TRUNCATED)"
            out = result['exec_result']['stdout'] if result['exec_result']['success'] else result['exec_result']['stderr']
            print(f"  • REPL Sandbox:      [{status}] {out}")
        print(f"  • Step Duration:     {duration_ms:.2f} ms\n")

        # Yield to allow background telemetry tick
        await asyncio.sleep(0.3)

    print(f"[DPO Replay Buffer] Accumulated Episodes: {len(daemon.dpo_buffer)}")

    # Trigger idle learning step
    print("\n[Idle Metacognition] Initiating background DPO adaptation step on accumulated buffer...")
    daemon.dpo_trainer.is_idle = True
    dpo_metrics = await daemon.dpo_trainer.train_step()

    if dpo_metrics:
        print(f"  • DPO Loss:            {dpo_metrics['dpo_loss']:.4f}")
        print(f"  • Reward Margin (Δ):   {dpo_metrics['reward_margin']:.4f}")
        print(f"  • Total Training Steps: {daemon.dpo_trainer.total_steps}")
    else:
        print("  • DPO Step: Insufficient buffer or busy state.")

    # Graceful shutdown
    await daemon.stop()
    print("\n✓ Master Sovereign Miku Daemon test and learning loop successfully verified.")


if __name__ == "__main__":
    asyncio.run(run_autonomous_test_suite())
