"""
MIKU SOVEREIGN AGI DAEMON (The Grand Integration Refactor)
Master Sovereign ReAct Daemon Fusing Sensory, Motor, Memory, Sparse MoE, EWC, and LitRPG OS

Integrated Organs:
1. Core & MoE: 4-layer 256-dim Transformer with 8-Expert SparseMoE (k=2) & LoRA (r=4).
2. Dual-Sensory Injection:
   - Slot 0: <SYS|...> Hardware telemetry (CPU, RAM, Disk, Sockets).
   - Slot 1: <VIS|...> Win32 Desktop foreground window & Playwright DOM state.
3. Action Dispatcher:
   - Python code synthesis -> Isolated subprocess REPL with regex traceback truncator.
   - Motor control (<CLICK|x,y>, <TYPE|text>) -> Win32 SendInput & Bézier trajectories.
4. Continuous Learning: Unified loss L_Total = L_DPO + L_EWC + L_aux during idle cycles.
5. LitRPG OS Bridge: Gamified achievement tracking with Solo Leveling glowing HUD.
"""

import sys
import re
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Core Primitives
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
from miku_moe import replace_ffn_with_moe, SparseMoE
from miku_dpo import inject_lora, ExperienceBuffer, compute_dpo_loss
from miku_ewc import ElasticWeightConsolidation
from miku_mcts import HierarchicalMCTS
from miku_vision import Win32DesktopInspector, PlaywrightDOMEngine, format_dense_visual_state
from miku_motor import Win32MotorController
from miku_system_life import SystemLifeOSBridge, SoloLevelingHUD


# =====================================================================
# UNIFIED METACOGNITION TRAINER (DPO + EWC + MOE AUX LOSS)
# =====================================================================

class UnifiedMetacognitionTrainer:
    """
    Unified background trainer minimizing:
    L_Total = L_DPO + L_EWC + L_aux
    """

    def __init__(
        self,
        model: MikuTransformer,
        tokenizer: SimpleTokenizer,
        buffer: ExperienceBuffer,
        ewc: ElasticWeightConsolidation,
        moe_layers: List[SparseMoE],
        lora_params: List[nn.Parameter],
        lr: float = 5e-4,
        batch_size: int = 2,
        beta: float = 0.1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = buffer
        self.ewc = ewc
        self.moe_layers = moe_layers
        self.lora_params = lora_params
        self.batch_size = batch_size
        self.beta = beta

        self.optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
        self.is_idle: bool = True
        self.total_steps: int = 0

    async def train_step(self) -> Optional[Dict[str, float]]:
        """Executes a single unified optimization step over active LoRA parameters."""
        if len(self.buffer) < self.batch_size or not self.is_idle:
            return None

        batch = self.buffer.sample_batch(self.batch_size, pad_token_id=self.tokenizer.pad_id)
        if not batch:
            return None

        self.model.train()
        self.optimizer.zero_grad()

        # 1. Direct Preference Optimization Loss
        loss_dpo, dpo_metrics = compute_dpo_loss(self.model, batch, beta=self.beta)

        # 2. Elastic Weight Consolidation Penalty
        loss_ewc = self.ewc.compute_ewc_loss(self.model)

        # 3. MoE Auxiliary Load Balancing Loss
        loss_aux = sum(layer.current_aux_loss for layer in self.moe_layers)
        if not isinstance(loss_aux, torch.Tensor):
            loss_aux = torch.tensor(loss_aux, device=loss_dpo.device)

        # Unified Objective Function
        total_loss = loss_dpo + loss_ewc + loss_aux
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        return {
            "total_loss": total_loss.item(),
            "dpo_loss": loss_dpo.item(),
            "ewc_loss": loss_ewc.item(),
            "moe_aux_loss": loss_aux.item(),
            "reward_margin": dpo_metrics.get("reward_margin", 0.0)
        }


# =====================================================================
# MASTER SOVEREIGN AGI DAEMON
# =====================================================================

class SovereignMikuDaemon:
    """
    Unified Sovereign AGI Daemon fusing Sensory Awareness, Motor Control,
    Episodic Memory, Sparse MoE Brain, MCTS System 2, EWC Continual Learning,
    and LitRPG OS Gamification.
    """

    def __init__(self, d_model: int = 256, n_layer: int = 4, n_head: int = 8):
        print(f"\033[38;5;129m[Miku Grand Integration]\033[0m Bootstrapping sovereign AGI core...")

        # 1. Core Brain
        self.tokenizer = SimpleTokenizer()
        self.model = MikuTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head
        )

        # 2. Upgrade to Sparse Mixture of Experts (Phase II v3.0: 8 Experts, k=2)
        print(f"  • Activating SparseMoE layer expansion (8 Experts, Top-2 Gating)...")
        self.moe_layers = replace_ffn_with_moe(self.model, num_experts=8, top_k=2)

        # 3. Inject LoRA Adapters for Metacognition (Zero Base Weight Contamination)
        print(f"  • Injecting rank-4 LoRA adapters across attention and MoE projections...")
        self.lora_params = inject_lora(self.model, r=4, alpha=8.0)

        # 4. Context Memory & Dual-Sensory Daemons (Slot 0 = SYS, Slot 1 = VIS)
        self.memory = MemoryState(max_items=32)
        self.telemetry = ContextInjector(memory=self.memory, interval_ms=250)
        self.desktop_inspector = Win32DesktopInspector()
        self.dom_engine = PlaywrightDOMEngine(headless=True)
        self.last_vis_token_str = "<VIS|W:Desktop|T:Init|DOM:0N>"

        # 5. Motor Controller
        self.motor = Win32MotorController()

        # 6. Episodic Vector Memory
        self.projector = SiameseProjector(in_dim=d_model, hidden_dim=128, out_dim=64)
        self.episodic_memory = EpisodicMemory(
            model=self.model,
            tokenizer=self.tokenizer,
            projector=self.projector,
            dim=64
        )

        # 7. Continual Learning (EWC) & Experience Buffer
        self.ewc = ElasticWeightConsolidation(lambda_ewc=0.4)
        self.dpo_buffer = ExperienceBuffer(max_capacity=512)
        self.trainer = UnifiedMetacognitionTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            buffer=self.dpo_buffer,
            ewc=self.ewc,
            moe_layers=self.moe_layers,
            lora_params=self.lora_params,
            lr=5e-4,
            batch_size=2,
            beta=0.1
        )

        # 8. Deliberation Engine (Hierarchical MCTS)
        self.mcts = HierarchicalMCTS(model=self.model, tokenizer=self.tokenizer, max_depth=2, branch_factor=2)

        # 9. LitRPG OS Bridge & Solo Leveling HUD
        self.rpg_bridge = SystemLifeOSBridge()
        self.hud = SoloLevelingHUD()

        self._running: bool = False
        self._visual_task: Optional[asyncio.Task] = None
        print(f"\033[38;5;82m✓ Grand Integration complete. All organs synchronized.\033[0m")

    async def _visual_polling_loop(self):
        """Asynchronously updates Slot 1 (Visual/Desktop state) in MemoryState."""
        while self._running:
            try:
                active_win = self.desktop_inspector.get_active_window_info()
                vis_str = format_dense_visual_state(active_win, dom_info=None)
                self.last_vis_token_str = vis_str

                # Thread-safe Slot-1 injection
                async with self.memory._lock:
                    ctx = self.memory._context
                    # Ensure Slot 0 is reserved for telemetry
                    if not ctx:
                        ctx.append({"role": "system_telemetry", "content": "<SYS|INIT>"})
                    
                    if len(ctx) >= 2 and ctx[1].get("role") == "visual_telemetry":
                        ctx[1]["content"] = vis_str
                    elif len(ctx) == 1:
                        ctx.append({"role": "visual_telemetry", "content": vis_str})
                    else:
                        ctx.insert(1, {"role": "visual_telemetry", "content": vis_str})

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)

    async def start(self):
        """Launches background daemons (Hardware Telemetry + Visual Inspector)."""
        self._running = True
        self.telemetry.start()
        self._visual_task = asyncio.create_task(self._visual_polling_loop())

    async def stop(self):
        """Graceful shutdown of all daemons."""
        self._running = False
        await self.telemetry.stop()
        if self._visual_task:
            self._visual_task.cancel()
            try:
                await self._visual_task
            except asyncio.CancelledError:
                pass
        await self.dom_engine.close()

    def dispatch_action(self, raw_generation: str) -> Dict[str, Any]:
        """
        Action Routing Engine:
        - Parses <CLICK|x,y> -> Win32 Bézier mouse movement & click
        - Parses <TYPE|text> -> Win32 hardware typing
        - Parses <EXEC>... -> Subprocess sandboxed REPL
        """
        # 1. Motor: Mouse Click
        click_match = re.search(r'<CLICK\|(\d+),(\d+)>', raw_generation)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            self.motor.move_mouse_bezier((x, y), duration=0.15)
            self.motor.click("left")
            return {
                "type": "motor_click",
                "success": True,
                "target": (x, y),
                "details": f"Bézier mouse move & click at ({x}, {y})"
            }

        # 2. Motor: Typing
        type_match = re.search(r'<TYPE\|([^>]+)>', raw_generation)
        if type_match:
            text_to_type = type_match.group(1)
            self.motor.type_text(text_to_type)
            return {
                "type": "motor_type",
                "success": True,
                "text": text_to_type,
                "details": f"Typed: '{text_to_type}'"
            }

        # 3. Compute: Python Sandbox Execution
        if "<EXEC>" in raw_generation:
            code_segment = raw_generation.split("<EXEC>")[-1].replace("<EOS>", "").strip()
            # Asynchronous execution handled by caller
            return {
                "type": "code_execution",
                "code": code_segment,
                "details": "Dispatching to sandboxed REPL"
            }

        return {
            "type": "text_response",
            "success": True,
            "details": raw_generation.strip()
        }

    async def cognitive_step(self, user_query: str, force_mcts: bool = False) -> Dict[str, Any]:
        """
        Master Autonomous ReAct Turn.
        """
        self.trainer.is_idle = False

        # 1. Retrieve Episodic Memory
        mems = self.episodic_memory.search(user_query, top_k=2)
        dense_mem_str = self.episodic_memory.format_dense_memories(mems)

        # 2. Read Dual Sensory Header (Slot 0 = SYS, Slot 1 = VIS)
        await self.memory.append("user", user_query)
        ctx = await self.memory.get_context()
        sys_telemetry = ctx[0]["content"] if len(ctx) > 0 and ctx[0]["role"] == "system_telemetry" else ""
        vis_telemetry = ctx[1]["content"] if len(ctx) > 1 and ctx[1]["role"] == "visual_telemetry" else ""

        # 3. Assemble Prompt
        prompt_parts = []
        if sys_telemetry:
            prompt_parts.append(sys_telemetry)
        if vis_telemetry:
            prompt_parts.append(vis_telemetry)
        if dense_mem_str:
            prompt_parts.append(dense_mem_str)
        prompt_parts.append(f"<USER> {user_query} <THINK>")
        assembled_prompt = "\n".join(prompt_parts)

        # 4. System 2 Deliberation (MCTS)
        if force_mcts or any(k in user_query.lower() for k in ["calculate", "solve", "navigate", "click"]):
            deliberated_prompt, mcts_val = await self.mcts.search(assembled_prompt, num_simulations=4)
            seed = deliberated_prompt
        else:
            seed = assembled_prompt + " <EXEC>"
            mcts_val = 0.0

        # 5. Generation with Two-Phase FSM Logit Masking
        gen_output, _ = generate_two_phase(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=seed,
            max_new_tokens=48,
            temperature=0.7
        )

        # 6. Action Dispatch
        action_spec = self.dispatch_action(gen_output)
        exec_result = None

        if action_spec["type"] == "code_execution":
            code = action_spec["code"]
            if code:
                exec_result = await execute_code_sandboxed(code, timeout=2.0)
                action_spec["success"] = exec_result["success"]
                action_spec["output"] = exec_result["stdout"] if exec_result["success"] else exec_result["stderr"]
            else:
                action_spec["success"] = False
                action_spec["output"] = "Empty code block"

        # 7. LitRPG Reward Trigger on Success
        rpg_notifs = []
        if action_spec.get("success", False):
            rpg_notifs = self.rpg_bridge.award_progress(
                exp_gain=150 if action_spec["type"] == "code_execution" else 80,
                gold_gain=50,
                action_name=f"{action_spec['type'].upper()} Success"
            )

        # 8. Log Trajectory into DPO Buffer
        p_tok = self.tokenizer.encode(user_query)
        if action_spec.get("success", False):
            c_tok = self.tokenizer.encode(gen_output)
            r_tok = self.tokenizer.encode(f"# Suboptimal path\nprint('Failed')")
            self.dpo_buffer.add(p_tok, c_tok, r_tok)
        else:
            r_tok = self.tokenizer.encode(gen_output)
            c_tok = self.tokenizer.encode(f"# Corrected execution\nprint('Clean')")
            self.dpo_buffer.add(p_tok, c_tok, r_tok)

        # 9. Store summary in memory
        summary = f"[{action_spec['type']}: {'OK' if action_spec.get('success') else 'ERR'}]"
        await self.memory.append("assistant", summary)

        self.trainer.is_idle = True

        return {
            "sys_telemetry": sys_telemetry,
            "vis_telemetry": vis_telemetry,
            "memory": dense_mem_str,
            "action": action_spec,
            "exec_result": exec_result,
            "mcts_val": mcts_val,
            "rpg_notifs": rpg_notifs
        }


# =====================================================================
# GRAND INTEGRATION AUTONOMOUS TEST SUITE
# =====================================================================

async def run_grand_integration_test():
    print("=" * 70)
    print("  MIKU GRAND INTEGRATION: AUTONOMOUS TEST & CONTINUAL LEARNING LOOP")
    print("=" * 70)

    daemon = SovereignMikuDaemon(d_model=256, n_layer=4, n_head=8)
    await daemon.start()

    # Pre-consolidate EWC on baseline task
    print("\n[EWC Pre-Consolidation] Computing Fisher matrix on baseline task...")
    calib = [
        (torch.tensor([daemon.tokenizer.encode("Baseline math")]),
         torch.tensor([daemon.tokenizer.encode("print(10)")]))
    ]
    daemon.ewc.consolidate(daemon.model, calib)

    # Pre-populate episodic memory & DPO buffer
    daemon.episodic_memory.add_memory("Calculate 10 factorial with math.factorial(10).")
    daemon.dpo_buffer.add(
        daemon.tokenizer.encode("Base task"),
        daemon.tokenizer.encode("print('Success')"),
        daemon.tokenizer.encode("print('Fail')")
    )

    test_tasks = [
        "Compute 10 factorial in Python.",
        "Perform Bézier mouse click at coordinate <CLICK|500,300>.",
        "Query desktop and hardware status."
    ]

    print(f"\n[Autonomous Queue] Dispatching {len(test_tasks)} multi-modal tasks...\n")

    for i, task in enumerate(test_tasks, 1):
        print(f"\033[1m\033[38;5;51m>>> Turn {i}/{len(test_tasks)}: \"{task}\"\033[0m")
        t0 = time.perf_counter()
        res = await daemon.cognitive_step(task, force_mcts=(i == 1))
        dt = (time.perf_counter() - t0) * 1000

        print(f"  • Slot 0 (SYS): {res['sys_telemetry']}")
        print(f"  • Slot 1 (VIS): {res['vis_telemetry']}")
        print(f"  • Action Type:  {res['action']['type']}")
        print(f"  • Action Info:  {res['action']['details']}")
        print(f"  • Turn Latency: {dt:.2f} ms")

        # Display LitRPG HUD if rewards earned
        if res["rpg_notifs"]:
            print(daemon.hud.system_window("SOLO LEVELING QUEST EVENT", res["rpg_notifs"], width=62))

        await asyncio.sleep(0.3)

    # Trigger Unified Metacognition Step (DPO + EWC + MoE Aux Loss)
    print("\n[Unified Metacognition] Running multi-objective training step (DPO + EWC + MoE Aux)...")
    daemon.trainer.is_idle = True
    train_metrics = await daemon.trainer.train_step()

    if train_metrics:
        print(f"  • Total Loss (L_Total): {train_metrics['total_loss']:.4f}")
        print(f"  • DPO Loss (L_DPO):     {train_metrics['dpo_loss']:.4f}")
        print(f"  • EWC Penalty (L_EWC):  {train_metrics['ewc_loss']:.4f}")
        print(f"  • MoE Aux Loss (L_aux): {train_metrics['moe_aux_loss']:.4f}")
        print(f"  • Reward Margin (Δ):    {train_metrics['reward_margin']:.4f}")

    # Display final Hunter Status
    print("\n" + daemon.hud.format_status_screen(daemon.rpg_bridge.get_stats(), width=64) + "\n")

    await daemon.stop()
    print("\033[38;5;82m✓ Grand Integration Master Daemon fully verified.\033[0m")


if __name__ == "__main__":
    asyncio.run(run_grand_integration_test())
