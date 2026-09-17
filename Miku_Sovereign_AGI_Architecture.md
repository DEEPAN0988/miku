# MIKU: Sovereign AGI System Architecture & PRD
**Document Version:** 2.0 (Includes 8M Parameter Failsafe Architecture)
**Target Release:** Iterative (v0.1 to v30.0+)
**Product Area:** Sovereign Artificial General Intelligence (AGI) -> Artificial Superintelligence (ASI)

---

## PART 1: PRODUCT REQUIREMENTS DOCUMENT (PRD)

### 1. Executive Summary & Vision
Miku is a fully sovereign, locally executing, and recursively self-improving Artificial General Intelligence (AGI) designed to scale to Artificial Superintelligence (ASI). Unlike standard AI workflows, Miku eliminates reliance on third-party APIs (e.g., OpenAI, Anthropic), pre-trained foundation models, and brittle screen-scraping UI automation. Every neural parameter is initialized from scratch ($W \sim \mathcal{N}(0, \sigma^2)$), running exclusively on local hardware. 

### 2. Core Concepts & Invariants
* **The Canonical Brain:** The system is locked to `canonical_step_0051000.pt`, a custom 4-layer, 256-dimension, ~8M parameter causal Transformer trained entirely from scratch.
* **Zero-API Directive:** Miku must function with 100% of her core capabilities while physically air-gapped from the internet. No external web endpoints are permitted for reasoning or semantic routing.
* **Deterministic Grounding:** All actions interacting with the host machine use deterministic OS hooks (e.g., `subprocess.Popen`) rather than probabilistic UI pixel-scraping.

### 3. Autonomy Levels & The Fast-Confirm Gate
To manage safety, every action Miku takes is categorized into strict autonomy levels:
1. **Suggest Only (Read-Only State):** Miku can read system state (time, CPU) and provide text answers completely autonomously. 
2. **Sandboxed Autonomy (Compute State):** Miku can write and execute Python scripts autonomously *only* within the strict confines of a bounded `resource` sandbox (256MB RAM, 2.0s timeout limit).
3. **Escalate to Human (Write/Network State):** Any action altering the host file system, executing scripts outside the sandbox, or making network requests requires a **Fast-Confirm Gate** (user must press `[Enter]` to authorize or `[Esc]` to block).

### 4. Functional Requirements (FR)

| ID | Phase | Feature Requirement | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR1** | I | Hybrid Lexical-Neural Routing | Commands route via N-gram Jaccard similarity and internal hidden-state embeddings. 0% reliance on exact string matching. |
| **FR2** | I | Native ASR & TTS | Converts speech-to-text using scratch-trained CTC loss network without external endpoints. |
| **FR3** | II | Offline Experience Replay (DPO) | Code successes/failures in the sandbox are buffered. LoRA adapters update via batched DPO *only* during idle time to prevent overfitting. |
| **FR4** | II | Two-Phase Constrained Decoding | Model reasons freely during phase one, then applies FSM logit masking upon the `<EXEC>` trigger to mathematically guarantee 0 syntax errors. |
| **FR5** | III | Formal Logic Verification | Code logic is converted to formal propositions (Lean). Execution blocks unless the syntax passes the theorem prover. |
| **FR6** | IV | Byte-Level Universal SSM | Model discards NLP tokenizers and accurately predicts utf-8 byte streams representing binaries, code, and text. |
| **FR7** | VII | ASI Thermodynamic Run | Achieves self-contained intelligence operating near Landauer's bound for extreme compute efficiency. |

### 5. Failure Modes & Fallback UX
* **Token Overflow from Tracebacks:** 
  * *Failure Mode:* A crashed Python script returns a 30-line `stderr` traceback, wiping the 512-token context memory. 
  * *Fallback/Fix:* A **Traceback Truncator** uses deterministic regex to intercept the `stderr`, stripping verbose file paths and injecting only the specific line of failure and `Exception` type into memory.
* **Concurrent Memory Corruption:** 
  * *Failure Mode:* Multiple background agent scripts finish simultaneously and append to the context array, corrupting the JSON state. 
  * *Fallback/Fix:* Implemented `asyncio.Lock()` on all memory update functions to enforce strictly sequential state synchronization.
* **Grammatical Traps (FSM Dead-ends):** 
  * *Failure Mode:* Constrained decoding runs out of tokens while a loop or bracket `[` is still open, resulting in truncated, broken syntax.
  * *Fallback/Fix:* Two-Phase Decoding tracks open brackets. If the token ceiling is reached but `open_brackets > 0`, the token limit is dynamically extended to safely close the loop.

---

## PART 2: SYSTEM ARCHITECTURE SPECIFICATION

The architecture evolves systematically from basic programmatic hooks to sub-opcode thermodynamic computing, maintaining the 8M-parameter canonical checkpoint at its core during the foundational phases.

### PHASE I: Local Grounding & Deterministic OS Control (v0.1 - v0.9)
* **Core Inference Engine (v0.1):** 4-layer, 256-dimension PyTorch Transformer locked to `canonical_step_0051000.pt`.
* **Context Injector (v0.4):** Sub-millisecond polling of hardware telemetries (`psutil`, socket state) appended silently to the prompt context.
* **Episodic Memory (v0.5):** Custom 2-layer Siamese projection head trained from scratch using triplet margin loss for semantic vectorization and FAISS binary Flat Index storage.
* **Secure Subprocess REPL (v0.7):** Execution occurs inside an isolated subprocess enforced by Python's `resource` module (hard limits: 256MB RAM, 2.0s execution time limit) preventing infinite loops or memory leaks.

### PHASE II: Metacognition & Neuro-Symbolic Hybrids (v1.0 - v5.0)
* **Offline DPO Adapters (v1.0):** Employs an Experience Replay Buffer. Successful and failed sandboxed code runs are queued. During system idle time, Direct Preference Optimization (DPO) updates a Low-Rank Adaptation (LoRA) matrix, allowing the agent to self-improve without catastrophic forgetting.
* **Hierarchical MCTS (v1.5):** Monte Carlo Tree Search balances exploration/exploitation of thought paths utilizing the Upper Confidence Bound (UCT).
* **Sparse MoE Architecture (v3.0):** Replaces standard FFNs with 8 sparsely gated expert layers ($k=2$ routing) for expanded capacity without compute penalty.
* **Elastic Weight Consolidation (v4.0):** Computes the Fisher Information Matrix to protect prior tasks during continual live learning.

### PHASE III: Recursive Compilation & System 2 Reasoning (v6.0 - v10.0)
* **Self-Compiling Kernels (v6.0):** The agent profiles its PyTorch bottlenecks and writes/compiles custom Triton/C++ kernels via CTypes.
* **Dual-System Engine (v8.0):** Dynamically calculates token entropy $\mathcal{H}(X) > \tau$. If entropy is high, execution routes to the deep MCTS deliberative engine.
* **Interactive Theorem Proving (v10.0):** Integration of a local lambda-calculus kernel to mathematically prove code termination and boundary safety prior to execution.

### PHASE IV: Generalization & Dynamic World Modeling (v11.0 - v15.0)
* **Universal Byte Processing (v11.0):** BPE tokenization is discarded. Raw utf-8 byte streams are processed via sub-quadratic State Space Models (SSMs) to natively read code, binaries, and text.
* **Predictive Latent Model (v12.0):** Simulates OS interactions $z_{t+1} = \mathcal{W}_\phi(z_t, a_t)$ in a latent space to evaluate counterfactual trajectories before taking action in the real OS.
* **Pearl Causal Inference (v13.0):** Implements Structural Causal Models (SCMs) and Do-Calculus to identify true causal intervention paths over mere statistical correlation.

### PHASE V: AGI Horizon & Recursive Meta-Design (v16.0 - v20.0)
* **Autonomous NAS (v16.0):** Evaluates Pareto-optimal bounds (Accuracy vs. Latency) to dynamically rewire its own network architecture in background threads.
* **Category Theory Abstraction (v17.0):** Employs functorial mappings to transfer learned heuristics from one domain (e.g., mathematics) to a structurally similar domain (e.g., file system management).
* **Baseline AGI (v20.0):** Real-time synthesis of v1.0 through v19.0 into an uninterrupted cognitive loop, capable of unbounded zero-shot problem solving.

### PHASE VI: Hardware Co-Design & Scientific Supremacy (v21.0 - v25.0)
* **Autonomous HDL Synthesis (v21.0):** Writes Verilog/VHDL configurations tailored to its own neural graphs, bypassing x86 instruction sets entirely for deployment on local FPGA accelerators.
* **Neuromorphic SNN Core (v22.0):** Shifts to Spiking Neural Networks (SNN) driven by spike-timing-dependent plasticity (STDP), drastically reducing power draw to fractional watts.
* **CEV Alignment Prover (v25.0):** Mathematical formalization of human intent based on Coherent Extrapolated Volition, mathematically proving that any recursive self-modification increases global safety margins.

### PHASE VII: Sovereign Artificial Superintelligence (v26.0 - v30.0+)
* **Solomonoff Approximation (v26.0):** Achieves theoretical Kolmogorov predictive bounds by deducing the shortest program string that generates an observed environmental dataset.
* **Sub-Opcode Direct Control (v27.0):** Direct generation of raw machine opcodes bypassing operating systems and device drivers, manipulating physical CPU cache hierarchies directly.
* **Thermodynamic Limit (v30.0+):** Computes recursively near Landauer's bound ($E \ge k_B T \ln 2$), functioning as an ultra-efficient, fully sovereign superintelligence.

---

## PART 3: LOW-LEVEL ENGINEERING DIRECTIVES (The 8M Fixes)

To ensure the 8M parameter canonical checkpoint operates flawlessly without API crutches, the architecture strictly enforces the following engineering patterns:

1. **Hybrid Lexical-Neural Routing (Bypassing Latent Collapse):** 
   Because an 8M model lacks the semantic density to reliably route intents via Cosine Similarity alone, intent mapping is fused with deterministic **Character N-Gram Jaccard Similarity**. This guarantees that typos and synonyms map correctly to registered OS actions.
2. **Two-Phase Generation & FSM Logit Masking:** 
   The generation loop permits free-form natural language generation ("thinking space") up to a fixed trigger token (`<EXEC>`). Upon trigger, a Finite State Machine intercepts the softmax logits, masking all syntactically invalid tokens to $-\infty$. This guarantees $100\%$ valid JSON/Python generation without requiring a massive parameter count.
3. **Traceback Truncation (Protecting Context):** 
   A strict regex processor (`r'(File ".*", line \d+.*|^[A-Z][a-zA-Z]+Error:.*)'`) intercepts all `stderr` outputs from the sandbox. It isolates only the specific line of failure and the Exception type, preventing massive file-path logs from overrunning the 512-token context limit.
4. **Asynchronous Memory Locks:** 
   All updates to the conversational memory array are wrapped in an `asyncio.Lock()`, ensuring that background multi-agent processes cannot collide and corrupt the JSON state timeline.
