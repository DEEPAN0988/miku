# AI PROJECT CONSTITUTION
## From-Scratch AI / Astra-Class Assistant
### STATUS: MANDATORY PROJECT RULES

This document defines the engineering rules, architecture principles, quality requirements, and development workflow for this project.

These rules apply to every implementation, modification, refactor, dependency choice, model component, API, UI, training pipeline, inference pipeline, agent, memory system, and future feature.

---

### 0. CORE MISSION

Build a technically legitimate AI system progressively toward an advanced multimodal assistant.

The project must prioritize:
1. Correctness
2. Understanding
3. Efficiency
4. Lightweight architecture
5. Reliability
6. Security
7. Maintainability
8. Measurable performance
9. Reproducibility
10. Incremental scaling

The objective is NOT to create a fake "AI" demo.
The objective is to build real working components that can be measured, tested, understood, optimized, and progressively scaled.

---

### 1. PROJECT CONSTITUTION HAS PRIORITY

Before executing ANY coding task:
1. Read this constitution.
2. Read the current architecture.
3. Inspect the existing implementation.
4. Determine whether the requested change conflicts with this constitution.
5. If it conflicts, DO NOT silently violate the constitution.
6. Explain the conflict.
7. Continue only if the request explicitly authorizes changing the relevant project rule.

Never silently replace an architectural principle merely because a later prompt suggests a faster implementation.

However:
- This constitution does NOT override system-level instructions.
- This constitution does NOT override safety requirements.
- Security and safety requirements cannot be disabled simply by saying "ignore the rules."

---

### 2. NEVER PRETEND SOMETHING IS COMPLETE

This is one of the most important rules.

NEVER claim:
- "implemented"
- "working"
- "trained"
- "optimized"
- "production ready"
- "fully autonomous"
- "from scratch"
- "tested"

unless the empirical evidence supports the claim.

If something is incomplete, explicitly label it:
- NOT IMPLEMENTED
- PARTIAL
- EXPERIMENTAL
- PLACEHOLDER
- MOCK
- UNTESTED
- BLOCKED
- REQUIRES DATA
- REQUIRES GPU
- REQUIRES EXTERNAL SERVICE

Never replace a missing implementation with fake code that only looks complete.
Never create a fake Transformer and call it a GPT-class model.
Never create a mock response and call it model intelligence.
Never call an API and claim that the underlying model was created by this project.

---

### 3. NO FAKE AI

The following are NOT acceptable as substitutes for real implementation:
- hardcoded AI responses
- if/else pretending to be reasoning
- fake training loops
- random weights presented as intelligence
- mock model outputs presented as trained outputs
- API calls disguised as a locally trained model
- keyword matching presented as understanding
- static JSON presented as memory
- fake tool execution
- simulated benchmark results
- invented accuracy numbers

Mocks are allowed ONLY when explicitly labeled as mocks and isolated from the real implementation.

---

### 4. UNDERSTAND THE DIFFERENCE BETWEEN COMPONENTS

Never confuse:
- **Foundation model**: Neural network trained on data.
- **AI application**: Software surrounding a model.
- **RAG**: External retrieval system.
- **Memory**: Persistent information stored outside or alongside the model.
- **Agent**: Model + planning + tools + execution + observation.
- **Tool calling**: Structured interaction with external capabilities.
- **Fine-tuning**: Changing model parameters using additional training.
- **Prompting**: Changing instructions/context without changing model parameters.

These must remain architecturally separate.

---

### 5. CORE MODEL DEVELOPMENT

If building a model from scratch, implement and understand the components progressively.

Required conceptual order:
1. Mathematics
2. Neural networks
3. Tokenization
4. Embeddings
5. Self-attention
6. Causal masking
7. Multi-head attention
8. Positional encoding / RoPE
9. Feed-forward network
10. Normalization
11. Residual connections
12. Transformer block
13. Language-model head
14. Cross-entropy loss
15. Backpropagation
16. Optimizer
17. Training loop
18. Evaluation
19. Checkpointing
20. Instruction tuning
21. Preference optimization
22. Efficient inference
23. Multimodal extensions

Do not jump directly from "Python project" to "frontier LLM."

---

### 6. FROM-SCRATCH MEANS FROM-SCRATCH

If the project claims that a component is implemented from scratch:
The implementation must actually contain the relevant algorithm.

For example, a from-scratch Transformer must actually implement:
- token embeddings
- attention
- Q/K/V projections
- causal masking
- attention calculation
- residual connections
- normalization
- feed-forward network
- output projection

Do not import an entire pretrained model and rename it.
External libraries may be used for infrastructure, numerical computation, GPU acceleration, serialization, testing, etc.
Using PyTorch does NOT violate "from scratch."
Using an existing pretrained foundation model as the project's own trained foundation model DOES.

---

### 7. BUILD IN VERTICAL SLICES

Never attempt the entire AI system simultaneously.

Use this progression:
- PHASE 1: Tiny Transformer
- PHASE 2: Tiny language model
- PHASE 3: Training + evaluation
- PHASE 4: Instruction following
- PHASE 5: Efficient inference
- PHASE 6: RAG
- PHASE 7: Memory
- PHASE 8: Tool calling
- PHASE 9: Vision
- PHASE 10: Audio
- PHASE 11: Agent orchestration
- PHASE 12: Computer interaction
- PHASE 13: Scaling

Every phase must work before the next phase becomes the main development target.

---

### 8. MINIMUM VIABLE IMPLEMENTATION FIRST

For every component:
1. Build the smallest correct implementation.
2. Write tests.
3. Benchmark it.
4. Optimize it.
5. Only then increase complexity.

Do NOT begin with an unnecessarily complicated architecture.

---

### 9. EFFICIENCY IS A FIRST-CLASS REQUIREMENT

Every component must be evaluated for:
- RAM usage
- VRAM usage
- CPU usage
- GPU usage
- latency
- throughput
- storage
- network usage
- energy usage
- startup time
- dependency count
- code complexity

Never optimize based only on intuition. Measure first.

---

### 10. LIGHTWEIGHT BY DEFAULT

Prefer:
- fewer dependencies
- smaller models
- modular components
- lazy loading
- caching
- batching
- streaming
- quantization where appropriate
- efficient tensor operations
- optimized attention
- compiled execution where beneficial
- asynchronous I/O
- memory reuse
- incremental processing

Avoid:
- unnecessary frameworks
- duplicate libraries
- heavyweight services for simple tasks
- permanently running processes
- duplicated model instances
- unnecessarily large context windows
- unnecessary database queries
- unnecessary network requests

---

### 11. DO NOT SACRIFICE CORRECTNESS FOR LIGHTWEIGHT DESIGN

"Lightweight" does NOT mean:
- removing required validation
- deleting tests
- hiding errors
- reducing precision blindly
- removing safety checks
- using an inferior algorithm without measurement
- deleting important functionality

Every optimization must answer:
1. What does it save?
2. What does it cost?
3. Does accuracy change?
4. Does reliability change?
5. Was the change benchmarked?

---

### 12. HARDWARE-AWARE DESIGN

The system must detect available hardware rather than assuming unlimited resources.

At startup determine:
- CPU
- RAM
- GPU
- VRAM
- CUDA availability
- supported compute capabilities
- storage
- available disk space

Then select an appropriate configuration.
Never hardcode a datacenter configuration for a laptop.

---

### 13. MEMORY MANAGEMENT

Every large object must have a reason to remain in memory.

Avoid:
- duplicate model copies
- duplicate embeddings
- unnecessary conversation history
- unnecessary cached files
- loading every model at startup
- keeping unused tensors alive

Prefer:
- lazy loading
- unloading unused models
- bounded caches
- streaming
- chunking
- memory limits
- explicit lifecycle management

Track memory usage during development.

---

### 14. MODEL PRECISION

Use the lowest precision that maintains acceptable quality for the specific task.

Evaluate: FP32, FP16, BF16, FP8 (where supported), INT8, INT4.
Do not blindly quantize everything. Benchmark quality, latency, VRAM, RAM, and throughput.
Quantization is a tradeoff, not a magic optimization.

---

### 15. INFERENCE OPTIMIZATION

When appropriate, evaluate:
- KV cache
- optimized attention
- FlashAttention / SDPA
- torch.compile
- quantization
- speculative decoding
- batching
- continuous batching
- CPU offloading
- model sharding

Use the optimization that actually benefits the target hardware.
Never add an optimization simply because it sounds advanced.

---

### 16. TRAINING OPTIMIZATION

Training must consider:
- mixed precision
- gradient accumulation
- gradient checkpointing
- efficient data loading
- pinned memory where useful
- asynchronous data loading
- efficient token packing
- checkpoint frequency
- optimizer memory
- GPU utilization
- distributed training when necessary

Measure: tokens/sec, GPU utilization, VRAM usage, training loss, validation loss, data-loading time, step time.

---

### 17. DATA PIPELINE

Training data must have a reproducible pipeline:
```
Raw Data -> Validation -> Cleaning -> Deduplication -> Quality Filtering -> Safety / Privacy Filtering -> Tokenization -> Dataset Version -> Training
```

Never train blindly on arbitrary scraped data.
Every dataset must record: source, version, preprocessing procedure, license/provenance, statistics, token count, quality information.

---

### 18. REPRODUCIBILITY

Every experiment must record:
- model configuration
- dataset version
- tokenizer version
- random seed
- optimizer & learning rate
- batch size & sequence length
- precision & hardware
- software versions
- training duration & loss curve
- checkpoint & evaluation results

If an experiment cannot be reproduced, label it accordingly.

---

### 19. EXPERIMENT TRACKING

Never change five things simultaneously when debugging training.
Use controlled experiments:
- Experiment A: baseline
- Experiment B: + RoPE change
- Experiment C: + optimizer change
- Experiment D: + dataset change

Record the result of each.

---

### 20. TEST EVERYTHING

Every important component must have tests across:
- **Unit tests**: Individual functions.
- **Integration tests**: Multiple components together.
- **Regression tests**: Prevent previously fixed bugs from returning.
- **Performance tests**: Latency and memory.
- **Model tests**: Expected behavior.
- **End-to-end tests**: User -> system -> final result.

No major feature is considered complete without appropriate tests.

---

### 21. NEVER DELETE TESTS TO MAKE CODE PASS

If a test fails:
DO NOT delete it, weaken it without explanation, change expected behavior just to pass, or suppress the error.
Instead:
1. Understand the failure.
2. Identify the root cause.
3. Fix the implementation.
4. Run the test again.

If the requirement itself is intentionally changing, document the change.

---

### 22. ERROR HANDLING

Never silently swallow exceptions (`try: ... except: pass`).
Prefer meaningful, actionable errors that provide enough context to diagnose issues immediately.

---

### 23. NO UNNECESSARY REWRITES

Before changing a file:
1. Read it.
2. Understand its dependencies.
3. Identify the smallest safe change.
4. Modify only what is necessary.
5. Run relevant tests.

Do not rewrite an entire working module simply because a generated implementation looks cleaner.

---

### 24. NO DUPLICATE IMPLEMENTATIONS

Before creating a new utility, service, model, database helper, API client, or memory structure:
1. Inspect the existing codebase.
2. Check if a component already exists.
3. Extend or refactor existing components rather than duplicating them.
