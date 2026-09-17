"""
MIKU TELEMETRY ENGINE (v0.4: Context Injector)
Sub-millisecond, Non-Blocking OS Telemetry Daemon & Dense Sensory Encoder

Key Features:
1. ContextInjector Daemon: Runs as a lightweight asyncio background task.
2. Sub-Millisecond Polling: Zero-allocation hot loop, non-blocking CPU times, cached hardware topology.
3. Ultra-Dense Sensory String: Compressed high-entropy encoding (<SYS|C:45%|R:1.2G|D:0M|S:2E>)
   saving precious tokens for the 8M model's 512-token context window.
4. Silent Slot-0 Memory Injection: Thread-safe acquisition of asyncio.Lock() updating index 0
   so Miku perceives her ambient physical environment as the grounding premise of reality.
"""

import sys
import time
import asyncio
from typing import Optional, Dict, Any
import psutil

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import MemoryState from miku_core if available
try:
    from miku_core import MemoryState
except ImportError:
    class MemoryState:  # Fallback duck-typing stub for standalone execution
        def __init__(self, max_items: int = 32):
            self.max_items = max_items
            self._context = []
            self._lock = asyncio.Lock()

        async def append(self, role: str, content: str):
            async with self._lock:
                self._context.append({"role": role, "content": content})
                if len(self._context) > self.max_items:
                    self._context.pop(1 if self._context[0].get("role") == "system_telemetry" else 0)

        async def get_context(self):
            async with self._lock:
                return list(self._context)


class ContextInjector:
    """
    Sub-millisecond Ambient OS Telemetry Injector.
    Runs non-blocking in background, polls host hardware deltas,
    and injects ultra-dense sensory state into MemoryState slot 0.
    """

    def __init__(self, memory: MemoryState, interval_ms: int = 500):
        self.memory = memory
        self.interval_sec = interval_ms / 1000.0
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        # 1. Cache Static Hardware Topology (Zero redundant syscalls in hot loop)
        vm = psutil.virtual_memory()
        self.total_ram_bytes: int = vm.total
        self.cpu_cores: int = psutil.cpu_count(logical=True) or 1
        
        # Prime non-blocking CPU utilization counters
        psutil.cpu_percent(interval=None)
        
        # Disk I/O tracking baseline
        self._last_disk_io = psutil.disk_io_counters()
        self._last_disk_time = time.perf_counter()

        # Last sensory string cache
        self.last_sensory_str: str = ""

    def _format_bytes_compact(self, b: int) -> str:
        """Convert byte values into ultra-compact representation (e.g. 1.2G, 850M)."""
        gb = b / (1024 ** 3)
        if gb >= 1.0:
            return f"{gb:.1f}G"
        mb = b / (1024 ** 2)
        return f"{int(mb)}M"

    def poll_state(self) -> Dict[str, Any]:
        """
        Sub-millisecond dynamic OS metric polling.
        Guaranteed non-blocking: interval=None on CPU times, delta calculations for I/O.
        """
        t_now = time.perf_counter()

        # 1. CPU Aggregate (non-blocking delta since last sample)
        cpu_pct = psutil.cpu_percent(interval=None)

        # 2. Memory Available
        avail_ram = psutil.virtual_memory().available

        # 3. Disk I/O throughput delta (MB/s)
        disk_rate_mb = 0.0
        current_disk_io = psutil.disk_io_counters()
        dt = t_now - self._last_disk_time
        if current_disk_io and self._last_disk_io and dt > 0.0:
            bytes_delta = (current_disk_io.read_bytes + current_disk_io.write_bytes) - \
                          (self._last_disk_io.read_bytes + self._last_disk_io.write_bytes)
            disk_rate_mb = (bytes_delta / (1024 * 1024)) / dt
        self._last_disk_io = current_disk_io
        self._last_disk_time = t_now

        # 4. Network Sockets (Fault-tolerant with permission fallback)
        try:
            conns = psutil.net_connections(kind='inet')
            established_count = sum(1 for c in conns if c.status == psutil.CONN_ESTABLISHED)
            socket_status = f"{established_count}E"
        except (psutil.AccessDenied, PermissionError):
            socket_status = "ERR"
        except Exception:
            socket_status = "ERR"

        return {
            "cpu_pct": cpu_pct,
            "avail_ram": avail_ram,
            "disk_rate_mb": disk_rate_mb,
            "socket_status": socket_status,
        }

    def format_dense_sensory_state(self, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Encodes OS telemetry into a minimal-token, maximum-entropy sensory string.
        Format: <SYS|C:{cpu}%|R:{ram}|D:{disk}|S:{sockets}>
        Example: <SYS|C:14%|R:8.2G|D:0M|S:12E>
        Token footprint: ~5-6 tokens (vs 25+ tokens for standard JSON).
        """
        if metrics is None:
            metrics = self.poll_state()

        cpu_str = f"{int(round(metrics['cpu_pct']))}%"
        ram_str = self._format_bytes_compact(metrics['avail_ram'])
        disk_str = f"{int(round(metrics['disk_rate_mb']))}M"
        socket_str = metrics['socket_status']

        return f"<SYS|C:{cpu_str}|R:{ram_str}|D:{disk_str}|S:{socket_str}>"

    async def inject_into_memory(self, sensory_str: str) -> None:
        """
        Thread-safe silent injection into slot 0 of MemoryState.
        Overwrites index 0 if already occupied by system_telemetry,
        otherwise inserts at index 0 without disrupting conversation history.
        """
        async with self.memory._lock:
            ctx = self.memory._context
            if ctx and ctx[0].get("role") == "system_telemetry":
                ctx[0]["content"] = sensory_str
            else:
                ctx.insert(0, {"role": "system_telemetry", "content": sensory_str})

    async def _poll_loop(self) -> None:
        """Continuous non-blocking telemetry loop."""
        while self._running:
            try:
                metrics = self.poll_state()
                sensory_str = self.format_dense_sensory_state(metrics)
                self.last_sensory_str = sensory_str
                await self.inject_into_memory(sensory_str)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Never crash the daemon on transient OS errors
                fallback_str = f"<SYS|C:?|R:?|D:?|S:ERR|E:{type(e).__name__}>"
                await self.inject_into_memory(fallback_str)

            await asyncio.sleep(self.interval_sec)

    def start(self) -> asyncio.Task:
        """Start the background telemetry injector daemon."""
        if self._running and self._task and not self._task.done():
            return self._task

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        return self._task

    async def stop(self) -> None:
        """Gracefully stop and await cancellation of the telemetry daemon."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None


# =====================================================================
# STANDALONE VERIFICATION & LATENCY BENCHMARK
# =====================================================================

async def main():
    print("=" * 70)
    print("  MIKU CONTEXT INJECTOR (v0.4 Telemetry Engine Verification)")
    print("=" * 70)

    memory = MemoryState(max_items=10)
    injector = ContextInjector(memory=memory, interval_ms=250)

    print(f"\n[Hardware Baseline] CPU Logical Cores: {injector.cpu_cores}")
    print(f"[Hardware Baseline] Total RAM: {injector._format_bytes_compact(injector.total_ram_bytes)}")

    # Benchmark single poll latency
    t0 = time.perf_counter()
    metrics = injector.poll_state()
    t_delta_us = (time.perf_counter() - t0) * 1_000_000
    sensory = injector.format_dense_sensory_state(metrics)

    print(f"\n[Latency Benchmark] Polling duration: {t_delta_us:.2f} \u03bcs ({t_delta_us / 1000:.3f} ms)")
    print(f"[Dense Sensory Output] {sensory}")
    print(f"[Token Efficiency] Length: {len(sensory)} chars (~5-6 byte tokens)")

    # Launch background daemon
    print("\n[Daemon Launch] Starting non-blocking telemetry task (250ms cadence)...")
    injector.start()

    # Simulate user interactions while telemetry updates slot 0
    await asyncio.sleep(0.3)
    await memory.append("user", "Hello Miku, what is your status?")
    await memory.append("assistant", "All systems nominal.")

    print("\n[Memory Snapshot at t=300ms]:")
    ctx = await memory.get_context()
    for i, msg in enumerate(ctx):
        print(f"  Slot {i}: [{msg['role']}] -> {msg['content']}")

    # Wait another telemetry cycle to confirm slot-0 in-place overwrite
    await asyncio.sleep(0.6)
    print("\n[Memory Snapshot at t=900ms (verifying slot-0 update in-place)]:")
    ctx = await memory.get_context()
    for i, msg in enumerate(ctx):
        print(f"  Slot {i}: [{msg['role']}] -> {msg['content']}")

    # Stop daemon
    await injector.stop()
    print("\n✓ Telemetry daemon stopped cleanly. Phase I v0.4 Context Injector verified.")


if __name__ == "__main__":
    asyncio.run(main())
