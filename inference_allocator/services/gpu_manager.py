import asyncio
from typing import Dict

from inference_allocator.models.gpu import GPUSlot, GPUState


class GPUManager:
    """Manages a pool of simulated GPU slots.

    Provides async acquire/release with condition-based waiting
    when no GPUs are available.
    """

    def __init__(self, gpu_count: int):
        self._gpus: Dict[int, GPUSlot] = {
            i: GPUSlot(gpu_id=i, state=GPUState.AVAILABLE)
            for i in range(gpu_count)
        }
        self._lock = asyncio.Lock()
        self._available = asyncio.Condition(self._lock)

    async def acquire_gpu(self) -> GPUSlot:
        """Acquire an available GPU. Blocks until one is free."""
        async with self._available:
            while True:
                # Find first available GPU
                for gpu in self._gpus.values():
                    if gpu.state == GPUState.AVAILABLE:
                        gpu.state = GPUState.BUSY
                        return gpu

                # No GPU available, wait
                await self._available.wait()

    async def release_gpu(self, gpu_id: int) -> None:
        """Release a GPU back to the pool."""
        async with self._available:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].state = GPUState.AVAILABLE
                self._available.notify()

    def get_gpu_state(self, gpu_id: int) -> GPUState:
        """Get state of a specific GPU."""
        return self._gpus[gpu_id].state

    def get_all_states(self) -> Dict[int, GPUState]:
        """Get states of all GPUs."""
        return {gpu_id: gpu.state for gpu_id, gpu in self._gpus.items()}

    def available_count(self) -> int:
        """Count available GPUs."""
        return sum(1 for gpu in self._gpus.values() if gpu.state == GPUState.AVAILABLE)

    def busy_count(self) -> int:
        """Count busy GPUs."""
        return sum(1 for gpu in self._gpus.values() if gpu.state == GPUState.BUSY)
