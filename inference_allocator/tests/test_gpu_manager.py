"""Tests for GPU allocation logic - critical path."""

import asyncio
import pytest

from inference_allocator.services.gpu_manager import GPUManager
from inference_allocator.models.gpu import GPUState


class TestGPUAcquisition:
    """Test GPU acquisition logic."""

    @pytest.mark.asyncio
    async def test_acquire_returns_available_gpu(self):
        """acquire_gpu() should return an available GPU."""
        manager = GPUManager(gpu_count=2)

        gpu = await manager.acquire_gpu()

        assert gpu is not None
        assert gpu.state == GPUState.BUSY

    @pytest.mark.asyncio
    async def test_acquire_marks_gpu_as_busy(self):
        """Acquired GPU should be marked BUSY."""
        manager = GPUManager(gpu_count=1)

        gpu = await manager.acquire_gpu()

        assert manager.get_gpu_state(gpu.gpu_id) == GPUState.BUSY

    @pytest.mark.asyncio
    async def test_cannot_acquire_same_gpu_twice(self):
        """A busy GPU should not be returned for acquisition."""
        manager = GPUManager(gpu_count=1)

        gpu1 = await manager.acquire_gpu()

        # Second acquire should block (we test with timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(manager.acquire_gpu(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_acquire_multiple_gpus(self):
        """Should be able to acquire all available GPUs."""
        manager = GPUManager(gpu_count=3)

        gpu1 = await manager.acquire_gpu()
        gpu2 = await manager.acquire_gpu()
        gpu3 = await manager.acquire_gpu()

        # All GPUs should be different
        gpu_ids = {gpu1.gpu_id, gpu2.gpu_id, gpu3.gpu_id}
        assert len(gpu_ids) == 3


class TestGPURelease:
    """Test GPU release logic."""

    @pytest.mark.asyncio
    async def test_release_makes_gpu_available(self):
        """Released GPU should become AVAILABLE."""
        manager = GPUManager(gpu_count=1)

        gpu = await manager.acquire_gpu()
        await manager.release_gpu(gpu.gpu_id)

        assert manager.get_gpu_state(gpu.gpu_id) == GPUState.AVAILABLE

    @pytest.mark.asyncio
    async def test_release_allows_reacquisition(self):
        """Released GPU can be acquired again."""
        manager = GPUManager(gpu_count=1)

        gpu1 = await manager.acquire_gpu()
        await manager.release_gpu(gpu1.gpu_id)

        gpu2 = await manager.acquire_gpu()
        assert gpu2.gpu_id == gpu1.gpu_id

    @pytest.mark.asyncio
    async def test_release_unblocks_waiting_acquire(self):
        """Releasing a GPU should unblock a waiting acquire."""
        manager = GPUManager(gpu_count=1)
        acquired_gpu_id = None

        # Acquire the only GPU
        gpu = await manager.acquire_gpu()

        async def delayed_release():
            await asyncio.sleep(0.05)
            await manager.release_gpu(gpu.gpu_id)

        async def blocking_acquire():
            nonlocal acquired_gpu_id
            new_gpu = await manager.acquire_gpu()
            acquired_gpu_id = new_gpu.gpu_id

        # Both should complete - release unblocks the acquire
        await asyncio.gather(delayed_release(), blocking_acquire())

        assert acquired_gpu_id == gpu.gpu_id


class TestGPUConcurrency:
    """Test concurrent GPU operations."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_get_different_gpus(self):
        """Concurrent acquires should not get the same GPU."""
        manager = GPUManager(gpu_count=4)
        acquired_ids = []
        lock = asyncio.Lock()

        async def acquire_and_record():
            gpu = await manager.acquire_gpu()
            async with lock:
                acquired_ids.append(gpu.gpu_id)

        # Run 4 concurrent acquires
        await asyncio.gather(*[acquire_and_record() for _ in range(4)])

        # All should be unique
        assert len(set(acquired_ids)) == 4

    @pytest.mark.asyncio
    async def test_fairness_fifo_waiting(self):
        """Waiters should be served in FIFO order."""
        manager = GPUManager(gpu_count=1)
        order = []

        # Acquire the only GPU
        gpu = await manager.acquire_gpu()

        async def waiter(name: str, delay: float):
            await asyncio.sleep(delay)  # Stagger the waits
            await manager.acquire_gpu()
            order.append(name)
            # Release immediately for next waiter
            await manager.release_gpu(0)

        # Start waiters with small delays to ensure ordering
        tasks = [
            asyncio.create_task(waiter("first", 0.01)),
            asyncio.create_task(waiter("second", 0.02)),
            asyncio.create_task(waiter("third", 0.03)),
        ]

        # Give waiters time to start waiting
        await asyncio.sleep(0.05)

        # Release GPU to let waiters proceed
        await manager.release_gpu(gpu.gpu_id)

        await asyncio.gather(*tasks)

        assert order == ["first", "second", "third"]


class TestGPUStatus:
    """Test GPU status reporting."""

    @pytest.mark.asyncio
    async def test_get_all_states(self):
        """Should report state of all GPUs."""
        manager = GPUManager(gpu_count=3)

        states = manager.get_all_states()

        assert len(states) == 3
        assert all(s == GPUState.AVAILABLE for s in states.values())

    @pytest.mark.asyncio
    async def test_available_count(self):
        """Should correctly count available GPUs."""
        manager = GPUManager(gpu_count=3)

        assert manager.available_count() == 3

        await manager.acquire_gpu()
        assert manager.available_count() == 2

        await manager.acquire_gpu()
        assert manager.available_count() == 1

    @pytest.mark.asyncio
    async def test_busy_count(self):
        """Should correctly count busy GPUs."""
        manager = GPUManager(gpu_count=3)

        assert manager.busy_count() == 0

        gpu1 = await manager.acquire_gpu()
        assert manager.busy_count() == 1

        gpu2 = await manager.acquire_gpu()
        assert manager.busy_count() == 2

        await manager.release_gpu(gpu1.gpu_id)
        assert manager.busy_count() == 1
