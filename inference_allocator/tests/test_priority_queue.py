"""Tests for priority queue ordering - critical path."""

import asyncio
import pytest

from inference_allocator.services.priority_queue import AsyncPriorityQueue
from inference_allocator.models.request import Priority


class TestPriorityOrdering:
    """Test that priority ordering works correctly."""

    @pytest.mark.asyncio
    async def test_high_priority_dequeued_before_low(self):
        """HIGH priority requests should be processed before LOW."""
        queue = AsyncPriorityQueue(max_size=10)

        # Add LOW first, then HIGH
        await queue.put("low_request", Priority.LOW)
        await queue.put("high_request", Priority.HIGH)

        # HIGH should come out first despite being added second
        first = await queue.get()
        second = await queue.get()

        assert first == "high_request"
        assert second == "low_request"

    @pytest.mark.asyncio
    async def test_priority_order_high_medium_low(self):
        """Requests should be dequeued in order: HIGH > MEDIUM > LOW."""
        queue = AsyncPriorityQueue(max_size=10)

        # Add in reverse priority order
        await queue.put("low", Priority.LOW)
        await queue.put("medium", Priority.MEDIUM)
        await queue.put("high", Priority.HIGH)

        results = []
        for _ in range(3):
            results.append(await queue.get())

        assert results == ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self):
        """Requests with same priority should be FIFO ordered."""
        queue = AsyncPriorityQueue(max_size=10)

        # Add multiple requests with same priority
        await queue.put("first", Priority.MEDIUM)
        await queue.put("second", Priority.MEDIUM)
        await queue.put("third", Priority.MEDIUM)

        results = []
        for _ in range(3):
            results.append(await queue.get())

        assert results == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_mixed_priorities_correct_order(self):
        """Complex scenario with interleaved priorities."""
        queue = AsyncPriorityQueue(max_size=10)

        # Simulate realistic arrival pattern
        await queue.put("med_1", Priority.MEDIUM)
        await queue.put("low_1", Priority.LOW)
        await queue.put("high_1", Priority.HIGH)
        await queue.put("med_2", Priority.MEDIUM)
        await queue.put("high_2", Priority.HIGH)
        await queue.put("low_2", Priority.LOW)

        results = []
        for _ in range(6):
            results.append(await queue.get())

        # Expected: all HIGH first (FIFO), then MEDIUM (FIFO), then LOW (FIFO)
        assert results == ["high_1", "high_2", "med_1", "med_2", "low_1", "low_2"]


class TestQueueBehavior:
    """Test async queue behavior."""

    @pytest.mark.asyncio
    async def test_get_blocks_when_empty(self):
        """get() should block until an item is available."""
        queue = AsyncPriorityQueue(max_size=10)
        result = None

        async def delayed_put():
            await asyncio.sleep(0.05)
            await queue.put("item", Priority.HIGH)

        async def blocking_get():
            nonlocal result
            result = await queue.get()

        # Start both concurrently
        await asyncio.gather(delayed_put(), blocking_get())

        assert result == "item"

    @pytest.mark.asyncio
    async def test_queue_size_tracking(self):
        """Queue should accurately track its size."""
        queue = AsyncPriorityQueue(max_size=10)

        assert queue.size() == 0

        await queue.put("a", Priority.HIGH)
        assert queue.size() == 1

        await queue.put("b", Priority.LOW)
        assert queue.size() == 2

        await queue.get()
        assert queue.size() == 1

        await queue.get()
        assert queue.size() == 0

    @pytest.mark.asyncio
    async def test_queue_rejects_when_full(self):
        """Queue should raise when max size exceeded."""
        queue = AsyncPriorityQueue(max_size=2)

        await queue.put("a", Priority.HIGH)
        await queue.put("b", Priority.HIGH)

        with pytest.raises(Exception):  # QueueFullError
            await queue.put("c", Priority.HIGH)

    @pytest.mark.asyncio
    async def test_empty_check(self):
        """is_empty() should reflect queue state."""
        queue = AsyncPriorityQueue(max_size=10)

        assert queue.is_empty()

        await queue.put("item", Priority.LOW)
        assert not queue.is_empty()

        await queue.get()
        assert queue.is_empty()
