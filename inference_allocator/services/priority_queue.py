import asyncio
import heapq
from typing import Any, Generic, TypeVar

from inference_allocator.models.request import Priority

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised when queue is at max capacity."""

    pass


class AsyncPriorityQueue(Generic[T]):
    """Async priority queue using heapq.

    Items are dequeued by priority (lower value = higher priority),
    with FIFO ordering within the same priority level.
    """

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._heap: list[tuple[int, int, T]] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

    async def put(self, item: T, priority: Priority) -> None:
        """Add item to queue with given priority.

        Raises QueueFullError if queue is at capacity.
        """
        async with self._lock:
            if len(self._heap) >= self._max_size:
                raise QueueFullError(f"Queue full (max_size={self._max_size})")

            entry = (int(priority), self._counter, item)
            self._counter += 1
            heapq.heappush(self._heap, entry)
            self._not_empty.notify()

    async def get(self) -> T:
        """Remove and return highest priority item.

        Blocks until an item is available.
        """
        async with self._not_empty:
            while len(self._heap) == 0:
                await self._not_empty.wait()

            _, _, item = heapq.heappop(self._heap)
            return item

    def size(self) -> int:
        """Return current queue size."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Return True if queue is empty."""
        return len(self._heap) == 0
