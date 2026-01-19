import asyncio
from typing import Any, Dict, Tuple

from inference_allocator.models.request import InferenceRequest, InferenceResponse
from inference_allocator.services.priority_queue import AsyncPriorityQueue
from inference_allocator.services.gpu_manager import GPUManager
from inference_allocator.services.inference_worker import InferenceWorker


class Orchestrator:
    """Coordinates request queuing, GPU allocation, and inference execution."""

    def __init__(
        self,
        gpu_count: int,
        queue_max_size: int,
        min_ms: int = 100,
        max_ms: int = 500
    ):
        self._queue: AsyncPriorityQueue[Tuple[InferenceRequest, asyncio.Future]] = (
            AsyncPriorityQueue(max_size=queue_max_size)
        )
        self._gpu_manager = GPUManager(gpu_count=gpu_count)
        self._worker = InferenceWorker(self._gpu_manager, min_ms=min_ms, max_ms=max_ms)
        self._running = False
        self._process_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background processing loop."""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop the background processing loop."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit request and wait for response."""
        future = await self._enqueue(request)
        return await future

    async def _enqueue(self, request: InferenceRequest) -> asyncio.Future:
        """Add request to queue, return Future for result."""
        future: asyncio.Future[InferenceResponse] = asyncio.Future()
        await self._queue.put((request, future), request.priority)
        return future

    async def _process_loop(self) -> None:
        """Background loop: dequeue requests and spawn execution tasks."""
        while self._running:
            try:
                request, future = await self._queue.get()
                # Spawn task for parallel execution
                asyncio.create_task(self._execute_and_resolve(request, future))
            except asyncio.CancelledError:
                break

    async def _execute_and_resolve(
        self,
        request: InferenceRequest,
        future: asyncio.Future
    ) -> None:
        """Execute request and resolve its future."""
        try:
            response = await self._worker.execute(request)
            future.set_result(response)
        except Exception as e:
            future.set_exception(e)

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "queue_size": self._queue.size(),
            "gpus_available": self._gpu_manager.available_count(),
            "gpus_busy": self._gpu_manager.busy_count(),
        }
