import asyncio
import random
import time
import uuid

from inference_allocator.services.gpu_manager import GPUManager
from inference_allocator.models.request import InferenceRequest, InferenceResponse


class InferenceWorker:
    """Executes inference requests on GPUs."""

    def __init__(self, gpu_manager: GPUManager, min_ms: int = 100, max_ms: int = 500):
        self._gpu_manager = gpu_manager
        self._min_ms = min_ms
        self._max_ms = max_ms

    async def execute(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference request.

        Acquires GPU, runs mock inference, releases GPU.
        """
        gpu = await self._gpu_manager.acquire_gpu()

        try:
            start_time = time.perf_counter()

            # Mock inference delay
            delay_ms = random.randint(self._min_ms, self._max_ms)
            await asyncio.sleep(delay_ms / 1000)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            request_id = request.request_id or str(uuid.uuid4())

            return InferenceResponse(
                request_id=request_id,
                model_id=request.model_id,
                output=f"Mock output for: {request.prompt[:50]}",
                gpu_id=gpu.gpu_id,
                inference_time_ms=elapsed_ms
            )
        finally:
            await self._gpu_manager.release_gpu(gpu.gpu_id)
