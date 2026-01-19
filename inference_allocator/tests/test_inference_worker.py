"""Tests for inference worker."""

import asyncio
import pytest

from inference_allocator.services.gpu_manager import GPUManager
from inference_allocator.services.inference_worker import InferenceWorker
from inference_allocator.models.request import InferenceRequest, Priority


class TestInferenceExecution:
    """Test inference execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_response(self):
        """execute() should return a valid response."""
        gpu_manager = GPUManager(gpu_count=2)
        worker = InferenceWorker(gpu_manager, min_ms=10, max_ms=20)

        request = InferenceRequest(
            model_id="test-model",
            prompt="Hello",
            priority=Priority.HIGH,
            request_id="req-123"
        )

        response = await worker.execute(request)

        assert response.request_id == "req-123"
        assert response.model_id == "test-model"
        assert response.gpu_id in [0, 1]
        assert response.inference_time_ms >= 10

    @pytest.mark.asyncio
    async def test_execute_releases_gpu_after_completion(self):
        """GPU should be released after inference completes."""
        gpu_manager = GPUManager(gpu_count=1)
        worker = InferenceWorker(gpu_manager, min_ms=10, max_ms=20)

        request = InferenceRequest(
            model_id="test-model",
            prompt="Hello",
            priority=Priority.HIGH
        )

        await worker.execute(request)

        assert gpu_manager.available_count() == 1

    @pytest.mark.asyncio
    async def test_execute_generates_request_id_if_missing(self):
        """Should generate request_id if not provided."""
        gpu_manager = GPUManager(gpu_count=1)
        worker = InferenceWorker(gpu_manager, min_ms=10, max_ms=20)

        request = InferenceRequest(
            model_id="test-model",
            prompt="Hello",
            priority=Priority.HIGH
        )

        response = await worker.execute(request)

        assert response.request_id is not None
        assert len(response.request_id) > 0


class TestParallelExecution:
    """Test parallel GPU usage."""

    @pytest.mark.asyncio
    async def test_parallel_executions_use_different_gpus(self):
        """Concurrent executions should use different GPUs."""
        gpu_manager = GPUManager(gpu_count=2)
        worker = InferenceWorker(gpu_manager, min_ms=50, max_ms=100)

        request1 = InferenceRequest(model_id="m1", prompt="a", priority=Priority.HIGH)
        request2 = InferenceRequest(model_id="m2", prompt="b", priority=Priority.HIGH)

        responses = await asyncio.gather(
            worker.execute(request1),
            worker.execute(request2)
        )

        used_gpus = {r.gpu_id for r in responses}
        assert len(used_gpus) == 2
