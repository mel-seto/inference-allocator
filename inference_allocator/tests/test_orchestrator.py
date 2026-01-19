"""Tests for orchestrator."""

import asyncio
import pytest

from inference_allocator.services.orchestrator import Orchestrator
from inference_allocator.models.request import InferenceRequest, Priority


class TestRequestSubmission:
    """Test request submission and response."""

    @pytest.mark.asyncio
    async def test_submit_returns_response(self):
        """submit() should return inference response."""
        orchestrator = Orchestrator(gpu_count=2, queue_max_size=10, min_ms=10, max_ms=20)
        await orchestrator.start()

        try:
            request = InferenceRequest(
                model_id="test-model",
                prompt="Hello",
                priority=Priority.HIGH,
                request_id="req-123"
            )

            response = await orchestrator.submit(request)

            assert response.request_id == "req-123"
            assert response.model_id == "test-model"
        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_submit_multiple_requests(self):
        """Should handle multiple sequential requests."""
        orchestrator = Orchestrator(gpu_count=2, queue_max_size=10, min_ms=10, max_ms=20)
        await orchestrator.start()

        try:
            for i in range(3):
                request = InferenceRequest(
                    model_id="model",
                    prompt=f"prompt-{i}",
                    priority=Priority.MEDIUM,
                    request_id=f"req-{i}"
                )
                response = await orchestrator.submit(request)
                assert response.request_id == f"req-{i}"
        finally:
            await orchestrator.stop()


class TestParallelProcessing:
    """Test parallel GPU utilization."""

    @pytest.mark.asyncio
    async def test_parallel_requests_use_multiple_gpus(self):
        """Concurrent requests should use multiple GPUs."""
        orchestrator = Orchestrator(gpu_count=2, queue_max_size=10, min_ms=50, max_ms=100)
        await orchestrator.start()

        try:
            requests = [
                InferenceRequest(model_id="m", prompt="a", priority=Priority.HIGH),
                InferenceRequest(model_id="m", prompt="b", priority=Priority.HIGH),
            ]

            responses = await asyncio.gather(*[orchestrator.submit(r) for r in requests])
            used_gpus = {r.gpu_id for r in responses}

            assert len(used_gpus) == 2
        finally:
            await orchestrator.stop()


class TestStatus:
    """Test status reporting."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should report queue and GPU status."""
        orchestrator = Orchestrator(gpu_count=2, queue_max_size=10, min_ms=10, max_ms=20)
        await orchestrator.start()

        try:
            status = orchestrator.get_status()

            assert "queue_size" in status
            assert "gpus_available" in status
            assert "gpus_busy" in status
            assert status["gpus_available"] == 2
            assert status["gpus_busy"] == 0
        finally:
            await orchestrator.stop()
