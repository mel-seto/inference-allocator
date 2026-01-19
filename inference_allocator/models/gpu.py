from enum import Enum
from dataclasses import dataclass


class GPUState(Enum):
    """GPU availability state."""

    AVAILABLE = "available"
    BUSY = "busy"


@dataclass
class GPUSlot:
    """Represents a single GPU slot."""

    gpu_id: int
    state: GPUState = GPUState.AVAILABLE
