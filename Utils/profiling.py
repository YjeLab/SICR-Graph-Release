from __future__ import annotations

from dataclasses import dataclass

import torch


def synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class FusionProfileSummary:
    forward_passes: int
    layer_invocations: int
    local_ms: float
    global_ms: float
    gate_ms: float

    @property
    def total_ms(self) -> float:
        return self.local_ms + self.global_ms + self.gate_ms


class FusionPathProfiler:
    def __init__(self) -> None:
        self.enabled = False
        self.max_forwards = 0
        self.reset()

    def configure(self, enabled: bool, max_forwards: int) -> None:
        self.enabled = enabled
        self.max_forwards = max(1, int(max_forwards)) if enabled else 0
        self.reset()

    def reset(self) -> None:
        self.remaining_forwards = self.max_forwards
        self.forward_passes = 0
        self.layer_invocations = 0
        self.local_ms = 0.0
        self.global_ms = 0.0
        self.gate_ms = 0.0

    def begin_forward(self) -> bool:
        if not self.enabled or self.remaining_forwards <= 0:
            return False
        self.remaining_forwards -= 1
        self.forward_passes += 1
        return True

    def record_invocation(
        self,
        local_ms: float,
        global_ms: float,
        gate_ms: float = 0.0,
    ) -> None:
        self.layer_invocations += 1
        self.local_ms += float(local_ms)
        self.global_ms += float(global_ms)
        self.gate_ms += float(gate_ms)

    def summary(self) -> FusionProfileSummary | None:
        if self.layer_invocations == 0:
            return None
        return FusionProfileSummary(
            forward_passes=self.forward_passes,
            layer_invocations=self.layer_invocations,
            local_ms=self.local_ms,
            global_ms=self.global_ms,
            gate_ms=self.gate_ms,
        )
