"""Model-side shared types and helpers for SICR-Graph.

This module contains only the definitions needed to build and run the model:
configuration dataclass, input types, base class, a tiny registry and the
fusion-path profiler. No data loading, training or evaluation code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Task and mode constants
# --------------------------------------------------------------------------- #
TASK_TRANSDUCTIVE = "transductive_reasoning"
TASK_INDUCTIVE = "inductive_reasoning"
TASK_LARGE_SCALE = "large_scale_kg"
TASK_TRIPLE_CLASSIFICATION = "triple_classification"

INTERNAL_COARSE_MODE_FLASH = "flash"
INTERNAL_COARSE_MODE_PRO = "pro"

INNOVATION2_MODE_OFF = "off"
INNOVATION2_MODE_AUTO = "auto"
INNOVATION2_MODE_DENSE_PRUNE = "dense_prune"
INNOVATION2_MODE_SPARSE_ROUTE = "sparse_route"

TRIPLE_CLASSIFICATION_ENCODER_FULL_GRAPH = "full_graph"
TRIPLE_CLASSIFICATION_ENCODER_LOCAL_SUBGRAPH = "local_subgraph"


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
@dataclass
class DuetGraphConfig:
    hidden_dim: int = 64
    num_layer: int = 3
    num_qk_layer: int = 2
    num_v_layer: int = 3
    num_heads: int = 4
    drop: float = 0.1

    use_innovation1: bool = True
    use_innovation2: bool = False
    innovation2_mode: str = INNOVATION2_MODE_AUTO
    innovation2_resolved_mode: str = INNOVATION2_MODE_OFF

    internal_coarse_to_fine_enable: bool = False
    internal_coarse_mode: str = INTERNAL_COARSE_MODE_FLASH
    internal_coarse_use_delta_decision: bool = True
    internal_coarse_topk: int = 4
    internal_coarse_delta: float = 8.0

    deep_supervision_enable: bool = False

    topology_prior_dim: int = 3
    topology_prior_hidden_dim: int = 32
    topology_gate_hidden_dim: int = 64
    topology_gate_cap: float = 0.95
    topology_gate_use_prior: bool = False

    innovation1_use_degree_bias: bool = False
    innovation1_degree_bias_init: float = 0.5

    innovation2_routing_hidden_dim: int = 32
    innovation2_routing_floor: float = 0.25
    innovation2_routing_bias_init: float = 2.0
    innovation2_routing_match_bias: float = 1.0
    innovation2_auto_avg_degree_threshold: float = 3.2
    innovation2_auto_low_degree_ratio_threshold: float = 0.65
    innovation2_auto_low_degree_cutoff: int = 2
    innovation2_auto_avg_degree: float = 0.0
    innovation2_auto_low_degree_ratio: float = 0.0
    innovation2_auto_median_degree: float = 0.0

    hypernym_path_prior_enable: bool = False
    hypernym_path_prior_gamma: float = 0.5
    hypernym_path_prior_max_hop: int = 3

    hypernym_hub_calibration_enable: bool = False
    hypernym_hub_calibration_gamma: float = -1.0
    hypernym_hub_calibration_feature: str = "derivational_out"
    hypernym_hub_calibration_apply_coarse: bool = False

    innovation4_tail_calibration_enable: bool = False
    innovation4_tail_calibration_test_only: bool = True
    innovation4_domain_boost_enable: bool = False
    innovation4_domain_boost_gamma: float = 0.5
    innovation4_domain_boost_top_relations: int = 6
    innovation4_domain_boost_min_sources: int = 5
    innovation4_hub_suppression_enable: bool = False
    innovation4_hub_suppression_gamma: float = 1.0
    innovation4_hub_suppression_feature: str = "derivational_out"

    triple_classification_encoder_mode: str = TRIPLE_CLASSIFICATION_ENCODER_FULL_GRAPH


# --------------------------------------------------------------------------- #
# Model input types
# --------------------------------------------------------------------------- #
@dataclass
class Graph:
    edge_index: torch.Tensor
    num_nodes: int
    num_relations: int
    id2relation: dict[int, str] | None = None
    node_ids: torch.Tensor | None = None
    node_prior: torch.Tensor | None = None
    local_subgraph_edge_pool: list[torch.Tensor] | None = None

    def to(self, device: torch.device) -> "Graph":
        self.edge_index = self.edge_index.to(device)
        if self.node_ids is not None:
            self.node_ids = self.node_ids.to(device)
        if self.node_prior is not None:
            self.node_prior = self.node_prior.to(device)
        return self


@dataclass
class DatasetInfo:
    name: str
    task_type: str
    num_relations: int
    train_num_entities: int
    eval_num_entities: int
    use_entity_embeddings: bool


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseReasoningModel(nn.Module, ABC):
    model_name = "base"
    supported_tasks: set[str] = set()

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def ensure_supports(self, task_type: str) -> None:
        if task_type not in self.supported_tasks:
            supported = ", ".join(sorted(self.supported_tasks))
            raise ValueError(
                f"Model '{self.model_name}' does not support task '{task_type}'. "
                f"Supported tasks: {supported}"
            )


# --------------------------------------------------------------------------- #
# Fusion-path profiler
# --------------------------------------------------------------------------- #
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

    def record_invocation(self, local_ms: float, global_ms: float, gate_ms: float = 0.0) -> None:
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


# --------------------------------------------------------------------------- #
# Minimal model registry
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name](*args, **kwargs)


def list_registered_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())