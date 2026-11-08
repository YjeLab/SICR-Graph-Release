from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

TASK_TYPES = (
    TASK_TRANSDUCTIVE,
    TASK_INDUCTIVE,
    TASK_LARGE_SCALE,
    TASK_TRIPLE_CLASSIFICATION,
)
INTERNAL_COARSE_MODES = (
    INTERNAL_COARSE_MODE_FLASH,
    INTERNAL_COARSE_MODE_PRO,
)
INNOVATION2_MODES = (
    INNOVATION2_MODE_OFF,
    INNOVATION2_MODE_AUTO,
    INNOVATION2_MODE_DENSE_PRUNE,
    INNOVATION2_MODE_SPARSE_ROUTE,
)
TRIPLE_CLASSIFICATION_ENCODER_MODES = (
    TRIPLE_CLASSIFICATION_ENCODER_FULL_GRAPH,
    TRIPLE_CLASSIFICATION_ENCODER_LOCAL_SUBGRAPH,
)

STAGE_TRAIN = "train"
STAGE_TEST = "test"
STAGES = (STAGE_TRAIN, STAGE_TEST)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: Path
    companion_relative_path: Path | None = None
    description: str = ""


DATASET_CATALOG: dict[str, dict[str, DatasetSpec]] = {
    TASK_TRANSDUCTIVE: {
        "fb15k-237": DatasetSpec("fb15k-237", Path("transductive") / "fb15k-237"),
        "wn18rr": DatasetSpec("wn18rr", Path("transductive") / "wn18rr"),
        "nell-995": DatasetSpec("nell-995", Path("transductive") / "nell-995"),
        "yago3-10": DatasetSpec("yago3-10", Path("transductive") / "yago3-10"),
        "umls": DatasetSpec("umls", Path("transductive") / "umls"),
        "family": DatasetSpec("family", Path("transductive") / "family"),
    },
    TASK_INDUCTIVE: {
        f"fb15k-237_v{idx}": DatasetSpec(
            f"fb15k-237_v{idx}",
            Path("inductive") / f"fb15k-237_v{idx}",
            Path("inductive") / f"fb15k-237_v{idx}_ind",
        )
        for idx in range(1, 5)
    }
    | {
        f"wn18rr_v{idx}": DatasetSpec(
            f"wn18rr_v{idx}",
            Path("inductive") / f"wn18rr_v{idx}",
            Path("inductive") / f"wn18rr_v{idx}_ind",
        )
        for idx in range(1, 5)
    }
    | {
        f"nell-995_v{idx}": DatasetSpec(
            f"nell-995_v{idx}",
            Path("inductive") / f"nell-995_v{idx}",
            Path("inductive") / f"nell-995_v{idx}_ind",
        )
        for idx in range(1, 5)
    },
    TASK_LARGE_SCALE: {
        "wikidata5m": DatasetSpec("wikidata5m", Path("large_scale") / "wikidata5m"),
        "freebase": DatasetSpec("freebase", Path("large_scale") / "freebase"),
    },
    TASK_TRIPLE_CLASSIFICATION: {
        "umls": DatasetSpec("umls", Path("triple_classification") / "umls"),
        "fb13": DatasetSpec("fb13", Path("triple_classification") / "fb13"),
        "wn11": DatasetSpec("wn11", Path("triple_classification") / "wn11"),
    },
}


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0 if os.name == "nt" else 4
    deterministic: bool = True
    rspmm_backend: str = "native"
    cpu_negative_sampling_for_determinism: bool = False
    amp: bool = False
    compile_model: bool = False
    profile_fusion_paths: bool = False
    profile_fusion_max_forwards: int = 20
    profile_training_step_breakdown: bool = False
    run_name: str | None = None
    output_root: Path = Path("Checkpoints")
    resume_from: str | None = None
    test_checkpoint: str | None = None
    ablation_preset: str = "full"


@dataclass
class DataConfig:
    root: Path = Path("Data")
    batch_size: int = 64
    eval_batch_size: int = 64
    remove_target_edges: bool = True
    remove_all_one_hop_edges: bool = False
    use_test_as_validation: bool = False
    topology_prior_cache: bool = True
    topology_pagerank_damping: float = 0.85
    topology_pagerank_iterations: int = 50
    topology_pagerank_tolerance: float = 1e-6
    local_subgraph_enable: bool = False
    local_subgraph_hops: int = 2
    local_subgraph_topk_1hop: int = 32
    local_subgraph_topk_2hop: int = 8
    local_subgraph_cache: bool = True
    local_subgraph_apply_mask: bool = True


@dataclass
class OptimizerConfig:
    optimizer: str = "Adam"
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    scheduler_milestones: list[int] = field(default_factory=lambda: [10, 15])
    scheduler_gamma: float = 0.1
    grad_clip_norm: float | None = None
    loss_fn: str = "bce"
    num_negative_sample: int = 8
    adversarial_temperature: float = 0.5


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
    internal_coarse_use_joint_loss: bool = True
    internal_coarse_use_delta_decision: bool = True
    internal_coarse_loss_weight: float = 1.0
    internal_coarse_topk: int = 4
    internal_coarse_delta: float = 8.0
    # 普通深度监督对照（意见1）：深层融合中间层挂同质辅助评分头，等权加入训练损失；推理时不参与
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
    hypernym_hard_negative_enable: bool = False
    hypernym_hard_negative_weight: float = 0.1
    hypernym_hard_negative_forward_weight: float | None = None
    hypernym_hard_negative_reverse_weight: float | None = None
    hypernym_hard_negative_count: int = 16
    hypernym_hard_negative_top_pool: int = 128
    hypernym_hard_negative_margin: float = 1.0
    hypernym_hard_negative_feature: str = "derivational_out"
    nell_inverse_domain_prior_enable: bool = False
    nell_inverse_domain_prior_gamma: float = 0.5
    nell_inverse_domain_prior_top_relations: int = 6
    nell_inverse_domain_prior_min_sources: int = 5
    nell_inverse_domain_prior_test_only: bool = True
    triple_classification_encoder_mode: str = TRIPLE_CLASSIFICATION_ENCODER_FULL_GRAPH


@dataclass
class ExperimentConfig:
    model_name: str = "our_duetgraph"
    task_type: str = TASK_TRANSDUCTIVE
    dataset: str = "fb15k-237"
    stage: str = STAGE_TRAIN
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: DuetGraphConfig = field(default_factory=DuetGraphConfig)

    def validate(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        if self.stage not in STAGES:
            raise ValueError(f"Unsupported stage: {self.stage}")
        if self.model.internal_coarse_to_fine_enable and self.model.internal_coarse_mode not in INTERNAL_COARSE_MODES:
            supported = ", ".join(INTERNAL_COARSE_MODES)
            raise ValueError(
                f"Unsupported internal coarse mode: {self.model.internal_coarse_mode}. "
                f"Supported modes: {supported}"
            )
        if self.model.innovation2_mode not in INNOVATION2_MODES:
            supported = ", ".join(INNOVATION2_MODES)
            raise ValueError(
                f"Unsupported innovation2 mode: {self.model.innovation2_mode}. "
                f"Supported modes: {supported}"
            )
        if self.model.innovation2_resolved_mode not in INNOVATION2_MODES:
            supported = ", ".join(INNOVATION2_MODES)
            raise ValueError(
                f"Unsupported resolved innovation2 mode: {self.model.innovation2_resolved_mode}. "
                f"Supported modes: {supported}"
            )
        if self.model.triple_classification_encoder_mode not in TRIPLE_CLASSIFICATION_ENCODER_MODES:
            supported = ", ".join(TRIPLE_CLASSIFICATION_ENCODER_MODES)
            raise ValueError(
                "Unsupported triple classification encoder mode: "
                f"{self.model.triple_classification_encoder_mode}. Supported modes: {supported}"
            )
        if self.model.internal_coarse_topk < 1:
            raise ValueError("internal_coarse_topk must be at least 1.")
        if self.model.internal_coarse_loss_weight < 0:
            raise ValueError("internal_coarse_loss_weight must be non-negative.")
        if self.model.innovation2_routing_floor < 0:
            raise ValueError("innovation2_routing_floor must be non-negative.")
        if self.model.hypernym_path_prior_gamma < 0:
            raise ValueError("hypernym_path_prior_gamma must be non-negative.")
        if self.model.hypernym_path_prior_max_hop < 1:
            raise ValueError("hypernym_path_prior_max_hop must be at least 1.")
        if self.model.hypernym_hub_calibration_feature not in {
            "derivational_out",
            "derivational_in",
            "degree",
        }:
            raise ValueError(
                "hypernym_hub_calibration_feature must be one of: "
                "derivational_out, derivational_in, degree."
            )
        if self.model.innovation4_domain_boost_gamma < 0:
            raise ValueError("innovation4_domain_boost_gamma must be non-negative.")
        if self.model.innovation4_domain_boost_top_relations < 1:
            raise ValueError("innovation4_domain_boost_top_relations must be at least 1.")
        if self.model.innovation4_domain_boost_min_sources < 1:
            raise ValueError("innovation4_domain_boost_min_sources must be at least 1.")
        if self.model.innovation4_hub_suppression_gamma < 0:
            raise ValueError("innovation4_hub_suppression_gamma must be non-negative.")
        if self.model.innovation4_hub_suppression_feature not in {
            "derivational_out",
            "derivational_in",
            "degree",
        }:
            raise ValueError(
                "innovation4_hub_suppression_feature must be one of: "
                "derivational_out, derivational_in, degree."
            )
        if self.model.hypernym_hard_negative_feature not in {
            "derivational_out",
            "derivational_in",
            "degree",
        }:
            raise ValueError(
                "hypernym_hard_negative_feature must be one of: "
                "derivational_out, derivational_in, degree."
            )
        if self.model.hypernym_hard_negative_weight < 0:
            raise ValueError("hypernym_hard_negative_weight must be non-negative.")
        if (
            self.model.hypernym_hard_negative_forward_weight is not None
            and self.model.hypernym_hard_negative_forward_weight < 0
        ):
            raise ValueError("hypernym_hard_negative_forward_weight must be non-negative.")
        if (
            self.model.hypernym_hard_negative_reverse_weight is not None
            and self.model.hypernym_hard_negative_reverse_weight < 0
        ):
            raise ValueError("hypernym_hard_negative_reverse_weight must be non-negative.")
        if self.model.hypernym_hard_negative_count < 1:
            raise ValueError("hypernym_hard_negative_count must be at least 1.")
        if self.model.hypernym_hard_negative_top_pool < 1:
            raise ValueError("hypernym_hard_negative_top_pool must be at least 1.")
        if self.model.hypernym_hard_negative_margin < 0:
            raise ValueError("hypernym_hard_negative_margin must be non-negative.")
        if self.model.nell_inverse_domain_prior_gamma < 0:
            raise ValueError("nell_inverse_domain_prior_gamma must be non-negative.")
        if self.model.nell_inverse_domain_prior_top_relations < 1:
            raise ValueError("nell_inverse_domain_prior_top_relations must be at least 1.")
        if self.model.nell_inverse_domain_prior_min_sources < 1:
            raise ValueError("nell_inverse_domain_prior_min_sources must be at least 1.")
        if self.dataset not in DATASET_CATALOG[self.task_type]:
            supported = ", ".join(sorted(DATASET_CATALOG[self.task_type].keys()))
            raise ValueError(
                f"Dataset '{self.dataset}' is not registered for task '{self.task_type}'. "
                f"Supported datasets: {supported}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DUETGRAPH_PRESETS: dict[tuple[str, str], dict[str, Any]] = {
    (TASK_TRANSDUCTIVE, "fb15k-237"): {
        "batch_size": 96,
        "eval_batch_size": 96,
        "epochs": 13,
        "hidden_dim": 32,
        "loss_fn": "bce",
        "negative_samples": 8,
        "weight_decay": 1e-4,
        "adversarial_temperature": 0.5,
        "remove_target_edges": True,
        "remove_all_one_hop_edges": True,
    },
    (TASK_TRANSDUCTIVE, "wn18rr"): {
        "batch_size": 32,
        "eval_batch_size": 32,
        "epochs": 15,
        "hidden_dim": 32,
        "loss_fn": "ce",
        "weight_decay": 1e-4,
        "remove_target_edges": False,
    },
    (TASK_TRANSDUCTIVE, "nell-995"): {
        "batch_size": 16,
        "eval_batch_size": 16,
        "epochs": 15,
        "hidden_dim": 32,
        "loss_fn": "bce",
        "negative_samples": 16,
        "weight_decay": 1e-4,
        "adversarial_temperature": 0.5,
    },
    (TASK_TRANSDUCTIVE, "yago3-10"): {
        "batch_size": 12,
        "eval_batch_size": 12,
        "epochs": 2,
        "hidden_dim": 32,
        "loss_fn": "bce",
        "negative_samples": 16,
        "weight_decay": 1e-4,
        "adversarial_temperature": 1.0,
    },
}

for version in range(1, 5):
    DUETGRAPH_PRESETS[(TASK_INDUCTIVE, f"fb15k-237_v{version}")] = {
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs": 20,
        "hidden_dim": 64,
        "loss_fn": "bce",
        "negative_samples": 6,
        "weight_decay": 1e-4,
        "adversarial_temperature": 0.5,
    }

for version in range(1, 5):
    DUETGRAPH_PRESETS[(TASK_INDUCTIVE, f"wn18rr_v{version}")] = {
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs": 20,
        "hidden_dim": 32,
        "loss_fn": "bce" if version <= 2 else "ce",
        "negative_samples": 8,
        "weight_decay": 1e-4,
        "adversarial_temperature": 0.5,
    }

NELL_PRESET_OVERRIDES = {
    1: {"loss_fn": "bce", "negative_samples": 16, "adversarial_temperature": 1.0, "weight_decay": 1e-5},
    2: {"loss_fn": "ce", "negative_samples": 16, "adversarial_temperature": 1.0, "weight_decay": 1e-5},
    3: {"loss_fn": "bce", "negative_samples": 16, "adversarial_temperature": 1.0, "weight_decay": 1e-5},
    4: {"loss_fn": "bce", "negative_samples": 16, "adversarial_temperature": 1.0, "weight_decay": 1e-5},
}
for version in range(1, 5):
    DUETGRAPH_PRESETS[(TASK_INDUCTIVE, f"nell-995_v{version}")] = {
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs": 20,
        "hidden_dim": 64,
        **NELL_PRESET_OVERRIDES[version],
    }

DUETGRAPH_PRESETS[(TASK_TRIPLE_CLASSIFICATION, "umls")] = {
    "batch_size": 64,
    "eval_batch_size": 128,
    "epochs": 20,
    "hidden_dim": 32,
    "loss_fn": "bce",
    "weight_decay": 1e-4,
    "remove_target_edges": False,
}
DUETGRAPH_PRESETS[(TASK_TRIPLE_CLASSIFICATION, "fb13")] = {
    "batch_size": 32,
    "eval_batch_size": 64,
    "epochs": 20,
    "hidden_dim": 32,
    "loss_fn": "bce",
    "weight_decay": 1e-4,
    "remove_target_edges": False,
}
DUETGRAPH_PRESETS[(TASK_TRIPLE_CLASSIFICATION, "wn11")] = {
    "batch_size": 64,
    "eval_batch_size": 128,
    "epochs": 20,
    "hidden_dim": 32,
    "loss_fn": "bce",
    "weight_decay": 1e-4,
    "remove_target_edges": False,
}


def resolve_dataset_spec(task_type: str, dataset_name: str) -> DatasetSpec:
    if task_type not in DATASET_CATALOG:
        raise ValueError(f"Unsupported task type: {task_type}")
    if dataset_name not in DATASET_CATALOG[task_type]:
        supported = ", ".join(sorted(DATASET_CATALOG[task_type].keys()))
        raise ValueError(
            f"Dataset '{dataset_name}' is not registered for task '{task_type}'. "
            f"Supported datasets: {supported}"
        )
    return DATASET_CATALOG[task_type][dataset_name]


def list_supported_datasets(task_type: str) -> list[str]:
    if task_type not in DATASET_CATALOG:
        return []
    return sorted(DATASET_CATALOG[task_type].keys())


def apply_recommended_preset(config: ExperimentConfig) -> None:
    preset = DUETGRAPH_PRESETS.get((config.task_type, config.dataset))
    if config.model_name not in {"duetgraph", "our_duetgraph"} or preset is None:
        return

    config.data.batch_size = preset.get("batch_size", config.data.batch_size)
    config.data.eval_batch_size = preset.get("eval_batch_size", config.data.eval_batch_size)
    config.data.remove_target_edges = preset.get(
        "remove_target_edges",
        config.data.remove_target_edges,
    )
    config.data.remove_all_one_hop_edges = preset.get(
        "remove_all_one_hop_edges",
        config.data.remove_all_one_hop_edges,
    )
    config.optimizer.epochs = preset.get("epochs", config.optimizer.epochs)
    config.optimizer.loss_fn = preset.get("loss_fn", config.optimizer.loss_fn)
    config.optimizer.num_negative_sample = preset.get(
        "negative_samples",
        config.optimizer.num_negative_sample,
    )
    config.optimizer.weight_decay = preset.get("weight_decay", config.optimizer.weight_decay)
    config.optimizer.adversarial_temperature = preset.get(
        "adversarial_temperature",
        config.optimizer.adversarial_temperature,
    )
    config.model.hidden_dim = preset.get("hidden_dim", config.model.hidden_dim)
