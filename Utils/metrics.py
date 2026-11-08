from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RankingMetrics:
    mr: float
    mrr: float
    hits1: float
    hits3: float
    hits10: float
    per_relation: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | dict[str, dict[str, float]]]:
        metrics: dict[str, float | dict[str, dict[str, float]]] = {
            "mr": self.mr,
            "mrr": self.mrr,
            "hits1": self.hits1,
            "hits3": self.hits3,
            "hits10": self.hits10,
        }
        if self.per_relation:
            metrics["per_relation"] = self.per_relation
        return metrics


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    mrr: float = 0.0
    hits1: float = 0.0
    hits3: float = 0.0
    hits10: float = 0.0
    mr: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mrr": self.mrr,
            "hits1": self.hits1,
            "hits3": self.hits3,
            "hits10": self.hits10,
            "mr": self.mr,
        }


class RankingMetricAccumulator:
    def __init__(self) -> None:
        self.rank_sum = 0.0
        self.reciprocal_rank_sum = 0.0
        self.hits1_sum = 0.0
        self.hits3_sum = 0.0
        self.hits10_sum = 0.0
        self.total = 0

    def update(self, ranks: torch.Tensor) -> None:
        ranks = ranks.detach().float()
        self.rank_sum += ranks.sum().item()
        self.reciprocal_rank_sum += (1.0 / ranks).sum().item()
        self.hits1_sum += (ranks <= 1).sum().item()
        self.hits3_sum += (ranks <= 3).sum().item()
        self.hits10_sum += (ranks <= 10).sum().item()
        self.total += ranks.numel()

    def compute(self) -> RankingMetrics:
        if self.total == 0:
            return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        total = float(self.total)
        return RankingMetrics(
            mr=self.rank_sum / total,
            mrr=self.reciprocal_rank_sum / total,
            hits1=self.hits1_sum / total,
            hits3=self.hits3_sum / total,
            hits10=self.hits10_sum / total,
        )
