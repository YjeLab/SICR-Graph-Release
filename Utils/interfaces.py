from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


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


class BaseTask(ABC):
    task_name = "base_task"
    primary_metric = "mrr"

    @abstractmethod
    def build_data_module(self, config, dataset_spec):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, model, batch, config, device):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model, dataloader, device):
        raise NotImplementedError
