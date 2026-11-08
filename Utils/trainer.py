from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from tqdm.auto import tqdm

import Our  # noqa: F401
import Utils.tasks  # noqa: F401
from Our.rspmm import get_rspmm_backend_info
from Utils.config import (
    INNOVATION2_MODE_AUTO,
    INNOVATION2_MODE_DENSE_PRUNE,
    INNOVATION2_MODE_OFF,
    INNOVATION2_MODE_SPARSE_ROUTE,
    STAGE_TEST,
    STAGE_TRAIN,
    resolve_dataset_spec,
)
from Utils.io import create_logger, prepare_run_artifacts, save_json
from Utils.registry import build_model, build_task


class ExperimentRunner:
    def __init__(self, config) -> None:
        self.config = config
        self.config.validate()
        os.environ["C3R_GRAPH_RSPMM_BACKEND"] = self.config.runtime.rspmm_backend
        self._sync_config_from_checkpoint()
        self.dataset_spec = resolve_dataset_spec(config.task_type, config.dataset)
        self._resolve_innovation2_mode()
        self.task = build_task(config.task_type)
        self.data_module = self.task.build_data_module(config, self.dataset_spec)
        self.artifacts = prepare_run_artifacts(config)
        self.logger = create_logger(self.artifacts.log_file)

        self._set_seed()
        if config.runtime.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("未检测到可用的 CUDA，自动切换到 CPU。")
            config.runtime.device = "cpu"
        self.device = torch.device(config.runtime.device)
        self.model = build_model(config.model_name, self.data_module.dataset_info, config.model)
        self.model.ensure_supports(config.task_type)
        self.model.to(self.device)

        if config.runtime.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        if (
            self.config.runtime.profile_fusion_paths
            and hasattr(self.model, "configure_fusion_profiling")
        ):
            self.model.configure_fusion_profiling(
                enabled=True,
                max_forwards=self.config.runtime.profile_fusion_max_forwards,
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_grad_scaler()

        save_json(self.artifacts.config_file, self.config.to_dict())
        self.logger.info("运行目录：%s", self.artifacts.result_dir)
        self.logger.info("数据集路径：%s", self.config.data.root / self.dataset_spec.relative_path)
        if self.config.data.use_test_as_validation:
            self.logger.info("选模口径：使用测试集充当验证集（valid 合入训练集）")
        else:
            self.logger.info("选模口径：使用标准验证集选模")
        if self.device.type == "cuda":
            self.logger.info("训练设备：CUDA")
        else:
            self.logger.info("训练设备：CPU")
        self.logger.info(
            "复现设置：deterministic=%s | rspmm后端=%s | 负采样后端=%s",
            "开启" if self.config.runtime.deterministic else "关闭",
            self.config.runtime.rspmm_backend,
            (
                "CPU（严格复现，较慢）"
                if self.config.runtime.cpu_negative_sampling_for_determinism
                else "GPU（默认，更快）"
            ),
        )
        rspmm_backend, rspmm_reason = get_rspmm_backend_info()
        if rspmm_reason is None:
            self.logger.info("算子后端：rspmm_backend=%s", rspmm_backend)
        else:
            self.logger.info(
                "算子后端：rspmm_backend=%s | reason=%s",
                rspmm_backend,
                rspmm_reason,
            )
        if self.config.runtime.profile_fusion_paths:
            self.logger.info(
                "融合通路计时：开启 | 每个阶段最多采样 %d 次前向",
                self.config.runtime.profile_fusion_max_forwards,
            )
        if self.config.runtime.profile_training_step_breakdown:
            self.logger.info("训练分阶段计时：开启 | 将统计 fetch / forward / backward / step 的平均耗时")
        self._log_experiment_configuration()
        self._log_data_pipeline_diagnostics()
        self._log_parameter_statistics()
        self._prepare_runtime_environment()

        if self.config.runtime.resume_from:
            self._load_checkpoint(self.config.runtime.resume_from)

    @staticmethod
    def _inspect_graph_structure(data_path, low_degree_cutoff: int) -> dict[str, float] | None:
        entities_path = data_path / "entities.txt"
        train_path = data_path / "train.txt"
        if not entities_path.exists() or not train_path.exists():
            return None

        entity_to_id: dict[str, int] = {}
        with entities_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                entity, index = stripped.split("\t")
                entity_to_id[entity] = int(index)

        if not entity_to_id:
            return None

        degree = [0] * len(entity_to_id)
        edge_count = 0
        with train_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                head, _, tail = stripped.split("\t")
                degree[entity_to_id[head]] += 1
                degree[entity_to_id[tail]] += 1
                edge_count += 1

        sorted_degree = sorted(degree)
        midpoint = len(sorted_degree) // 2
        if len(sorted_degree) % 2 == 0:
            median_degree = 0.5 * (sorted_degree[midpoint - 1] + sorted_degree[midpoint])
        else:
            median_degree = float(sorted_degree[midpoint])

        node_count = len(degree)
        return {
            "node_count": float(node_count),
            "raw_edge_count": float(edge_count),
            "avg_degree": float(sum(degree) / node_count),
            "median_degree": float(median_degree),
            "low_degree_ratio": float(sum(item <= low_degree_cutoff for item in degree) / node_count),
        }

    @staticmethod
    def _combine_graph_structure_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
        return {
            "node_count": sum(item["node_count"] for item in stats_list),
            "raw_edge_count": sum(item["raw_edge_count"] for item in stats_list),
            "avg_degree": min(item["avg_degree"] for item in stats_list),
            "median_degree": min(item["median_degree"] for item in stats_list),
            "low_degree_ratio": max(item["low_degree_ratio"] for item in stats_list),
        }

    @staticmethod
    def _rewrite_innovation2_preset(preset: str, mode: str, resolved_mode: str) -> str:
        parts = preset.split("|")
        rewritten = []
        for part in parts:
            if part.startswith("I2="):
                rewritten.append(f"I2={mode}->{resolved_mode}" if mode == INNOVATION2_MODE_AUTO else f"I2={resolved_mode}")
            else:
                rewritten.append(part)
        return "|".join(rewritten)

    def _resolve_innovation2_mode(self) -> None:
        model_cfg = self.config.model
        if not getattr(model_cfg, "use_innovation2", False):
            model_cfg.innovation2_resolved_mode = INNOVATION2_MODE_OFF
            return

        requested_mode = model_cfg.innovation2_mode
        if requested_mode not in {
            INNOVATION2_MODE_OFF,
            INNOVATION2_MODE_AUTO,
            INNOVATION2_MODE_DENSE_PRUNE,
            INNOVATION2_MODE_SPARSE_ROUTE,
        }:
            requested_mode = INNOVATION2_MODE_AUTO
            model_cfg.innovation2_mode = requested_mode

        if requested_mode == INNOVATION2_MODE_OFF:
            model_cfg.use_innovation2 = False
            model_cfg.innovation2_resolved_mode = INNOVATION2_MODE_OFF
            self.config.data.local_subgraph_enable = False
            self.config.data.local_subgraph_apply_mask = False
            self.config.runtime.ablation_preset = self._rewrite_innovation2_preset(
                self.config.runtime.ablation_preset,
                requested_mode,
                INNOVATION2_MODE_OFF,
            )
            return

        if requested_mode == INNOVATION2_MODE_DENSE_PRUNE:
            resolved_mode = INNOVATION2_MODE_DENSE_PRUNE
        elif requested_mode == INNOVATION2_MODE_SPARSE_ROUTE:
            resolved_mode = INNOVATION2_MODE_SPARSE_ROUTE
        else:
            data_paths = [self.config.data.root / self.dataset_spec.relative_path]
            if self.dataset_spec.companion_relative_path is not None:
                data_paths.append(self.config.data.root / self.dataset_spec.companion_relative_path)
            stats_list = [
                stats
                for stats in (
                    self._inspect_graph_structure(path, model_cfg.innovation2_auto_low_degree_cutoff)
                    for path in data_paths
                )
                if stats is not None
            ]
            if stats_list:
                stats = self._combine_graph_structure_stats(stats_list)
                model_cfg.innovation2_auto_avg_degree = stats["avg_degree"]
                model_cfg.innovation2_auto_low_degree_ratio = stats["low_degree_ratio"]
                model_cfg.innovation2_auto_median_degree = stats["median_degree"]
                is_sparse = (
                    stats["avg_degree"] < model_cfg.innovation2_auto_avg_degree_threshold
                    or stats["low_degree_ratio"] >= model_cfg.innovation2_auto_low_degree_ratio_threshold
                )
                resolved_mode = INNOVATION2_MODE_SPARSE_ROUTE if is_sparse else INNOVATION2_MODE_DENSE_PRUNE
            else:
                resolved_mode = INNOVATION2_MODE_SPARSE_ROUTE

        model_cfg.innovation2_resolved_mode = resolved_mode
        self.config.data.local_subgraph_enable = resolved_mode == INNOVATION2_MODE_DENSE_PRUNE
        self.config.data.local_subgraph_apply_mask = resolved_mode == INNOVATION2_MODE_DENSE_PRUNE
        self.config.runtime.ablation_preset = self._rewrite_innovation2_preset(
            self.config.runtime.ablation_preset,
            requested_mode,
            resolved_mode,
        )

    def _sync_config_from_checkpoint(self) -> None:
        checkpoint_path = None
        sync_reason = None
        if self.config.stage == STAGE_TEST and self.config.runtime.test_checkpoint:
            checkpoint_path = self.config.runtime.test_checkpoint
            sync_reason = "测试"
        elif self.config.runtime.resume_from:
            checkpoint_path = self.config.runtime.resume_from
            sync_reason = "断点续训"

        if checkpoint_path is None:
            return
        if not os.path.exists(checkpoint_path):
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            return

        model_config = checkpoint_config.get("model", {})
        data_config = checkpoint_config.get("data", {})
        optimizer_config = checkpoint_config.get("optimizer", {})
        configured_dropout = self.config.model.drop
        test_time_model_overrides = {}
        if self.config.stage == STAGE_TEST:
            for key in (
                "hypernym_path_prior_enable",
                "hypernym_path_prior_gamma",
                "hypernym_path_prior_max_hop",
                "hypernym_hub_calibration_enable",
                "hypernym_hub_calibration_gamma",
                "hypernym_hub_calibration_feature",
                "hypernym_hub_calibration_apply_coarse",
                "innovation4_tail_calibration_enable",
                "innovation4_tail_calibration_test_only",
                "innovation4_domain_boost_enable",
                "innovation4_domain_boost_gamma",
                "innovation4_domain_boost_top_relations",
                "innovation4_domain_boost_min_sources",
                "innovation4_hub_suppression_enable",
                "innovation4_hub_suppression_gamma",
                "innovation4_hub_suppression_feature",
            ):
                test_time_model_overrides[key] = getattr(self.config.model, key, None)

        for key, value in model_config.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)

        # dropout 没有权重形状约束，优先使用当前手动配置，方便测试时做显式覆盖。
        self.config.model.drop = configured_dropout
        for key, value in test_time_model_overrides.items():
            if value is not None and hasattr(self.config.model, key):
                setattr(self.config.model, key, value)

        # 测试阶段优先复用检查点中的评测 batch 配置，避免手工测试时遗漏。
        if self.config.stage == STAGE_TEST:
            for key in ("eval_batch_size", "use_test_as_validation"):
                if key in data_config and hasattr(self.config.data, key):
                    setattr(self.config.data, key, data_config[key])

        # 断点续训时同步优化器相关超参，避免恢复后配置与状态不一致。
        if sync_reason == "断点续训":
            for key, value in optimizer_config.items():
                if hasattr(self.config.optimizer, key):
                    setattr(self.config.optimizer, key, value)

        checkpoint_model_name = checkpoint_config.get("model_name")
        checkpoint_task_type = checkpoint_config.get("task_type")
        checkpoint_dataset = checkpoint_config.get("dataset")
        if checkpoint_model_name and checkpoint_model_name != self.config.model_name:
            raise ValueError(
                f"检查点模型类型不匹配：当前为 {self.config.model_name}，"
                f"检查点为 {checkpoint_model_name}。"
            )
        if checkpoint_task_type and checkpoint_task_type != self.config.task_type:
            raise ValueError(
                f"检查点任务类型不匹配：当前为 {self.config.task_type}，"
                f"检查点为 {checkpoint_task_type}。"
            )
        if checkpoint_dataset and checkpoint_dataset != self.config.dataset:
            raise ValueError(
                f"检查点数据集不匹配：当前为 {self.config.dataset}，"
                f"检查点为 {checkpoint_dataset}。"
            )

    def _set_seed(self) -> None:
        seed = self.config.runtime.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.config.runtime.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _build_optimizer(self):
        no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
        decay_params = []
        nodecay_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(tag in name for tag in no_decay):
                nodecay_params.append(parameter)
            else:
                decay_params.append(parameter)

        optimizer_cls = getattr(torch.optim, self.config.optimizer.optimizer)
        return optimizer_cls(
            [
                {"params": decay_params, "weight_decay": self.config.optimizer.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.config.optimizer.learning_rate,
        )

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.optimizer.scheduler_milestones,
            gamma=self.config.optimizer.scheduler_gamma,
        )

    def _build_grad_scaler(self):
        amp_enabled = self.config.runtime.amp and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
            return torch.amp.GradScaler(scaler_device, enabled=amp_enabled)
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            return torch.cuda.amp.GradScaler(enabled=amp_enabled)
        return None

    def _autocast_context(self):
        amp_enabled = self.config.runtime.amp and self.device.type == "cuda"
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type=self.device.type, enabled=amp_enabled)
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast(enabled=amp_enabled)
        return nullcontext()

    def _save_checkpoint(self, path, epoch, best_metric) -> None:
        checkpoint = {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint_path) -> dict:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info("已加载检查点：%s", checkpoint_path)
        return checkpoint

    def _prepare_runtime_environment(self) -> None:
        self.logger.info("正在准备运行环境...")
        if self.device.type == "cuda":
            self.logger.info("正在初始化 CUDA 上下文。这一步首次运行会稍慢。")
            self.logger.info("说明：图结构仍保留在 CPU，避免与 DataLoader 多进程传输产生冲突。")
            torch.empty(1, device=self.device)
            torch.cuda.synchronize()
        self.logger.info("运行环境准备完成。")

    def _synchronize_for_timing(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _format_param_count(count: int) -> str:
        if count >= 1_000_000:
            return f"{count:,} ({count / 1_000_000:.2f} M)"
        if count >= 1_000:
            return f"{count:,} ({count / 1_000:.2f} K)"
        return str(count)

    def _log_experiment_configuration(self) -> None:
        model_cfg = asdict(self.config.model)
        data_cfg = asdict(self.config.data)
        opt_cfg = asdict(self.config.optimizer)

        self.logger.info(
            "模型配置：hidden_dim=%d | num_layer=%d | num_qk_layer=%d | "
            "num_v_layer=%d | num_heads=%d | dropout=%.2f | 全局通路=linear attention | "
            "融合增强=%s | I2模式=%s/%s | 消融=%s",
            model_cfg["hidden_dim"],
            model_cfg["num_layer"],
            model_cfg["num_qk_layer"],
            model_cfg["num_v_layer"],
            model_cfg["num_heads"],
            model_cfg["drop"],
            "query+head-state节点标量门控" if model_cfg.get("use_innovation1", True) else "baseline标量融合",
            model_cfg.get("innovation2_mode", "off"),
            model_cfg.get("innovation2_resolved_mode", "off"),
            self.config.runtime.ablation_preset,
        )
        if model_cfg.get("internal_coarse_to_fine_enable", False):
            self.logger.info(
                "内部级联粗到细：开启 | mode=%s | joint_loss=%s | delta_decision=%s | "
                "coarse_loss_weight=%.2f | topk=%d | delta=%.2f",
                model_cfg.get("internal_coarse_mode", "flash"),
                "开启" if model_cfg.get("internal_coarse_use_joint_loss", True) else "关闭",
                "开启" if model_cfg.get("internal_coarse_use_delta_decision", True) else "关闭",
                model_cfg.get("internal_coarse_loss_weight", 1.0),
                model_cfg.get("internal_coarse_topk", 4),
                model_cfg.get("internal_coarse_delta", 8.0),
            )
        else:
            self.logger.info("内部级联粗到细：关闭")
        if model_cfg.get("innovation2_resolved_mode") == INNOVATION2_MODE_SPARSE_ROUTE:
            self.logger.info(
                "关系引导软路由：开启 | hidden_dim=%d | floor=%.2f | bias_init=%.2f | "
                "match_bias=%.2f",
                model_cfg.get("innovation2_routing_hidden_dim", 32),
                model_cfg.get("innovation2_routing_floor", 0.25),
                model_cfg.get("innovation2_routing_bias_init", 2.0),
                model_cfg.get("innovation2_routing_match_bias", 1.0),
            )
        if model_cfg.get("innovation2_mode") == INNOVATION2_MODE_AUTO:
            self.logger.info(
                "I2自动判定：avg_degree=%.3f | median_degree=%.3f | low_degree_ratio=%.3f | "
                "threshold(avg<%.2f or low>=%.2f)",
                model_cfg.get("innovation2_auto_avg_degree", 0.0),
                model_cfg.get("innovation2_auto_median_degree", 0.0),
                model_cfg.get("innovation2_auto_low_degree_ratio", 0.0),
                model_cfg.get("innovation2_auto_avg_degree_threshold", 3.2),
                model_cfg.get("innovation2_auto_low_degree_ratio_threshold", 0.65),
            )
        if model_cfg.get("hypernym_path_prior_enable", False):
            self.logger.info(
                "Hypernym路径先验：开启 | gamma=%.3f | max_hop=%d",
                model_cfg.get("hypernym_path_prior_gamma", 0.5),
                model_cfg.get("hypernym_path_prior_max_hop", 3),
            )
        if model_cfg.get("hypernym_hub_calibration_enable", False):
            self.logger.info(
                "Hypernym hub校准：开启 | feature=%s | gamma=%.3f",
                model_cfg.get("hypernym_hub_calibration_feature", "derivational_out"),
                model_cfg.get("hypernym_hub_calibration_gamma", -1.0),
            )
            if model_cfg.get("hypernym_hub_calibration_apply_coarse", False):
                self.logger.info("Hypernym hub校准：同步作用于coarse分数")
        if model_cfg.get("hypernym_hard_negative_enable", False):
            self.logger.info(
                "Hypernym hard negative：开启 | feature=%s | weight=%.3f | "
                "forward_weight=%s | reverse_weight=%s | count=%d | top_pool=%d | margin=%.3f",
                model_cfg.get("hypernym_hard_negative_feature", "derivational_out"),
                model_cfg.get("hypernym_hard_negative_weight", 0.1),
                model_cfg.get("hypernym_hard_negative_forward_weight", None),
                model_cfg.get("hypernym_hard_negative_reverse_weight", None),
                model_cfg.get("hypernym_hard_negative_count", 16),
                model_cfg.get("hypernym_hard_negative_top_pool", 128),
                model_cfg.get("hypernym_hard_negative_margin", 1.0),
            )
        if model_cfg.get("innovation4_tail_calibration_enable", False):
            self.logger.info(
                "创新点4 尾实体结构校准：开启 | test_only=%s | domain_boost=%s | "
                "hub_suppression=%s",
                "是" if model_cfg.get("innovation4_tail_calibration_test_only", True) else "否",
                "开启" if model_cfg.get("innovation4_domain_boost_enable", False) else "关闭",
                "开启" if model_cfg.get("innovation4_hub_suppression_enable", False) else "关闭",
            )
            if model_cfg.get("innovation4_domain_boost_enable", False):
                self.logger.info(
                    "创新点4 domain_boost：gamma=%.3f | top_relations=%d | min_sources=%d",
                    model_cfg.get("innovation4_domain_boost_gamma", 0.5),
                    model_cfg.get("innovation4_domain_boost_top_relations", 6),
                    model_cfg.get("innovation4_domain_boost_min_sources", 5),
                )
            if model_cfg.get("innovation4_hub_suppression_enable", False):
                self.logger.info(
                    "创新点4 hub_suppression：feature=%s | gamma=%.3f",
                    model_cfg.get("innovation4_hub_suppression_feature", "derivational_out"),
                    model_cfg.get("innovation4_hub_suppression_gamma", 1.0),
                )
        if model_cfg.get("nell_inverse_domain_prior_enable", False):
            self.logger.info(
                "NELL反向domain先验：开启 | gamma=%.3f | top_relations=%d | min_sources=%d | test_only=%s",
                model_cfg.get("nell_inverse_domain_prior_gamma", 0.5),
                model_cfg.get("nell_inverse_domain_prior_top_relations", 6),
                model_cfg.get("nell_inverse_domain_prior_min_sources", 5),
                "是" if model_cfg.get("nell_inverse_domain_prior_test_only", True) else "否",
            )
        self.logger.info(
            "训练配置：batch_size=%d | eval_batch_size=%d | epochs=%d | lr=%.2e | wd=%.2e",
            data_cfg["batch_size"],
            data_cfg["eval_batch_size"],
            opt_cfg["epochs"],
            opt_cfg["learning_rate"],
            opt_cfg["weight_decay"],
        )
        if data_cfg.get("local_subgraph_enable", False):
            self.logger.info(
                "局部子图结构缓存：开启 | apply_mask=%s | hops=%d | topk_1hop=%d | topk_2hop=%d | cache=%s",
                "开启" if data_cfg.get("local_subgraph_apply_mask", True) else "关闭",
                data_cfg["local_subgraph_hops"],
                data_cfg["local_subgraph_topk_1hop"],
                data_cfg["local_subgraph_topk_2hop"],
                "开启" if data_cfg["local_subgraph_cache"] else "关闭",
            )
        else:
            self.logger.info("局部子图裁剪：关闭")
        self.logger.info(
            "损失与采样：loss=%s | negative_samples=%d (实际=%d) | adv_temp=%.2f | grad_clip=%s",
            opt_cfg["loss_fn"],
            opt_cfg["num_negative_sample"],
            2 ** int(opt_cfg["num_negative_sample"]),
            opt_cfg["adversarial_temperature"],
            "关闭" if opt_cfg["grad_clip_norm"] is None else f"{opt_cfg['grad_clip_norm']}",
        )

    def _log_parameter_statistics(self) -> None:
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.logger.info(
            "模型参数：总参数量=%s | 可训练参数=%s",
            self._format_param_count(total_params),
            self._format_param_count(trainable_params),
        )

    def _log_data_pipeline_diagnostics(self) -> None:
        info = self.data_module.dataset_info
        self.logger.info(
            "图规模：train_entities=%d | train_edges=%d | eval_entities=%d | eval_edges=%d | relations=%d",
            info.train_num_entities,
            info.train_graph_num_edges,
            info.eval_num_entities,
            info.eval_graph_num_edges,
            info.num_relations,
        )
        self.logger.info(
            "拓扑先验缓存：train=%s | eval=%s",
            info.train_topology_prior_status,
            info.eval_topology_prior_status,
        )
        self.logger.info(
            "局部子图缓存：train=%s(avg_edges=%.2f,max_edges=%d) | eval=%s(avg_edges=%.2f,max_edges=%d)",
            info.train_local_subgraph_status,
            info.train_local_subgraph_avg_edges,
            info.train_local_subgraph_max_edges,
            info.eval_local_subgraph_status,
            info.eval_local_subgraph_avg_edges,
            info.eval_local_subgraph_max_edges,
        )
        if hasattr(info, "triple_subgraph_template_status"):
            self.logger.info(
                "三元分类样本级子图模板缓存：%s",
                info.triple_subgraph_template_status,
            )

    @staticmethod
    def _core_metrics(metrics) -> str:
        if hasattr(metrics, "accuracy"):
            return "Acc={:.4f} | Precision={:.4f} | Recall={:.4f} | F1={:.4f}".format(
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1,
            )
        return "MRR={:.4f} | Hits@1={:.4f} | Hits@10={:.4f}".format(
            metrics.mrr,
            metrics.hits1,
            metrics.hits10,
        )

    def _primary_metric_value(self, metrics) -> float:
        return float(getattr(metrics, self.task.primary_metric))

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_stage_title(self, title: str) -> None:
        line = "=" * 18
        self.logger.info("%s %s %s", line, title, line)

    def _reset_fusion_profile(self) -> None:
        if hasattr(self.model, "reset_fusion_profile"):
            self.model.reset_fusion_profile()

    def _log_fusion_profile(self, stage_name: str) -> None:
        if not hasattr(self.model, "get_fusion_profile_summary"):
            return
        summary = self.model.get_fusion_profile_summary()
        if summary is None:
            return

        total_ms = max(summary.total_ms, 1e-12)
        local_ratio = 100.0 * summary.local_ms / total_ms
        global_ratio = 100.0 * summary.global_ms / total_ms
        gate_ratio = 100.0 * summary.gate_ms / total_ms
        avg_local_ms = summary.local_ms / summary.layer_invocations
        avg_global_ms = summary.global_ms / summary.layer_invocations
        avg_gate_ms = summary.gate_ms / summary.layer_invocations
        self.logger.info(
            "融合通路计时[%s]：前向采样=%d | 层调用=%d | "
            "局部=%.2fms(%.1f%%) | 全局=%.2fms(%.1f%%) | 门控=%.2fms(%.1f%%)",
            stage_name,
            summary.forward_passes,
            summary.layer_invocations,
            summary.local_ms,
            local_ratio,
            summary.global_ms,
            global_ratio,
            summary.gate_ms,
            gate_ratio,
        )
        self.logger.info(
            "融合通路单层均值[%s]：局部=%.3fms | 全局=%.3fms | 门控=%.3fms",
            stage_name,
            avg_local_ms,
            avg_global_ms,
            avg_gate_ms,
        )

    def _log_relation_breakdown(self, split_name: str, metrics) -> None:
        per_relation = getattr(metrics, "per_relation", None)
        if not per_relation:
            return
        if split_name != "test":
            return

        self.logger.info("测试集按关系指标（按MRR升序）：")
        for relation_name, relation_metrics in sorted(
            per_relation.items(),
            key=lambda item: item[1].get("mrr", 0.0),
        ):
            self.logger.info(
                "  %s | count=%d | MRR=%.4f | Hits@1=%.4f | Hits@10=%.4f",
                relation_name,
                int(relation_metrics.get("count", 0)),
                relation_metrics.get("mrr", 0.0),
                relation_metrics.get("hits1", 0.0),
                relation_metrics.get("hits10", 0.0),
            )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._reset_fusion_profile()
        total_loss = 0.0
        total_steps = 0
        total_fetch_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_step_time = 0.0
        current_lr = self._current_learning_rate()
        epoch_start_time = time.perf_counter()
        train_loader = self.data_module.train_dataloader()
        total_batches = len(train_loader)
        profile_step_breakdown = self.config.runtime.profile_training_step_breakdown

        progress_bar = tqdm(
            train_loader,
            total=total_batches,
            desc=f"训练轮次 {epoch:02d}/{self.config.optimizer.epochs:02d}",
            dynamic_ncols=True,
            leave=False,
            bar_format="{desc} |{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        last_step_end_time = time.perf_counter()
        for step_idx, batch in enumerate(progress_bar, start=1):
            step_start_time = time.perf_counter()
            total_fetch_time += step_start_time - last_step_end_time
            self.optimizer.zero_grad(set_to_none=True)

            if profile_step_breakdown:
                self._synchronize_for_timing()
                forward_start = time.perf_counter()
            with self._autocast_context():
                loss = self.task.training_step(self.model, batch, self.config, self.device)
            if profile_step_breakdown:
                self._synchronize_for_timing()
                total_forward_time += time.perf_counter() - forward_start

            if profile_step_breakdown:
                self._synchronize_for_timing()
                backward_start = time.perf_counter()
            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            if self.config.optimizer.grad_clip_norm is not None:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm,
                )
            if profile_step_breakdown:
                self._synchronize_for_timing()
                total_backward_time += time.perf_counter() - backward_start

            if profile_step_breakdown:
                self._synchronize_for_timing()
                optimizer_step_start = time.perf_counter()
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if profile_step_breakdown:
                self._synchronize_for_timing()
                total_step_time += time.perf_counter() - optimizer_step_start

            total_loss += loss.item()
            total_steps += 1
            last_step_end_time = time.perf_counter()
        self.scheduler.step()
        average_loss = total_loss / max(total_steps, 1)
        elapsed = time.perf_counter() - epoch_start_time
        self.logger.info(
            "第 %d 轮训练完成：平均损失 = %.6f | 当前学习率 = %.2e | 轮次耗时 = %.2fs",
            epoch,
            average_loss,
            current_lr,
            elapsed,
        )
        if profile_step_breakdown and total_steps > 0:
            self.logger.info(
                "训练分阶段耗时[epoch-%d]：fetch=%.3fs/step | forward=%.3fs/step | backward=%.3fs/step | step=%.3fs/step",
                epoch,
                total_fetch_time / total_steps,
                total_forward_time / total_steps,
                total_backward_time / total_steps,
                total_step_time / total_steps,
            )
        self._log_fusion_profile(f"train-epoch-{epoch}")
        return average_loss

    def _evaluate_split(self, split_name: str):
        self._reset_fusion_profile()
        eval_start_time = time.perf_counter()
        if split_name == "valid":
            metrics = self.task.evaluate(
                self.model,
                self.data_module.val_dataloader(),
                self.device,
                self.config,
                split_name="valid",
            )
        elif split_name == "test":
            metrics = self.task.evaluate(
                self.model,
                self.data_module.test_dataloader(),
                self.device,
                self.config,
                split_name="test",
            )
        else:
            raise ValueError(f"不支持的数据划分：{split_name}")

        split_to_cn = {"valid": "验证集", "test": "测试集"}
        split_name_cn = split_to_cn.get(split_name, split_name)
        elapsed = time.perf_counter() - eval_start_time
        self.logger.info("%s：%s | 耗时 = %.2fs", split_name_cn, self._core_metrics(metrics), elapsed)
        self._log_relation_breakdown(split_name, metrics)
        if hasattr(self.task, "last_eval_threshold"):
            self.logger.info("%s阈值：threshold=%.4f", split_name_cn, float(self.task.last_eval_threshold))
        self._log_fusion_profile(split_name)
        return metrics

    def _train(self) -> None:
        best_metric = float("-inf")
        best_hits1 = 0.0
        best_hits10 = 0.0
        history = []
        total_epochs = self.config.optimizer.epochs
        self._log_stage_title("开始训练")
        self.logger.info(
            "模型：%s | 任务：%s | 数据集：%s | 总轮数：%d | 批大小：%d",
            self.config.model_name,
            self.config.task_type,
            self.config.dataset,
            total_epochs,
            self.config.data.batch_size,
        )

        for epoch in range(1, total_epochs + 1):
            self._log_stage_title(f"第 {epoch}/{total_epochs} 轮")
            self.logger.info(
                "开始第 %d/%d 轮训练：当前学习率 = %.2e",
                epoch,
                total_epochs,
                self._current_learning_rate(),
            )
            train_loss = self._train_one_epoch(epoch)
            valid_metrics = self._evaluate_split("valid")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": self._current_learning_rate(),
                    "valid": valid_metrics.to_dict(),
                }
            )

            valid_primary = self._primary_metric_value(valid_metrics)
            if valid_primary >= best_metric:
                best_metric = valid_primary
                best_hits1 = valid_metrics.hits1
                best_hits10 = valid_metrics.hits10
                self._save_checkpoint(self.artifacts.best_checkpoint, epoch, best_metric)
                self.logger.info("已保存新的最佳检查点：%s <-----------------!!!", self._core_metrics(valid_metrics))
                self.logger.info("最佳检查点路径：%s", self.artifacts.best_checkpoint)
            self.logger.info("本轮总结：%s | 当前最佳 %s = %.4f", self._core_metrics(valid_metrics), self.task.primary_metric, best_metric)

        self._load_checkpoint(self.artifacts.best_checkpoint)
        self._log_stage_title("最佳模型测试")
        test_metrics = self._evaluate_split("test")
        payload = {
            f"best_valid_{self.task.primary_metric}": best_metric,
            "best_valid_hits1": best_hits1,
            "best_valid_hits10": best_hits10,
            "test": test_metrics.to_dict(),
            "history": history,
            "best_checkpoint": str(self.artifacts.best_checkpoint),
        }
        save_json(self.artifacts.metrics_file, payload)
        self.logger.info(
            "训练结束，最佳验证指标：%s=%.4f | Hits@1=%.4f | Hits@10=%.4f >>>>>",
            self.task.primary_metric,
            best_metric,
            best_hits1,
            best_hits10,
        )
        self.logger.info("最佳检查点路径：%s", self.artifacts.best_checkpoint)

    def _test(self) -> None:
        checkpoint = self.config.runtime.test_checkpoint or self.artifacts.best_checkpoint
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise RuntimeError("当前请求使用 CUDA，但环境中不可用。")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"未找到测试检查点：{checkpoint}。"
                "请先训练模型，或显式设置 config.runtime.test_checkpoint。"
            )
        self._load_checkpoint(checkpoint)
        test_metrics = self._evaluate_split("test")
        payload = {
            "checkpoint": str(checkpoint),
            "test": test_metrics.to_dict(),
        }
        save_json(self.artifacts.metrics_file, payload)
        self.logger.info("测试完成：%s", self._core_metrics(test_metrics))
        self.logger.info("测试所用检查点：%s", checkpoint)

    def run(self) -> None:
        if self.config.stage == STAGE_TRAIN:
            self._train()
        elif self.config.stage == STAGE_TEST:
            self._test()
        else:
            raise ValueError(f"不支持的运行阶段：{self.config.stage}")
