"""Our DuetGraph experiment entry."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

import Our  # noqa: F401
from Utils.config import (
    INTERNAL_COARSE_MODES,
    INNOVATION2_MODES,
    INNOVATION2_MODE_AUTO,
    STAGES,
    TASK_TYPES,
    ExperimentConfig,
    apply_recommended_preset,
)
from Utils.trainer import ExperimentRunner


def _set_attrs(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        setattr(target, key, value)


def _set_if_not_none(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is not None:
            setattr(target, key, value)


def apply_our_component_switches(
    config: ExperimentConfig,
    *,
    use_innovation1: bool,
    use_innovation2: bool,
    use_innovation3: bool,
    use_innovation4: bool,
    innovation2_mode: str,
    innovation3_mode: str,
    innovation3_use_joint_loss: bool,
    innovation3_use_delta_decision: bool,
) -> None:
    config.model_name = "our_duetgraph"

    config.model.use_innovation1 = use_innovation1
    config.model.use_innovation2 = use_innovation2
    config.model.innovation2_mode = innovation2_mode
    if use_innovation2 and config.model.innovation2_mode == "off":
        config.model.innovation2_mode = INNOVATION2_MODE_AUTO
    config.data.local_subgraph_enable = False

    config.model.internal_coarse_to_fine_enable = use_innovation3
    config.model.internal_coarse_mode = innovation3_mode
    config.model.internal_coarse_use_joint_loss = use_innovation3 and innovation3_use_joint_loss
    config.model.internal_coarse_use_delta_decision = use_innovation3 and innovation3_use_delta_decision

    config.model.innovation4_tail_calibration_enable = use_innovation4

    config.runtime.ablation_preset = (
        f"I1={'on' if use_innovation1 else 'off'}|"
        f"I2={(config.model.innovation2_mode if use_innovation2 else 'off')}|"
        f"I3={'off' if not use_innovation3 else innovation3_mode}"
        f"|joint={'on' if innovation3_use_joint_loss and use_innovation3 else 'off'}"
        f"|delta={'on' if innovation3_use_delta_decision and use_innovation3 else 'off'}|"
        f"I4={'on' if use_innovation4 else 'off'}"
    )


def configure_runtime(config: ExperimentConfig) -> None:
    # 运行环境配置：一般只需要改 seed、device、test_checkpoint。
    _set_attrs(
        config.runtime,
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # 训练设备；有 CUDA 就用 GPU，否则自动 CPU
            "seed": 42,  # 随机种子；复现实验时保持不变
            "num_workers": 0,  # DataLoader 进程数；Windows 下 0 通常更稳
            "deterministic": True,  # 是否启用确定性算法；True 更可复现但可能稍慢
            "rspmm_backend": "native",  # 稀疏消息传递后端：native 稳定；extension 尝试 CUDA/C++ 扩展
            "cpu_negative_sampling_for_determinism": True,  # 负采样放 CPU 上做，减少随机性波动
            "run_name": None,  # 运行名；None 时自动用时间戳
            "resume_from": None,  # 断点续训 checkpoint 路径
            "test_checkpoint": None,  # stage=test 时加载的 checkpoint 路径
            "profile_fusion_paths": False,  # 是否统计局部/全局/融合门控通路耗时
            "profile_fusion_max_forwards": 20,  # 通路耗时最多统计多少个 forward
            "profile_training_step_breakdown": False,  # 是否统计 fetch/forward/backward/step 分阶段耗时
        },
    )
    # config.runtime.test_checkpoint = "Checkpoints\\models\\our_duetgraph\\inductive_reasoning\\nell-995_v1\\20260503_195729\\best.pt"


def configure_data(config: ExperimentConfig) -> None:
    # 数据与图结构配置：batch_size、eval_batch_size 和 remove_target_edges 最常调。
    _set_attrs(
        config.data,
        {
            "batch_size": 32,  # 训练 batch size
            "eval_batch_size": 32,  # 验证/测试 batch size
            "use_test_as_validation": True,  # 是否用测试集选模；正式实验建议 False
            "remove_target_edges": True,  # 训练/推理时是否移除当前待预测目标边，防止信息泄漏
            "remove_all_one_hop_edges": False,  # 是否移除头尾实体间所有一跳边；更严格但可能损失有效局部结构
            # Innovation 2 dense_prune branch. Sparse_route will ignore the mask after auto resolution.
            "local_subgraph_hops": 2,  # 创新点2 dense_prune 分支：局部子图裁剪半径
            "local_subgraph_topk_1hop": 32,  # dense_prune：一跳邻居最多保留数
            "local_subgraph_topk_2hop": 8,  # dense_prune：二跳邻居最多保留数
            "local_subgraph_cache": True,  # 是否缓存裁剪结果；通常开启更快
        },
    )


def configure_optimizer(config: ExperimentConfig) -> None:
    # 优化器配置：NELL 当前最好路线主要保持这组值。
    _set_attrs(
        config.optimizer,
        {
            "epochs": 20,  # 训练轮数
            "learning_rate": 5e-3,  # 学习率
            "weight_decay": 1e-4,  # 权重衰减
            "loss_fn": "bce",  # 损失函数：bce 或 ce；当前归纳实验主要用 bce
            "num_negative_sample": 6,  # 负样本指数；实际负样本数为 2^n，6=64
            "adversarial_temperature": 0.5,  # 自对抗负采样温度；越小越偏向高分困难负例
        },
    )


def configure_model(
    config: ExperimentConfig,
    *,
    innovation3_loss_weight: float,
    innovation3_topk: int,
    innovation3_delta: float,
    use_innovation4: bool,
) -> None:
    # 模型与创新点细节配置：hidden_dim、I2 路由维度、I3/I4 参数是重点。
    _set_attrs(
        config.model,
        {
            "hidden_dim": 32,  # 隐藏维度；NELL 历史最好 checkpoint 使用 64
            "num_layer": 3,  # 双通路融合层数
            "num_qk_layer": 2,  # Query/Key 编码层数
            "num_v_layer": 3,  # Value/局部消息传递层数
            "num_heads": 4,  # 全局线性注意力头数
            "drop": 0.1,  # dropout 比例
            "innovation2_routing_hidden_dim": 32,  # 创新点2 sparse_route 分支的路由 MLP 隐藏维度
            "internal_coarse_loss_weight": innovation3_loss_weight,  # 创新点3 coarse loss 权重
            "internal_coarse_topk": innovation3_topk,  # 创新点3推理时 coarse 粗筛候选数
            "internal_coarse_delta": innovation3_delta,  # 创新点3 top-k/delta 动态决策阈值
            # WN18RR B-group training switch. NELL keeps it disabled.
            "hypernym_hard_negative_enable": True,  # WN18RR hypernym hard negative；NELL 关闭
            # Innovation 4: validation-guided relation-adaptive tail calibration.
            "innovation4_tail_calibration_enable": use_innovation4,  # 创新点4总开关：尾实体结构校准
            "innovation4_tail_calibration_test_only": True,  # 仅测试阶段生效，避免影响验证选模
            "innovation4_domain_boost_enable": True,  # I4 domain_boost：提升结构域匹配尾实体；NELL 有效
            "innovation4_domain_boost_gamma": 1.0,  # domain_boost 强度；你实验中 1.0 最好
            "innovation4_domain_boost_top_relations": 6,  # 构造 domain signature 时取前多少个高频关系
            "innovation4_domain_boost_min_sources": 5,  # 构造先验所需最少源实体数，太小会不稳
            "innovation4_hub_suppression_enable": False,  # I4 hub_suppression：压制 hub 混淆；WN 可开，NELL 关
            "innovation4_hub_suppression_gamma": 1.0,  # hub_suppression 强度
            "innovation4_hub_suppression_feature": "derivational_out",  # hub 特征类型：derivational_out/in 或 degree
            # Legacy single-dataset post-processing entries are kept off; I4 is the unified path.
            "hypernym_hub_calibration_enable": False,  # 旧版 WN 后处理入口；统一到 I4 后保持关闭
            "nell_inverse_domain_prior_enable": False,  # 旧版 NELL 后处理入口；统一到 I4 后保持关闭
        },
    )


def build_manual_config() -> ExperimentConfig:
    # =========================
    # 一、实验对象
    # =========================
    config = ExperimentConfig(
        model_name="our_duetgraph",  # 模型入口，固定为 Our 版本
        task_type="inductive_reasoning",  # 任务类型：inductive_reasoning / transductive_reasoning
        dataset="fb15k-237_v4",  # 数据集：如 nell-995_v1、wn18rr_v1、fb15k-237_v1
        stage="train",  # 运行阶段： train 训练并测试最佳模型；test 只测试 checkpoint
    )
    apply_recommended_preset(config)

    # =========================
    # 二、创新点组合
    # =========================
    use_innovation1 = True  # 创新点1：查询感知细粒度融合
    use_innovation2 = True  # 创新点2：图结构自适应局部处理；稠密裁剪、稀疏软路由
    use_innovation3 = True  # 创新点3：内部粗到细推理
    use_innovation4 = True  # 创新点4：验证集引导的尾实体结构校准

    innovation2_mode = "dense_prune"  # auto 自动判定； dense_prune 强制裁剪； sparse_route 强制软路由；off 关闭
    innovation3_mode = "flash"  # flash 轻量共享头；pro 增强分支
    innovation3_use_joint_loss = True  # 是否训练 coarse loss + fine loss
    innovation3_use_delta_decision = True  # 是否测试时使用 top-k + delta 动态决策
    innovation3_loss_weight = 1.0  # coarse loss 权重
    innovation3_topk = 4  # coarse 阶段保留候选数
    innovation3_delta = 8.0  # 动态决策阈值

    configure_runtime(config)
    apply_our_component_switches(
        config,
        use_innovation1=use_innovation1,
        use_innovation2=use_innovation2,
        use_innovation3=use_innovation3,
        use_innovation4=use_innovation4,
        innovation2_mode=innovation2_mode,
        innovation3_mode=innovation3_mode,
        innovation3_use_joint_loss=innovation3_use_joint_loss,
        innovation3_use_delta_decision=innovation3_use_delta_decision,
    )

    # =========================
    # 三、核心超参
    # =========================
    configure_data(config)
    configure_optimizer(config)
    configure_model(
        config,
        innovation3_loss_weight=innovation3_loss_weight,
        innovation3_topk=innovation3_topk,
        innovation3_delta=innovation3_delta,
        use_innovation4=use_innovation4,
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Our DuetGraph entry.")
    parser.add_argument("--task-type", choices=TASK_TYPES, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--cpu-negative-sampling", action="store_true")
    parser.add_argument("--gpu-negative-sampling", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--test-checkpoint", default=None)
    parser.add_argument("--profile-fusion-paths", action="store_true")
    parser.add_argument("--profile-fusion-max-forwards", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--loss-fn", choices=["bce", "ce"], default=None)
    parser.add_argument("--negative-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layer", type=int, default=None)
    parser.add_argument("--num-qk-layer", type=int, default=None)
    parser.add_argument("--num-v-layer", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--innovation1-enable", action="store_true")
    parser.add_argument("--innovation1-disable", action="store_true")
    parser.add_argument("--innovation2-enable", action="store_true")
    parser.add_argument("--innovation2-disable", action="store_true")
    parser.add_argument("--innovation2-mode", choices=INNOVATION2_MODES, default=None)
    parser.add_argument("--innovation3-enable", action="store_true")
    parser.add_argument("--innovation3-disable", action="store_true")
    parser.add_argument("--innovation3-mode", choices=INTERNAL_COARSE_MODES, default=None)
    parser.add_argument("--innovation3-use-joint-loss", action="store_true")
    parser.add_argument("--innovation3-no-joint-loss", action="store_true")
    parser.add_argument("--innovation3-use-delta-decision", action="store_true")
    parser.add_argument("--innovation3-no-delta-decision", action="store_true")
    parser.add_argument("--innovation3-loss-weight", type=float, default=None)
    parser.add_argument("--innovation3-topk", type=int, default=None)
    parser.add_argument("--innovation3-delta", type=float, default=None)
    parser.add_argument("--adversarial-temperature", type=float, default=None)
    parser.add_argument("--local-subgraph-enable", action="store_true")
    parser.add_argument("--local-subgraph-disable", action="store_true")
    parser.add_argument("--local-subgraph-hops", type=int, default=None)
    parser.add_argument("--local-subgraph-topk-1hop", type=int, default=None)
    parser.add_argument("--local-subgraph-topk-2hop", type=int, default=None)
    parser.add_argument("--no-local-subgraph-cache", action="store_true")
    parser.add_argument("--remove-target-edges", action="store_true")
    parser.add_argument("--no-remove-target-edges", action="store_true")
    parser.add_argument("--remove-all-one-hop-edges", action="store_true")
    parser.add_argument("--no-remove-all-one-hop-edges", action="store_true")
    parser.add_argument("--use-test-as-validation", action="store_true")
    parser.add_argument("--use-valid-as-validation", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.task_type is not None:
        config.task_type = args.task_type
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.stage is not None:
        config.stage = args.stage
    if args.task_type is not None or args.dataset is not None:
        apply_recommended_preset(config)

    _set_if_not_none(
        config.runtime,
        {
            "device": args.device,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "run_name": args.run_name,
            "resume_from": args.resume_from,
            "test_checkpoint": args.test_checkpoint,
            "profile_fusion_max_forwards": args.profile_fusion_max_forwards,
        },
    )
    _set_if_not_none(
        config.optimizer,
        {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_fn": args.loss_fn,
            "num_negative_sample": args.negative_samples,
            "adversarial_temperature": args.adversarial_temperature,
        },
    )
    _set_if_not_none(
        config.data,
        {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "local_subgraph_hops": args.local_subgraph_hops,
            "local_subgraph_topk_1hop": args.local_subgraph_topk_1hop,
            "local_subgraph_topk_2hop": args.local_subgraph_topk_2hop,
        },
    )
    _set_if_not_none(
        config.model,
        {
            "hidden_dim": args.hidden_dim,
            "num_layer": args.num_layer,
            "num_qk_layer": args.num_qk_layer,
            "num_v_layer": args.num_v_layer,
            "num_heads": args.num_heads,
            "drop": args.dropout,
        },
    )

    if args.deterministic:
        config.runtime.deterministic = True
    if args.non_deterministic:
        config.runtime.deterministic = False
    if args.cpu_negative_sampling:
        config.runtime.cpu_negative_sampling_for_determinism = True
    if args.gpu_negative_sampling:
        config.runtime.cpu_negative_sampling_for_determinism = False
    if args.profile_fusion_paths:
        config.runtime.profile_fusion_paths = True
    if args.amp:
        config.runtime.amp = True
    if args.compile_model:
        config.runtime.compile_model = True

    use_innovation1 = config.model.use_innovation1
    use_innovation2 = config.model.use_innovation2
    use_innovation3 = config.model.internal_coarse_to_fine_enable
    use_innovation4 = config.model.innovation4_tail_calibration_enable
    innovation2_mode = args.innovation2_mode or config.model.innovation2_mode
    innovation3_mode = args.innovation3_mode or config.model.internal_coarse_mode
    innovation3_use_joint_loss = config.model.internal_coarse_use_joint_loss
    innovation3_use_delta_decision = config.model.internal_coarse_use_delta_decision

    if args.innovation1_enable:
        use_innovation1 = True
    if args.innovation1_disable:
        use_innovation1 = False
    if args.innovation2_enable:
        use_innovation2 = True
    if args.innovation2_disable:
        use_innovation2 = False
    if args.innovation2_mode is not None:
        use_innovation2 = args.innovation2_mode != "off"
    if args.innovation3_enable:
        use_innovation3 = True
    if args.innovation3_disable:
        use_innovation3 = False
    if args.innovation3_use_joint_loss:
        innovation3_use_joint_loss = True
    if args.innovation3_no_joint_loss:
        innovation3_use_joint_loss = False
    if args.innovation3_use_delta_decision:
        innovation3_use_delta_decision = True
    if args.innovation3_no_delta_decision:
        innovation3_use_delta_decision = False

    apply_our_component_switches(
        config,
        use_innovation1=use_innovation1,
        use_innovation2=use_innovation2,
        use_innovation3=use_innovation3,
        use_innovation4=use_innovation4,
        innovation2_mode=innovation2_mode,
        innovation3_mode=innovation3_mode,
        innovation3_use_joint_loss=innovation3_use_joint_loss,
        innovation3_use_delta_decision=innovation3_use_delta_decision,
    )

    if args.innovation3_loss_weight is not None:
        config.model.internal_coarse_loss_weight = args.innovation3_loss_weight
    if args.innovation3_topk is not None:
        config.model.internal_coarse_topk = args.innovation3_topk
    if args.innovation3_delta is not None:
        config.model.internal_coarse_delta = args.innovation3_delta
    if args.local_subgraph_enable:
        config.data.local_subgraph_enable = True
    if args.local_subgraph_disable:
        config.data.local_subgraph_enable = False
    if args.no_local_subgraph_cache:
        config.data.local_subgraph_cache = False
    if args.remove_target_edges:
        config.data.remove_target_edges = True
    if args.no_remove_target_edges:
        config.data.remove_target_edges = False
    if args.remove_all_one_hop_edges:
        config.data.remove_all_one_hop_edges = True
    if args.no_remove_all_one_hop_edges:
        config.data.remove_all_one_hop_edges = False
    if args.use_test_as_validation:
        config.data.use_test_as_validation = True
    if args.use_valid_as_validation:
        config.data.use_test_as_validation = False


def main() -> None:
    config = build_manual_config()
    if len(sys.argv) > 1:
        apply_cli_overrides(config, parse_args())

    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
