from __future__ import annotations

import torch
import torch.nn.functional as F

from Utils.config import (
    TASK_INDUCTIVE,
    TASK_LARGE_SCALE,
    TASK_TRANSDUCTIVE,
    TASK_TRIPLE_CLASSIFICATION,
    TRIPLE_CLASSIFICATION_ENCODER_LOCAL_SUBGRAPH,
)
from Utils.data import build_data_module
from Utils.interfaces import BaseTask
from Utils.metrics import ClassificationMetrics, RankingMetricAccumulator
from Utils.registry import register_task


class LinkPredictionTask(BaseTask):
    primary_metric = "mrr"

    def build_data_module(self, config, dataset_spec):
        return build_data_module(
            root=config.data.root,
            dataset_spec=dataset_spec,
            task_type=self.task_name,
            batch_size=config.data.batch_size,
            eval_batch_size=config.data.eval_batch_size,
            num_workers=config.runtime.num_workers,
            use_test_as_validation=config.data.use_test_as_validation,
            enable_topology_prior=False,
            topology_prior_cache=config.data.topology_prior_cache,
            topology_pagerank_damping=config.data.topology_pagerank_damping,
            topology_pagerank_iterations=config.data.topology_pagerank_iterations,
            topology_pagerank_tolerance=config.data.topology_pagerank_tolerance,
            enable_local_subgraph=config.data.local_subgraph_enable,
            local_subgraph_cache=config.data.local_subgraph_cache,
            local_subgraph_hops=config.data.local_subgraph_hops,
            local_subgraph_topk_1hop=config.data.local_subgraph_topk_1hop,
            local_subgraph_topk_2hop=config.data.local_subgraph_topk_2hop,
        )

    @staticmethod
    def _negative_sample(
        mask: torch.Tensor,
        num_negative_sample: int,
        *,
        sample_on_cpu: bool = False,
    ) -> torch.Tensor:
        sample_count = min(mask.size(1), num_negative_sample)

        if sample_on_cpu and mask.device.type != "cpu":
            sample_space = (~mask).to(device="cpu", dtype=torch.float32)
            sampled_index = torch.multinomial(
                sample_space,
                num_samples=sample_count,
                replacement=True,
            )
            return sampled_index.to(mask.device)

        sample_space = (~mask).float()
        return torch.multinomial(
            sample_space,
            num_samples=sample_count,
            replacement=True,
        )

    @staticmethod
    def _remove_target_edges(batch: dict[str, torch.Tensor], remove_all: bool) -> dict[str, torch.Tensor]:
        graph = batch["graph"]
        h_index = batch["h_index"]
        r_index = batch["r_index"]
        t_index = batch["t_index"]

        h_remove = torch.cat([h_index, t_index], dim=0)
        r_remove = torch.cat(
            [r_index, torch.where(r_index % 2 == 0, r_index + 1, r_index - 1)],
            dim=0,
        )
        t_remove = torch.cat([t_index, h_index], dim=0)

        if remove_all:
            source_hash = graph.edge_index[:, 0] + graph.edge_index[:, 2] * graph.num_nodes
            target_hash = h_remove + t_remove * graph.num_nodes
        else:
            source_hash = graph.edge_index[:, 2] + (
                graph.edge_index[:, 0] + graph.edge_index[:, 1] * graph.num_nodes
            ) * graph.num_nodes
            target_hash = t_remove + (h_remove + r_remove * graph.num_nodes) * graph.num_nodes

        batch["graph_mask"] = ~torch.isin(source_hash, target_hash)
        return batch

    @staticmethod
    def _compute_loss(score: torch.Tensor, batch: dict[str, torch.Tensor], config) -> torch.Tensor:
        positive_index = batch["t_index"].unsqueeze(1)
        filter_mask = batch["filter_mask"].bool()

        if config.optimizer.loss_fn == "bce":
            negative_index = batch.get("negative_index")
            if negative_index is None:
                negative_sample_count = min(
                    filter_mask.size(1),
                    2 ** config.optimizer.num_negative_sample,
                )
                negative_index = LinkPredictionTask._negative_sample(
                    filter_mask,
                    negative_sample_count,
                    sample_on_cpu=config.runtime.cpu_negative_sampling_for_determinism,
                )
                batch["negative_index"] = negative_index
            indices = torch.cat([positive_index, negative_index], dim=1)
            logits = torch.gather(score, 1, indices)
            target = torch.zeros_like(logits)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            weights = torch.ones_like(logits)
            with torch.no_grad():
                weights[:, 1:] = F.softmax(
                    logits[:, 1:] / config.optimizer.adversarial_temperature,
                    dim=-1,
                )
            return (loss * weights).sum()

        return F.cross_entropy(score, positive_index.view(-1))

    @classmethod
    def _compute_hypernym_hard_negative_loss(
        cls,
        score: torch.Tensor,
        batch: dict[str, torch.Tensor],
        config,
    ) -> torch.Tensor:
        if not getattr(config.model, "hypernym_hard_negative_enable", False):
            return score.new_zeros(())

        graph = batch["graph"]
        relation_weights = cls._hypernym_relation_weights(graph, config)
        if not relation_weights:
            return score.new_zeros(())

        relation_index = batch["r_index"].detach().cpu()
        query_mask = torch.tensor(
            [
                int(relation_id) in relation_weights
                and relation_weights[int(relation_id)] > 0.0
                for relation_id in relation_index.tolist()
            ],
            dtype=torch.bool,
            device=score.device,
        )
        if not bool(query_mask.any()):
            return score.new_zeros(())

        feature_name = getattr(
            config.model,
            "hypernym_hard_negative_feature",
            "derivational_out",
        )
        feature = cls._get_hypernym_hub_feature(graph, feature_name, score.device)
        if feature is None:
            return score.new_zeros(())

        top_pool = min(
            int(getattr(config.model, "hypernym_hard_negative_top_pool", 128)),
            score.size(1),
        )
        sample_count = min(
            int(getattr(config.model, "hypernym_hard_negative_count", 16)),
            top_pool,
        )
        if top_pool <= 0 or sample_count <= 0:
            return score.new_zeros(())

        candidate_index = torch.topk(feature, k=top_pool, largest=True).indices
        filter_mask = batch["filter_mask"].bool()
        answer = batch["t_index"]
        margin = float(getattr(config.model, "hypernym_hard_negative_margin", 1.0))
        losses: list[torch.Tensor] = []

        for batch_index in torch.nonzero(query_mask, as_tuple=False).flatten().tolist():
            valid_candidate_mask = ~filter_mask[batch_index, candidate_index]
            valid_candidate_mask &= candidate_index != answer[batch_index]
            valid_candidates = candidate_index[valid_candidate_mask]
            if valid_candidates.numel() == 0:
                continue
            selected_candidates = valid_candidates[:sample_count]
            positive_score = score[batch_index, answer[batch_index]]
            negative_score = score[batch_index, selected_candidates]
            relation_id = int(batch["r_index"][batch_index].detach().cpu().item())
            relation_weight = relation_weights.get(relation_id, 0.0)
            if relation_weight <= 0.0:
                continue
            losses.append(
                relation_weight
                * F.softplus(negative_score - positive_score + margin).mean()
            )

        if not losses:
            return score.new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def _split_model_output(output) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(output, dict):
            score_fine = output.get("score_fine")
            score_coarse = output.get("score_coarse")
            if score_fine is None:
                raise ValueError("Model output dict must contain 'score_fine'.")
            return score_fine, score_coarse
        return output, None

    @staticmethod
    def _compute_coarse_to_fine_ranks(
        score_fine: torch.Tensor,
        score_coarse: torch.Tensor,
        filter_mask: torch.Tensor,
        answer: torch.Tensor,
        top_k: int,
        delta: float,
    ) -> torch.Tensor:
        if score_fine.size(1) <= 1:
            return torch.ones_like(answer, dtype=torch.long)

        max_topk = max(1, score_fine.size(1) - 1)
        top_k = min(max(int(top_k), 1), max_topk)

        coarse_mask = filter_mask.clone()
        coarse_mask[torch.arange(filter_mask.size(0), device=filter_mask.device), answer] = False
        coarse_score = score_coarse.masked_fill(coarse_mask, float("-inf"))
        sorted_index = torch.argsort(coarse_score, dim=1, descending=True)

        top_index = sorted_index[:, :top_k]
        other_index = sorted_index[:, top_k:]
        answer_score = score_fine.gather(1, answer.unsqueeze(1))
        full_rank = ((score_fine >= answer_score) & (~filter_mask)).sum(dim=1) + 1

        if other_index.size(1) == 0:
            return full_rank

        top_score = torch.gather(score_fine, 1, top_index)
        other_score = torch.gather(score_fine, 1, other_index)
        top_mask = torch.gather(filter_mask, 1, top_index)
        other_mask = torch.gather(filter_mask, 1, other_index)

        top_max = top_score.max(dim=1).values
        other_max = other_score.max(dim=1).values
        diff = other_max - top_max
        in_topk = (answer.unsqueeze(1) == top_index).any(dim=1)

        top_rank = ((top_score >= answer_score) & (~top_mask)).sum(dim=1) + 1
        other_rank = ((other_score >= answer_score) & (~other_mask)).sum(dim=1) + 1

        select_top = (diff <= delta) & in_topk
        select_other = (diff > delta) & (~in_topk)
        return torch.where(select_top, top_rank, torch.where(select_other, other_rank, full_rank))

    @staticmethod
    def _apply_local_subgraph_mask(
        batch: dict[str, torch.Tensor],
        *,
        apply_mask: bool,
    ) -> dict[str, torch.Tensor]:
        if not apply_mask:
            return batch

        local_subgraph_mask = batch.get("local_subgraph_mask")
        if local_subgraph_mask is None:
            return batch

        local_subgraph_mask = local_subgraph_mask.bool()
        if "graph_mask" in batch:
            batch["graph_mask"] = batch["graph_mask"].bool() & local_subgraph_mask
        else:
            batch["graph_mask"] = local_subgraph_mask
        return batch

    @staticmethod
    def _build_relation_adjacency(graph, relation_id: int) -> dict[int, list[int]]:
        edge_index = graph.edge_index.detach().cpu()
        relation_mask = edge_index[:, 1] == int(relation_id)
        relation_edges = edge_index[relation_mask]
        adjacency: dict[int, list[int]] = {}
        for source, _, target in relation_edges.tolist():
            adjacency.setdefault(int(source), []).append(int(target))
        return adjacency

    @staticmethod
    def _hypernym_relation_ids(graph) -> set[int]:
        id2relation = getattr(graph, "id2relation", None)
        if not id2relation:
            return set()
        return {
            int(relation_id)
            for relation_id, relation_name in id2relation.items()
            if relation_name in {"_hypernym", "-_hypernym"}
        }

    @staticmethod
    def _hypernym_relation_weights(graph, config) -> dict[int, float]:
        id2relation = getattr(graph, "id2relation", None)
        if not id2relation:
            return {}

        base_weight = float(getattr(config.model, "hypernym_hard_negative_weight", 0.1))
        forward_weight = getattr(config.model, "hypernym_hard_negative_forward_weight", None)
        reverse_weight = getattr(config.model, "hypernym_hard_negative_reverse_weight", None)
        if forward_weight is None:
            forward_weight = base_weight
        if reverse_weight is None:
            reverse_weight = base_weight

        weights: dict[int, float] = {}
        for relation_id, relation_name in id2relation.items():
            if relation_name == "_hypernym":
                weights[int(relation_id)] = float(forward_weight)
            elif relation_name == "-_hypernym":
                weights[int(relation_id)] = float(reverse_weight)
        return weights

    @staticmethod
    def _relation_ids_by_name(graph, relation_names: set[str]) -> set[int]:
        id2relation = getattr(graph, "id2relation", None)
        if not id2relation:
            return set()
        return {
            int(relation_id)
            for relation_id, relation_name in id2relation.items()
            if relation_name in relation_names
        }

    @classmethod
    def _get_hypernym_hub_feature(
        cls,
        graph,
        feature_name: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        cache = getattr(graph, "_hypernym_hub_feature_cache", None)
        if cache is None:
            cache = {}
            setattr(graph, "_hypernym_hub_feature_cache", cache)
        if feature_name in cache:
            return cache[feature_name].to(device)

        edge_index = graph.edge_index.detach().cpu()
        if edge_index.numel() == 0:
            feature = torch.zeros(graph.num_nodes, dtype=torch.float32)
            cache[feature_name] = feature
            return feature.to(device)

        source = edge_index[:, 0].long()
        relation = edge_index[:, 1].long()
        target = edge_index[:, 2].long()
        degree = torch.zeros(graph.num_nodes, dtype=torch.float32)

        if feature_name in {"derivational_out", "derivational_in"}:
            derivational_ids = cls._relation_ids_by_name(
                graph,
                {"_derivationally_related_form"},
            )
            if not derivational_ids:
                return None
            relation_mask = torch.zeros_like(relation, dtype=torch.bool)
            for relation_id in derivational_ids:
                relation_mask |= relation == int(relation_id)
            selected_source = source[relation_mask]
            selected_target = target[relation_mask]
            if feature_name == "derivational_out":
                degree.index_add_(
                    0,
                    selected_source,
                    torch.ones(selected_source.numel(), dtype=torch.float32),
                )
            else:
                degree.index_add_(
                    0,
                    selected_target,
                    torch.ones(selected_target.numel(), dtype=torch.float32),
                )
        elif feature_name == "degree":
            degree.index_add_(0, source, torch.ones(source.numel(), dtype=torch.float32))
            degree.index_add_(0, target, torch.ones(target.numel(), dtype=torch.float32))
        else:
            return None

        feature = torch.log1p(degree)
        cache[feature_name] = feature
        return feature.to(device)

    @classmethod
    def _apply_hypernym_hub_calibration(
        cls,
        score: torch.Tensor,
        batch: dict[str, torch.Tensor],
        config,
    ) -> torch.Tensor:
        if config is None or not getattr(config.model, "hypernym_hub_calibration_enable", False):
            return score

        gamma = float(getattr(config.model, "hypernym_hub_calibration_gamma", -1.0))
        if gamma == 0.0:
            return score

        graph = batch["graph"]
        hypernym_ids = cls._hypernym_relation_ids(graph)
        if not hypernym_ids:
            return score

        feature_name = getattr(
            config.model,
            "hypernym_hub_calibration_feature",
            "derivational_out",
        )
        feature = cls._get_hypernym_hub_feature(graph, feature_name, score.device)
        if feature is None:
            return score

        relation_index = batch["r_index"].detach().cpu()
        query_mask = torch.tensor(
            [int(relation_id) in hypernym_ids for relation_id in relation_index.tolist()],
            dtype=torch.bool,
            device=score.device,
        )
        if not bool(query_mask.any()):
            return score

        adjusted_score = score.clone()
        adjusted_score[query_mask] = adjusted_score[query_mask] + gamma * feature.unsqueeze(0)
        return adjusted_score

    @classmethod
    def _get_nell_inverse_domain_prior(cls, graph, config, device: torch.device) -> dict[int, torch.Tensor]:
        if getattr(config.model, "innovation4_tail_calibration_enable", False):
            top_relations = int(getattr(config.model, "innovation4_domain_boost_top_relations", 6))
            min_sources = int(getattr(config.model, "innovation4_domain_boost_min_sources", 5))
        else:
            top_relations = int(getattr(config.model, "nell_inverse_domain_prior_top_relations", 6))
            min_sources = int(getattr(config.model, "nell_inverse_domain_prior_min_sources", 5))
        cache_key = (top_relations, min_sources)
        cache = getattr(graph, "_nell_inverse_domain_prior_cache", None)
        if cache is None:
            cache = {}
            setattr(graph, "_nell_inverse_domain_prior_cache", cache)
        if cache_key in cache:
            return {
                relation_id: prior.to(device)
                for relation_id, prior in cache[cache_key].items()
            }

        id2relation = getattr(graph, "id2relation", None)
        if not id2relation:
            cache[cache_key] = {}
            return {}

        relation2id = {relation_name: int(relation_id) for relation_id, relation_name in id2relation.items()}
        edge_index = graph.edge_index.detach().cpu()
        if edge_index.numel() == 0:
            cache[cache_key] = {}
            return {}

        source = edge_index[:, 0].long()
        relation = edge_index[:, 1].long()
        target = edge_index[:, 2].long()
        node_relation_profile: list[set[int]] = [set() for _ in range(graph.num_nodes)]
        for source_id, relation_id, target_id in edge_index.tolist():
            node_relation_profile[int(source_id)].add(int(relation_id))
            node_relation_profile[int(target_id)].add(int(relation_id))

        prior_cache: dict[int, torch.Tensor] = {}
        for relation_id, relation_name in id2relation.items():
            if relation_name.startswith("-"):
                continue
            inverse_relation_id = relation2id.get(f"-{relation_name}")
            if inverse_relation_id is None:
                continue

            relation_mask = relation == int(relation_id)
            relation_sources = source[relation_mask].unique(sorted=False)
            if relation_sources.numel() < min_sources:
                continue

            signature_counts: dict[int, int] = {}
            for source_id in relation_sources.tolist():
                for profile_relation_id in node_relation_profile[int(source_id)]:
                    if profile_relation_id in {int(relation_id), int(inverse_relation_id)}:
                        continue
                    signature_counts[profile_relation_id] = signature_counts.get(profile_relation_id, 0) + 1
            if not signature_counts:
                continue

            signature_ids = [
                relation_id
                for relation_id, _ in sorted(
                    signature_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ][:top_relations]
            candidate_score = torch.zeros(graph.num_nodes, dtype=torch.float32)
            signature_set = set(signature_ids)
            for node_id, profile in enumerate(node_relation_profile):
                overlap = len(profile & signature_set)
                if overlap > 0:
                    candidate_score[node_id] = float(overlap)
            if float(candidate_score.max().item()) > 0.0:
                candidate_score = candidate_score / candidate_score.max()
                prior_cache[int(inverse_relation_id)] = candidate_score

        cache[cache_key] = prior_cache
        return {relation_id: prior.to(device) for relation_id, prior in prior_cache.items()}

    @classmethod
    def _apply_nell_inverse_domain_prior(
        cls,
        score: torch.Tensor,
        batch: dict[str, torch.Tensor],
        config,
        split_name: str | None = None,
    ) -> torch.Tensor:
        if config is None or not getattr(config.model, "nell_inverse_domain_prior_enable", False):
            return score
        if (
            getattr(config.model, "nell_inverse_domain_prior_test_only", True)
            and split_name != "test"
        ):
            return score
        gamma = float(getattr(config.model, "nell_inverse_domain_prior_gamma", 0.5))
        if gamma <= 0.0:
            return score

        prior_by_relation = cls._get_nell_inverse_domain_prior(batch["graph"], config, score.device)
        if not prior_by_relation:
            return score

        adjusted_score = score.clone()
        relation_index = batch["r_index"].detach().cpu().tolist()
        for batch_index, relation_id in enumerate(relation_index):
            prior = prior_by_relation.get(int(relation_id))
            if prior is None:
                continue
            adjusted_score[batch_index] = adjusted_score[batch_index] + gamma * prior
        return adjusted_score

    @classmethod
    def _apply_innovation4_tail_calibration(
        cls,
        score: torch.Tensor,
        batch: dict[str, torch.Tensor],
        config,
        split_name: str | None = None,
    ) -> torch.Tensor:
        if config is None or not getattr(config.model, "innovation4_tail_calibration_enable", False):
            return score
        if (
            getattr(config.model, "innovation4_tail_calibration_test_only", True)
            and split_name != "test"
        ):
            return score

        adjusted_score = score
        if getattr(config.model, "innovation4_hub_suppression_enable", False):
            gamma = float(getattr(config.model, "innovation4_hub_suppression_gamma", 1.0))
            if gamma > 0.0:
                feature_name = getattr(
                    config.model,
                    "innovation4_hub_suppression_feature",
                    "derivational_out",
                )
                graph = batch["graph"]
                hypernym_ids = cls._hypernym_relation_ids(graph)
                feature = cls._get_hypernym_hub_feature(graph, feature_name, score.device)
                if hypernym_ids and feature is not None:
                    relation_index = batch["r_index"].detach().cpu()
                    query_mask = torch.tensor(
                        [
                            int(relation_id) in hypernym_ids
                            for relation_id in relation_index.tolist()
                        ],
                        dtype=torch.bool,
                        device=score.device,
                    )
                    if bool(query_mask.any()):
                        if adjusted_score is score:
                            adjusted_score = score.clone()
                        adjusted_score[query_mask] = (
                            adjusted_score[query_mask] - gamma * feature.unsqueeze(0)
                        )

        if getattr(config.model, "innovation4_domain_boost_enable", False):
            gamma = float(getattr(config.model, "innovation4_domain_boost_gamma", 0.5))
            if gamma > 0.0:
                prior_by_relation = cls._get_nell_inverse_domain_prior(
                    batch["graph"],
                    config,
                    score.device,
                )
                if prior_by_relation:
                    if adjusted_score is score:
                        adjusted_score = score.clone()
                    relation_index = batch["r_index"].detach().cpu().tolist()
                    for batch_index, relation_id in enumerate(relation_index):
                        prior = prior_by_relation.get(int(relation_id))
                        if prior is None:
                            continue
                        adjusted_score[batch_index] = adjusted_score[batch_index] + gamma * prior

        return adjusted_score

    @classmethod
    def _apply_hypernym_path_prior(
        cls,
        score: torch.Tensor,
        batch: dict[str, torch.Tensor],
        config,
    ) -> torch.Tensor:
        if config is None or not getattr(config.model, "hypernym_path_prior_enable", False):
            return score

        graph = batch["graph"]
        relation_ids = cls._hypernym_relation_ids(graph)
        if not relation_ids:
            return score

        max_hop = int(getattr(config.model, "hypernym_path_prior_max_hop", 3))
        gamma = float(getattr(config.model, "hypernym_path_prior_gamma", 0.5))
        if max_hop < 1 or gamma <= 0.0:
            return score

        adjusted_score = score.clone()
        adjacency_cache: dict[int, dict[int, list[int]]] = {}
        h_index = batch["h_index"].detach().cpu().tolist()
        r_index = batch["r_index"].detach().cpu().tolist()

        for batch_index, (head_id, relation_id) in enumerate(zip(h_index, r_index)):
            relation_id = int(relation_id)
            if relation_id not in relation_ids:
                continue
            adjacency = adjacency_cache.get(relation_id)
            if adjacency is None:
                adjacency = cls._build_relation_adjacency(graph, relation_id)
                adjacency_cache[relation_id] = adjacency

            visited = {int(head_id)}
            frontier = [int(head_id)]
            for hop in range(1, max_hop + 1):
                next_frontier: list[int] = []
                prior_value = gamma / float(hop)
                for node_id in frontier:
                    for target_id in adjacency.get(node_id, []):
                        if target_id in visited:
                            continue
                        visited.add(target_id)
                        next_frontier.append(target_id)
                        adjusted_score[batch_index, target_id] = (
                            adjusted_score[batch_index, target_id] + prior_value
                        )
                frontier = next_frontier
                if not frontier:
                    break
        return adjusted_score

    def training_step(self, model, batch, config, device):
        batch = move_batch_to_device(batch, device)
        if config.data.remove_target_edges:
            batch = self._remove_target_edges(batch, config.data.remove_all_one_hop_edges)
        batch = self._apply_local_subgraph_mask(
            batch,
            apply_mask=config.data.local_subgraph_apply_mask,
        )
        score_output = model(batch)
        score_fine, score_coarse = self._split_model_output(score_output)
        loss = self._compute_loss(score_fine, batch, config)
        loss = loss + self._compute_hypernym_hard_negative_loss(score_fine, batch, config)
        if (
            score_coarse is not None
            and config.model.internal_coarse_to_fine_enable
            and config.model.internal_coarse_use_joint_loss
            and config.model.internal_coarse_loss_weight > 0.0
        ):
            coarse_loss = self._compute_loss(score_coarse, batch, config)
            loss = loss + config.model.internal_coarse_loss_weight * coarse_loss
        # 普通深度监督对照（意见1）：各中间层辅助头与细头共用同一BCE与负采样，等权平均
        aux_scores = score_output.get("aux_scores") if isinstance(score_output, dict) else None
        if aux_scores is not None and bool(getattr(config.model, "deep_supervision_enable", False)):
            aux_loss = torch.stack(
                [self._compute_loss(s, batch, config) for s in aux_scores]
            ).mean()
            loss = loss + aux_loss
        return loss

    def evaluate(self, model, dataloader, device, config=None, split_name=None):
        accumulator = RankingMetricAccumulator()
        relation_accumulators: dict[int, RankingMetricAccumulator] = {}
        relation_names: dict[int, str] = {}
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                batch = self._apply_local_subgraph_mask(
                    batch,
                    apply_mask=config.data.local_subgraph_apply_mask if config is not None else True,
                )
                filter_mask = batch["filter_mask"].bool()
                score_output = model(batch)
                score_fine, score_coarse = self._split_model_output(score_output)
                score_fine = self._apply_hypernym_path_prior(score_fine, batch, config)
                if (
                    config is not None
                    and getattr(config.model, "innovation4_tail_calibration_enable", False)
                ):
                    score_fine = self._apply_innovation4_tail_calibration(
                        score_fine,
                        batch,
                        config,
                        split_name,
                    )
                else:
                    score_fine = self._apply_hypernym_hub_calibration(score_fine, batch, config)
                    score_fine = self._apply_nell_inverse_domain_prior(
                        score_fine,
                        batch,
                        config,
                        split_name,
                    )
                if (
                    score_coarse is not None
                    and config is not None
                    and getattr(config.model, "hypernym_hub_calibration_apply_coarse", False)
                ):
                    score_coarse = self._apply_hypernym_hub_calibration(score_coarse, batch, config)
                if (
                    config is not None
                    and score_coarse is not None
                    and config.model.internal_coarse_to_fine_enable
                    and config.model.internal_coarse_use_delta_decision
                ):
                    ranks = self._compute_coarse_to_fine_ranks(
                        score_fine,
                        score_coarse,
                        filter_mask,
                        batch["t_index"],
                        config.model.internal_coarse_topk,
                        config.model.internal_coarse_delta,
                    )
                else:
                    answer_score = score_fine.gather(1, batch["t_index"].unsqueeze(1))
                    ranks = ((score_fine >= answer_score) & (~filter_mask)).sum(dim=1) + 1
                accumulator.update(ranks)
                id2relation = getattr(batch["graph"], "id2relation", None) or {}
                relation_index = batch["r_index"].detach().cpu()
                for relation_id_tensor in relation_index.unique(sorted=True):
                    relation_id = int(relation_id_tensor.item())
                    relation_mask = batch["r_index"] == relation_id
                    relation_accumulator = relation_accumulators.setdefault(
                        relation_id,
                        RankingMetricAccumulator(),
                    )
                    relation_accumulator.update(ranks[relation_mask])
                    relation_names[relation_id] = id2relation.get(
                        relation_id,
                        str(relation_id),
                    )

        metrics = accumulator.compute()
        metrics.per_relation = {
            relation_names.get(relation_id, str(relation_id)): {
                **relation_accumulator.compute().to_dict(),
                "count": float(relation_accumulator.total),
            }
            for relation_id, relation_accumulator in sorted(relation_accumulators.items())
        }
        return metrics


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to") and key == "graph":
            moved[key] = value.to(device)
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@register_task(TASK_TRANSDUCTIVE)
class TransductiveReasoningTask(LinkPredictionTask):
    task_name = TASK_TRANSDUCTIVE


@register_task(TASK_INDUCTIVE)
class InductiveReasoningTask(LinkPredictionTask):
    task_name = TASK_INDUCTIVE


@register_task(TASK_LARGE_SCALE)
class LargeScaleKGTask(LinkPredictionTask):
    task_name = TASK_LARGE_SCALE


@register_task(TASK_TRIPLE_CLASSIFICATION)
class TripleClassificationTask(BaseTask):
    task_name = TASK_TRIPLE_CLASSIFICATION
    primary_metric = "accuracy"

    def __init__(self) -> None:
        super().__init__()
        self.selected_threshold = 0.5
        self.last_eval_threshold = 0.5

    def build_data_module(self, config, dataset_spec):
        use_local_subgraph = (
            config.model.triple_classification_encoder_mode == TRIPLE_CLASSIFICATION_ENCODER_LOCAL_SUBGRAPH
        )
        return build_data_module(
            root=config.data.root,
            dataset_spec=dataset_spec,
            task_type=self.task_name,
            batch_size=config.data.batch_size,
            eval_batch_size=config.data.eval_batch_size,
            num_workers=config.runtime.num_workers,
            use_test_as_validation=config.data.use_test_as_validation,
            enable_topology_prior=False,
            enable_local_subgraph=use_local_subgraph,
            local_subgraph_cache=config.data.local_subgraph_cache,
            local_subgraph_hops=config.data.local_subgraph_hops,
            local_subgraph_topk_1hop=config.data.local_subgraph_topk_1hop,
            local_subgraph_topk_2hop=config.data.local_subgraph_topk_2hop,
        )

    def training_step(self, model, batch, config, device):
        batch = move_batch_to_device(batch, device)
        score_output = model(batch)
        if isinstance(score_output, dict) and "triple_score" in score_output:
            return F.binary_cross_entropy_with_logits(score_output["triple_score"], batch["label"].float())

        score_fine, score_coarse = LinkPredictionTask._split_model_output(score_output)
        triple_score = score_fine.gather(1, batch["t_index"].unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(triple_score, batch["label"].float())
        if (
            score_coarse is not None
            and config.model.internal_coarse_to_fine_enable
            and config.model.internal_coarse_use_joint_loss
            and config.model.internal_coarse_loss_weight > 0.0
        ):
            coarse_score = score_coarse.gather(1, batch["t_index"].unsqueeze(1)).squeeze(1)
            coarse_loss = F.binary_cross_entropy_with_logits(coarse_score, batch["label"].float())
            loss = loss + config.model.internal_coarse_loss_weight * coarse_loss
        return loss

    @staticmethod
    def _compute_classification_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
    ) -> ClassificationMetrics:
        prediction = (torch.sigmoid(logits) >= threshold).long()
        label = labels.long()

        correct = (prediction == label).sum().item()
        total = label.numel()
        true_positive = ((prediction == 1) & (label == 1)).sum().item()
        false_positive = ((prediction == 1) & (label == 0)).sum().item()
        false_negative = ((prediction == 0) & (label == 1)).sum().item()

        accuracy = correct / total if total else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=accuracy,
            hits1=accuracy,
            hits10=f1,
        )

    @classmethod
    def _select_best_threshold(
        cls,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, ClassificationMetrics]:
        probabilities = torch.sigmoid(logits)
        candidate_thresholds = torch.unique(probabilities.detach().cpu())
        candidate_thresholds = torch.cat(
            [
                candidate_thresholds,
                torch.tensor([0.5], dtype=candidate_thresholds.dtype),
            ]
        ).unique(sorted=True)

        best_threshold = 0.5
        best_metrics = cls._compute_classification_metrics(logits, labels, best_threshold)
        best_key = (best_metrics.accuracy, best_metrics.f1, best_metrics.precision, best_metrics.recall)

        for threshold_tensor in candidate_thresholds:
            threshold = float(threshold_tensor.item())
            metrics = cls._compute_classification_metrics(logits, labels, threshold)
            current_key = (metrics.accuracy, metrics.f1, metrics.precision, metrics.recall)
            if current_key > best_key:
                best_threshold = threshold
                best_metrics = metrics
                best_key = current_key

        return best_threshold, best_metrics

    def evaluate(self, model, dataloader, device, config=None, split_name=None):
        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                score_output = model(batch)
                if isinstance(score_output, dict) and "triple_score" in score_output:
                    logits = score_output["triple_score"]
                else:
                    score_fine, _ = LinkPredictionTask._split_model_output(score_output)
                    logits = score_fine.gather(1, batch["t_index"].unsqueeze(1)).squeeze(1)
                logits_list.append(logits.detach().cpu())
                labels_list.append(batch["label"].detach().cpu())

        if not logits_list:
            self.last_eval_threshold = self.selected_threshold
            return self._compute_classification_metrics(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                self.selected_threshold,
            )

        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        if split_name == "valid":
            self.selected_threshold, metrics = self._select_best_threshold(all_logits, all_labels)
            self.last_eval_threshold = self.selected_threshold
            return metrics

        self.last_eval_threshold = self.selected_threshold
        return self._compute_classification_metrics(all_logits, all_labels, self.selected_threshold)
