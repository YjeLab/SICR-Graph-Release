from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from Our.rspmm import generalized_rspmm
from Utils.config import (
    INTERNAL_COARSE_MODE_PRO,
    INNOVATION2_MODE_SPARSE_ROUTE,
    TASK_INDUCTIVE,
    TASK_LARGE_SCALE,
    TASK_TRIPLE_CLASSIFICATION,
    TASK_TRANSDUCTIVE,
    DuetGraphConfig,
)
from Utils.data import DatasetInfo, Graph
from Utils.interfaces import BaseReasoningModel
from Utils.profiling import FusionPathProfiler, synchronize_for_timing
from Utils.registry import register_model


class DualFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.fc2(x)


class RelationMessagePassing(nn.Module):
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int,
        *,
        use_relation_guided_routing: bool = False,
        routing_hidden_dim: int = 32,
        routing_floor: float = 0.25,
        routing_bias_init: float = 2.0,
        routing_match_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.use_relation_guided_routing = use_relation_guided_routing
        self.routing_floor = max(0.0, min(float(routing_floor), 1.0))
        self.relation_projection = nn.Linear(hidden_dim, hidden_dim * num_relations)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta = nn.Parameter(torch.empty(1, hidden_dim))
        self.eps = nn.Parameter(torch.tensor([0.0]))
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.beta)
        if self.use_relation_guided_routing:
            self.routing_query_projection = nn.Linear(hidden_dim, routing_hidden_dim)
            self.routing_relation_embedding = nn.Embedding(num_relations, routing_hidden_dim)
            nn.init.normal_(self.routing_relation_embedding.weight, std=0.02)
            self.routing_bias = nn.Parameter(torch.tensor([routing_bias_init], dtype=torch.float32))
            self.routing_match_bias = nn.Parameter(torch.tensor([routing_match_bias], dtype=torch.float32))

    def _compute_relation_routing_gate(
        self,
        query: torch.Tensor,
        query_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        query_context = F.normalize(self.routing_query_projection(query), dim=-1)
        relation_context = F.normalize(self.routing_relation_embedding.weight, dim=-1)
        learned_score = torch.matmul(query_context, relation_context.transpose(0, 1))
        exact_match = F.one_hot(
            query_relation_index.long(),
            num_classes=self.num_relations,
        ).to(dtype=query.dtype, device=query.device)
        routing_score = learned_score + self.routing_bias + self.routing_match_bias * exact_match
        return self.routing_floor + (1.0 - self.routing_floor) * torch.sigmoid(routing_score)

    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        query_relation_index: torch.Tensor,
        graph: Graph,
        graph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        relation = self.relation_projection(query).view(batch_size, self.num_relations, self.hidden_dim)
        if self.use_relation_guided_routing:
            routing_gate = self._compute_relation_routing_gate(query, query_relation_index).unsqueeze(-1)
            relation = relation * routing_gate
        edge_index = graph.edge_index if graph_mask is None else graph.edge_index[graph_mask]
        node_to_node = edge_index[:, [0, 2]].transpose(0, 1).contiguous()
        relation_type = edge_index[:, 1].contiguous()
        edge_weight = torch.ones(edge_index.size(0), device=x.device, dtype=x.dtype)

        split_relation = relation.permute(1, 0, 2).reshape(self.num_relations, batch_size * self.hidden_dim)
        split_input = x.permute(1, 0, 2).reshape(x.size(1), batch_size * self.hidden_dim)
        output = generalized_rspmm(
            node_to_node,
            relation_type,
            edge_weight,
            relation=split_relation.float(),
            input=split_input.float(),
        )
        output = output.view(x.size(1), batch_size, self.hidden_dim).permute(1, 0, 2)

        shortcut = x
        x = self.output_projection(output + self.beta * x + self.eps)
        x = self.norm(x)
        return x + shortcut


class DualPathFusionMixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        gate_hidden_dim: int,
        gate_cap: float,
        use_query_aware_fusion: bool,
    ) -> None:
        super().__init__()
        self.use_query_aware_fusion = use_query_aware_fusion
        self.scalar_fusion_logit = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.gate_cap = max(0.0, min(float(gate_cap), 1.0))
        self.last_alpha: torch.Tensor | None = None
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        query: torch.Tensor,
        head_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_query_aware_fusion:
            scalar_alpha = torch.sigmoid(self.scalar_fusion_logit)
            self.last_alpha = scalar_alpha.detach()
            return scalar_alpha * local + (1.0 - scalar_alpha) * global_view

        query_context = torch.cat([query, head_state], dim=-1).unsqueeze(1).expand(-1, local.size(1), -1)
        gate_input = torch.cat([local, global_view, query_context], dim=-1)
        alpha = self.gate_cap * torch.sigmoid(self.gate_network(gate_input))
        self.last_alpha = alpha.detach()
        return alpha * local + (1.0 - alpha) * global_view


class DualFusionLayer(nn.Module):
    def __init__(
        self,
        num_relations: int,
        num_v_layer: int,
        hidden_dim: int,
        num_heads: int,
        drop: float,
        gate_hidden_dim: int,
        gate_cap: float,
        use_query_aware_fusion: bool,
        use_relation_guided_routing: bool = False,
        routing_hidden_dim: int = 32,
        routing_floor: float = 0.25,
        routing_bias_init: float = 2.0,
        routing_match_bias: float = 1.0,
        profiler: FusionPathProfiler | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.profiler = profiler

        self.message_layers = nn.ModuleList(
            [
                RelationMessagePassing(
                    num_relations,
                    hidden_dim,
                    use_relation_guided_routing=use_relation_guided_routing,
                    routing_hidden_dim=routing_hidden_dim,
                    routing_floor=routing_floor,
                    routing_bias_init=routing_bias_init,
                    routing_match_bias=routing_match_bias,
                )
                for _ in range(num_v_layer)
            ]
        )
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = DualFeedForward(hidden_dim, drop)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mixer = DualPathFusionMixer(
            hidden_dim=hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            gate_cap=gate_cap,
            use_query_aware_fusion=use_query_aware_fusion,
        )

    def _linear_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads
        q = self.attn_q(x).view(batch_size, num_nodes, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.attn_k(x).view(batch_size, num_nodes, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.attn_v(x).view(batch_size, num_nodes, self.num_heads, head_dim).permute(0, 2, 1, 3)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        numerator = torch.einsum("bhnd,bhde->bhne", q, kv)
        numerator = numerator + v.sum(dim=2, keepdim=True) + v * num_nodes

        denominator = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))
        denominator = denominator.unsqueeze(-1) + (2 * num_nodes)

        attended = numerator / denominator
        attended = attended.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.hidden_dim)
        x = self.attn_norm(x + attended)
        return self.ffn_norm(x + self.ffn(x))

    def _global_path(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_attention(x)

    def forward_global_only(self, x: torch.Tensor) -> torch.Tensor:
        return self._global_path(x)

    def _message_passing(
        self,
        h_index: torch.Tensor,
        r_index: torch.Tensor,
        x: torch.Tensor,
        query: torch.Tensor,
        graph: Graph,
        graph_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        seed = torch.zeros(batch_size, graph.num_nodes, self.hidden_dim, device=x.device)
        seed[torch.arange(batch_size, device=x.device), h_index] = 1
        x = self.input_projection(torch.cat([x, seed], dim=-1))
        for layer in self.message_layers:
            x = layer(
                x,
                query,
                r_index,
                graph,
                graph_mask,
            )
        return x

    def forward(
        self,
        h_index: torch.Tensor,
        r_index: torch.Tensor,
        x: torch.Tensor,
        query: torch.Tensor,
        graph: Graph,
        graph_mask: torch.Tensor | None,
        profile_enabled: bool = False,
    ) -> torch.Tensor:
        local_ms = 0.0
        global_ms = 0.0
        gate_ms = 0.0
        batch_size = x.size(0)
        head_state = x[torch.arange(batch_size, device=x.device), h_index]
        if profile_enabled:
            synchronize_for_timing(x.device)
            start_time = time.perf_counter()
        local = self._message_passing(
            h_index,
            r_index,
            x,
            query,
            graph,
            graph_mask,
        )
        if profile_enabled:
            synchronize_for_timing(x.device)
            local_ms = (time.perf_counter() - start_time) * 1000.0
            start_time = time.perf_counter()
        global_view = self._global_path(x)
        if profile_enabled:
            synchronize_for_timing(x.device)
            global_ms = (time.perf_counter() - start_time) * 1000.0
            start_time = time.perf_counter()
        fused = self.fusion_mixer(local, global_view, query, head_state)

        if profile_enabled and self.profiler is not None:
            synchronize_for_timing(x.device)
            gate_ms = (time.perf_counter() - start_time) * 1000.0
            self.profiler.record_invocation(
                local_ms=local_ms,
                global_ms=global_ms,
                gate_ms=gate_ms,
            )
        return fused


@register_model("our_duetgraph")
class OurDuetGraphModel(BaseReasoningModel):
    model_name = "our_duetgraph"
    supported_tasks = {
        TASK_TRANSDUCTIVE,
        TASK_INDUCTIVE,
        TASK_LARGE_SCALE,
        TASK_TRIPLE_CLASSIFICATION,
    }

    def __init__(self, dataset_info: DatasetInfo, model_config: DuetGraphConfig) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.config = model_config
        self.hidden_dim = model_config.hidden_dim
        self.fusion_profiler = FusionPathProfiler()
        self.use_innovation1 = getattr(model_config, "use_innovation1", True)
        self.use_sparse_route = (
            getattr(model_config, "innovation2_resolved_mode", "off")
            == INNOVATION2_MODE_SPARSE_ROUTE
        )

        self.query_embedding = nn.Embedding(dataset_info.num_relations, model_config.hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
        )
        self.bootstrap_layers = self._build_message_layers(
            dataset_info.num_relations,
            model_config,
            use_relation_guided_routing=self.use_sparse_route,
        )

        self.fusion_layers = nn.ModuleList(
            [
                self._build_fusion_layer(
                    dataset_info.num_relations,
                    model_config,
                    use_relation_guided_routing=self.use_sparse_route,
                )
                for _ in range(model_config.num_layer)
            ]
        )
        self.output_projection = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, 1),
        )
        self.triple_classifier = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.drop),
            nn.Linear(model_config.hidden_dim, 1),
        )
        if model_config.internal_coarse_to_fine_enable:
            self.coarse_projection = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)
            self.coarse_output_projection = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)
            self.coarse_scorer = nn.Sequential(
                nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
                nn.ReLU(),
                nn.Linear(model_config.hidden_dim, 1),
            )
        else:
            self.coarse_projection = None
            self.coarse_output_projection = None
            self.coarse_scorer = None

        # 普通深度监督对照（意见1）：深层融合各中间层的同质辅助评分头，仅训练期使用
        if getattr(model_config, "deep_supervision_enable", False):
            self.aux_scorers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(model_config.hidden_dim, 1),
                    )
                    for _ in range(max(model_config.num_layer - 1, 0))
                ]
            )
        else:
            self.aux_scorers = None

        if dataset_info.use_entity_embeddings:
            self.entity_embedding = nn.Embedding(
                dataset_info.train_num_entities,
                model_config.hidden_dim,
            )
        else:
            self.entity_embedding = None

    @staticmethod
    def _build_message_layers(
        num_relations: int,
        model_config: DuetGraphConfig,
        *,
        use_relation_guided_routing: bool,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                RelationMessagePassing(
                    num_relations,
                    model_config.hidden_dim,
                    use_relation_guided_routing=use_relation_guided_routing,
                    routing_hidden_dim=model_config.innovation2_routing_hidden_dim,
                    routing_floor=model_config.innovation2_routing_floor,
                    routing_bias_init=model_config.innovation2_routing_bias_init,
                    routing_match_bias=model_config.innovation2_routing_match_bias,
                )
                for _ in range(model_config.num_v_layer)
            ]
        )

    def _build_fusion_layer(
        self,
        num_relations: int,
        model_config: DuetGraphConfig,
        *,
        use_relation_guided_routing: bool,
    ) -> DualFusionLayer:
        return DualFusionLayer(
            num_relations,
            model_config.num_v_layer,
            model_config.hidden_dim,
            model_config.num_heads,
            model_config.drop,
            model_config.topology_gate_hidden_dim,
            model_config.topology_gate_cap,
            self.use_innovation1,
            use_relation_guided_routing=use_relation_guided_routing,
            routing_hidden_dim=model_config.innovation2_routing_hidden_dim,
            routing_floor=model_config.innovation2_routing_floor,
            routing_bias_init=model_config.innovation2_routing_bias_init,
            routing_match_bias=model_config.innovation2_routing_match_bias,
            profiler=self.fusion_profiler,
        )

    def _initial_features(self, batch_size: int, graph: Graph, device: torch.device) -> torch.Tensor:
        if self.entity_embedding is not None:
            if graph.node_ids is not None:
                base_embedding = self.entity_embedding(graph.node_ids.to(device))
                return base_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            if graph.num_nodes == self.entity_embedding.num_embeddings:
                return self.entity_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.zeros(batch_size, graph.num_nodes, self.hidden_dim, device=device)

    def _compute_coarse_score(
        self,
        shared_x: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.coarse_projection is None or self.coarse_scorer is None:
            return None
        coarse_hidden = self.coarse_projection(shared_x)
        if self.config.internal_coarse_mode == INTERNAL_COARSE_MODE_PRO:
            for layer in self.fusion_layers:
                coarse_hidden = layer.forward_global_only(coarse_hidden)
        if self.coarse_output_projection is not None:
            coarse_hidden = self.coarse_output_projection(coarse_hidden)
        return self.coarse_scorer(coarse_hidden).squeeze(-1)

    def _encode_graph(
        self,
        batch: dict[str, torch.Tensor | Graph],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        graph = batch["graph"]
        h_index = batch["h_index"]
        r_index = batch["r_index"]
        graph_mask = batch.get("graph_mask")

        if not isinstance(graph, Graph):
            raise TypeError("batch['graph'] must be a Graph instance.")

        batch_size = h_index.size(0)
        query = self.query_embedding(r_index)
        x = self._initial_features(batch_size, graph, h_index.device)

        seed = torch.zeros(batch_size, graph.num_nodes, self.hidden_dim, device=h_index.device)
        seed[torch.arange(batch_size, device=h_index.device), h_index] = 1
        x = self.input_projection(torch.cat([x, seed], dim=-1))

        for layer in self.bootstrap_layers:
            x = layer(
                x,
                query,
                r_index,
                graph,
                graph_mask,
            )
        coarse_base_x = x

        profile_enabled = self.fusion_profiler.begin_forward()
        coarse_score = self._compute_coarse_score(shared_x=coarse_base_x)
        aux_scores: list[torch.Tensor] | None = None
        if self.aux_scorers is not None and self.training:
            aux_scores = []
        for layer_idx, layer in enumerate(self.fusion_layers):
            x = layer(
                h_index,
                r_index,
                x,
                query,
                graph,
                graph_mask,
                profile_enabled=profile_enabled,
            )
            if aux_scores is not None and layer_idx < len(self.aux_scorers):
                aux_scores.append(self.aux_scorers[layer_idx](x).squeeze(-1))

        x = self.output_projection(x)
        return x, query, coarse_score, aux_scores

    def _forward_triple_classification(self, batch: dict[str, torch.Tensor | Graph]) -> dict[str, torch.Tensor]:
        if self.entity_embedding is None:
            raise RuntimeError("Triple classification requires entity embeddings.")

        h_index = batch["h_index"]
        r_index = batch["r_index"]
        t_index = batch["t_index"]

        x, query, _, _ = self._encode_graph(batch)
        batch_arange = torch.arange(h_index.size(0), device=h_index.device)
        head = x[batch_arange, h_index]
        relation = query
        tail = x[batch_arange, t_index]
        feature = torch.cat([head, relation, tail, head * relation * tail], dim=-1)
        return {"triple_score": self.triple_classifier(feature).squeeze(-1)}

    def forward(self, batch: dict[str, torch.Tensor | Graph]) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.dataset_info.task_type == TASK_TRIPLE_CLASSIFICATION:
            return self._forward_triple_classification(batch)

        x, _, coarse_score, aux_scores = self._encode_graph(batch)
        fine_score = self.scorer(x).squeeze(-1)
        if coarse_score is None:
            if aux_scores is not None:
                return {"score_fine": fine_score, "aux_scores": torch.stack(aux_scores)}
            return fine_score
        result = {
            "score_fine": fine_score,
            "score_coarse": coarse_score,
        }
        if aux_scores is not None:
            result["aux_scores"] = torch.stack(aux_scores)
        return result

    def configure_fusion_profiling(self, enabled: bool, max_forwards: int) -> None:
        self.fusion_profiler.configure(enabled=enabled, max_forwards=max_forwards)

    def reset_fusion_profile(self) -> None:
        self.fusion_profiler.reset()

    def get_fusion_profile_summary(self):
        return self.fusion_profiler.summary()

    def get_last_fusion_alphas(self) -> list[torch.Tensor]:
        alphas: list[torch.Tensor] = []
        for layer in self.fusion_layers:
            alpha = layer.fusion_mixer.last_alpha
            if alpha is not None:
                alphas.append(alpha)
        return alphas

    def describe_global_path(self) -> str:
        return "mode=linear_attention"
