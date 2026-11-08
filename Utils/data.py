from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from Utils.config import TASK_INDUCTIVE, TASK_TRIPLE_CLASSIFICATION


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
    train_graph_num_edges: int = 0
    eval_graph_num_edges: int = 0
    train_topology_prior_status: str = "disabled"
    eval_topology_prior_status: str = "disabled"
    train_local_subgraph_status: str = "disabled"
    eval_local_subgraph_status: str = "disabled"
    train_local_subgraph_avg_edges: float = 0.0
    eval_local_subgraph_avg_edges: float = 0.0
    train_local_subgraph_max_edges: int = 0
    eval_local_subgraph_max_edges: int = 0
    triple_subgraph_template_status: str = "disabled"


@dataclass
class EvalCollator:
    knowledge_graph: object
    split: str

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor | Graph]:
        return self.knowledge_graph.eval_collate_fn(batch, self.split)


class _CollatedBatchDataset:
    def __init__(self, triplets: torch.Tensor, batch_size: int, collate_fn) -> None:
        self.triplets = triplets
        self.batch_size = max(1, int(batch_size))
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return (int(self.triplets.size(0)) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | Graph]:
        start = int(index) * self.batch_size
        end = min(start + self.batch_size, int(self.triplets.size(0)))
        return self.collate_fn([item for item in self.triplets[start:end]])


def _read_mapping_file(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    item_to_id: dict[str, int] = {}
    id_to_item: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item, idx = line.strip().split("\t")
            idx_int = int(idx)
            item_to_id[item] = idx_int
            id_to_item[idx_int] = item
    return item_to_id, id_to_item


def _read_triplets(path: Path) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            head, relation, tail = line.strip().split("\t")
            triplets.append((head, relation, tail))
    return triplets


def _read_labels(path: Path) -> torch.Tensor:
    labels: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            value = line.strip()
            if value == "1":
                labels.append(1.0)
            elif value == "-1":
                labels.append(0.0)
            else:
                raise ValueError(f"{path}:{line_no} expected label 1 or -1, got {value!r}")
    return torch.tensor(labels, dtype=torch.float32)


def _encode_triplets(
    raw_triplets: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    encoded: list[tuple[int, int, int]] = []
    for head, relation, tail in raw_triplets:
        h_id = entity2id[head]
        r_id = relation2id[relation]
        t_id = entity2id[tail]
        rev_r_id = relation2id[f"-{relation}"]
        encoded.append((h_id, r_id, t_id))
        encoded.append((t_id, rev_r_id, h_id))
    return encoded


def _encode_triplets_without_inverse(
    raw_triplets: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    return [
        (entity2id[head], relation2id[relation], entity2id[tail])
        for head, relation, tail in raw_triplets
    ]


def _forward_relation_count(relation2id: dict[str, int]) -> int:
    return sum(1 for relation in relation2id if not relation.startswith("-"))


def _build_filters(triplets: list[tuple[int, int, int]]) -> defaultdict[tuple[int, int], set[int]]:
    filters: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for head, relation, tail in triplets:
        filters[(head, relation)].add(tail)
    return filters


def _compute_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    damping: float,
    num_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.zeros(0, dtype=torch.float32)

    source = edge_index[:, 0].long()
    target = edge_index[:, 2].long()
    out_degree = torch.bincount(source, minlength=num_nodes).float()
    pagerank = torch.full((num_nodes,), 1.0 / num_nodes, dtype=torch.float32)
    dangling_mask = out_degree == 0
    teleport = (1.0 - damping) / num_nodes

    for _ in range(num_iterations):
        updated = torch.full_like(pagerank, teleport)
        if source.numel() > 0:
            contribution = pagerank[source] / out_degree[source].clamp_min(1.0)
            updated.index_add_(0, target, damping * contribution)
        dangling_mass = pagerank[dangling_mask].sum()
        if dangling_mass.item() > 0:
            updated += damping * dangling_mass / num_nodes
        if torch.max(torch.abs(updated - pagerank)).item() < tolerance:
            pagerank = updated
            break
        pagerank = updated

    return pagerank.clamp_min(1e-12)


def _compute_relation_entropy(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_relations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = edge_index[:, 0].long()
    relation = edge_index[:, 1].long()
    degree = torch.bincount(source, minlength=num_nodes).float()
    relation_counts = torch.zeros((num_nodes, num_relations), dtype=torch.float32)

    if source.numel() > 0:
        relation_counts.index_put_(
            (source, relation),
            torch.ones(source.size(0), dtype=torch.float32),
            accumulate=True,
        )

    probability = relation_counts / degree.clamp_min(1.0).unsqueeze(-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    return degree, entropy


def _compute_topology_prior(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    damping: float,
    num_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    degree, entropy = _compute_relation_entropy(edge_index, num_nodes, num_relations)
    pagerank = _compute_pagerank(edge_index, num_nodes, damping, num_iterations, tolerance)
    return torch.stack(
        [
            torch.log1p(degree),
            entropy,
            torch.log(pagerank),
        ],
        dim=-1,
    )


def _load_cached_topology_prior(
    cache_path: Path,
    num_nodes: int,
    num_relations: int,
    num_edges: int,
) -> torch.Tensor | None:
    if not cache_path.exists():
        return None
    if cache_path.stat().st_size == 0:
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        node_prior = payload.get("node_prior")
        cached_nodes = payload.get("num_nodes")
        cached_relations = payload.get("num_relations")
        cached_edges = payload.get("num_edges")
        if (
            isinstance(node_prior, torch.Tensor)
            and cached_nodes == num_nodes
            and cached_relations == num_relations
            and cached_edges == num_edges
            and node_prior.shape == (num_nodes, 3)
        ):
            return node_prior.float()
        return None

    if isinstance(payload, torch.Tensor) and payload.shape == (num_nodes, 3):
        return payload.float()
    return None


def _get_or_compute_topology_prior(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    cache_path: Path | None,
    use_cache: bool,
    damping: float,
    num_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, str]:
    if use_cache and cache_path is not None:
        cached = _load_cached_topology_prior(
            cache_path,
            num_nodes=num_nodes,
            num_relations=num_relations,
            num_edges=edge_index.size(0),
        )
        if cached is not None:
            return cached, "loaded"

    node_prior = _compute_topology_prior(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_relations=num_relations,
        damping=damping,
        num_iterations=num_iterations,
        tolerance=tolerance,
    )

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "node_prior": node_prior,
                "num_nodes": num_nodes,
                "num_relations": num_relations,
                "num_edges": edge_index.size(0),
            },
            cache_path,
        )

    return node_prior, "computed"


def _select_topk_edge_ids(
    edge_ids: torch.Tensor,
    edge_priority: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if edge_ids.numel() == 0 or topk <= 0:
        return edge_ids.new_empty(0)
    if edge_ids.numel() <= topk:
        return edge_ids
    priority = edge_priority[edge_ids]
    selected = priority.topk(topk, largest=True).indices
    return edge_ids[selected]


def _compute_local_edge_priority(
    edge_index: torch.Tensor,
    node_prior: torch.Tensor | None,
) -> torch.Tensor:
    if node_prior is None:
        return torch.ones(edge_index.size(0), dtype=torch.float32)

    target_index = edge_index[:, 2].long()
    log_degree = node_prior[target_index, 0]
    relation_entropy = node_prior[target_index, 1]
    log_pagerank = node_prior[target_index, 2]
    return log_pagerank + (0.25 * relation_entropy) - (0.10 * log_degree)


def _build_outgoing_edge_ids(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    outgoing_edge_ids: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_id, source_index in enumerate(edge_index[:, 0].tolist()):
        outgoing_edge_ids[source_index].append(edge_id)
    return [
        torch.tensor(edge_ids, dtype=torch.long) if edge_ids else torch.empty(0, dtype=torch.long)
        for edge_ids in outgoing_edge_ids
    ]


def _compute_local_subgraph_edge_pool(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_prior: torch.Tensor | None,
    topk_1hop: int,
    topk_2hop: int,
    hops: int,
) -> list[torch.Tensor]:
    outgoing_edge_ids = _build_outgoing_edge_ids(edge_index, num_nodes)
    edge_priority = _compute_local_edge_priority(edge_index, node_prior)
    local_subgraph_edge_pool: list[torch.Tensor] = []

    for node_index in range(num_nodes):
        selected_edge_groups: list[torch.Tensor] = []
        first_hop_edge_ids = _select_topk_edge_ids(
            outgoing_edge_ids[node_index],
            edge_priority,
            topk_1hop,
        )
        if first_hop_edge_ids.numel() > 0:
            selected_edge_groups.append(first_hop_edge_ids)

        if hops >= 2 and topk_2hop > 0 and first_hop_edge_ids.numel() > 0:
            first_hop_targets = edge_index[first_hop_edge_ids, 2].unique(sorted=False)
            for intermediate_node in first_hop_targets.tolist():
                second_hop_edge_ids = _select_topk_edge_ids(
                    outgoing_edge_ids[intermediate_node],
                    edge_priority,
                    topk_2hop,
                )
                if second_hop_edge_ids.numel() > 0:
                    selected_edge_groups.append(second_hop_edge_ids)

        if selected_edge_groups:
            merged_edge_ids = torch.cat(selected_edge_groups).unique(sorted=False)
        else:
            merged_edge_ids = torch.empty(0, dtype=torch.long)
        local_subgraph_edge_pool.append(merged_edge_ids)

    return local_subgraph_edge_pool


def _load_cached_local_subgraph_edge_pool(
    cache_path: Path,
    num_nodes: int,
    num_edges: int,
    topk_1hop: int,
    topk_2hop: int,
    hops: int,
) -> list[torch.Tensor] | None:
    if not cache_path.exists():
        return None
    if cache_path.stat().st_size == 0:
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None

    edge_pool = payload.get("edge_pool")
    if not isinstance(edge_pool, list):
        return None

    if (
        payload.get("num_nodes") != num_nodes
        or payload.get("num_edges") != num_edges
        or payload.get("topk_1hop") != topk_1hop
        or payload.get("topk_2hop") != topk_2hop
        or payload.get("hops") != hops
        or len(edge_pool) != num_nodes
    ):
        return None

    normalized_pool: list[torch.Tensor] = []
    for edge_ids in edge_pool:
        if not isinstance(edge_ids, torch.Tensor):
            return None
        normalized_pool.append(edge_ids.long())
    return normalized_pool


def _get_or_compute_local_subgraph_edge_pool(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_prior: torch.Tensor | None,
    cache_path: Path | None,
    use_cache: bool,
    topk_1hop: int,
    topk_2hop: int,
    hops: int,
) -> tuple[list[torch.Tensor], str]:
    if use_cache and cache_path is not None:
        cached_pool = _load_cached_local_subgraph_edge_pool(
            cache_path=cache_path,
            num_nodes=num_nodes,
            num_edges=edge_index.size(0),
            topk_1hop=topk_1hop,
            topk_2hop=topk_2hop,
            hops=hops,
        )
        if cached_pool is not None:
            return cached_pool, "loaded"

    edge_pool = _compute_local_subgraph_edge_pool(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_prior=node_prior,
        topk_1hop=topk_1hop,
        topk_2hop=topk_2hop,
        hops=hops,
    )

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "edge_pool": edge_pool,
                "num_nodes": num_nodes,
                "num_edges": edge_index.size(0),
                "topk_1hop": topk_1hop,
                "topk_2hop": topk_2hop,
                "hops": hops,
            },
            cache_path,
        )

    return edge_pool, "computed"


def _summarize_local_subgraph_edge_pool(
    edge_pool: list[torch.Tensor] | None,
) -> tuple[float, int]:
    if not edge_pool:
        return 0.0, 0
    lengths = [int(edge_ids.numel()) for edge_ids in edge_pool]
    if not lengths:
        return 0.0, 0
    return float(sum(lengths)) / float(len(lengths)), max(lengths)


def _build_local_subgraph_cache_path(
    root_path: Path,
    hops: int,
    topk_1hop: int,
    topk_2hop: int,
) -> Path:
    return root_path / f"local_subgraph_pool_h{hops}_k{topk_1hop}_{topk_2hop}.pt"


def _build_triple_subgraph_template_cache_path(
    root_path: Path,
    hops: int,
    topk_1hop: int,
    topk_2hop: int,
) -> Path:
    return root_path / f"triple_subgraph_templates_h{hops}_k{topk_1hop}_{topk_2hop}.pt"


def _build_batch_local_subgraph_mask(
    graph: Graph,
    head_index: torch.Tensor,
) -> torch.Tensor | None:
    if graph.local_subgraph_edge_pool is None:
        return None

    selected_edge_ids: list[torch.Tensor] = []
    for node_index in head_index.detach().cpu().unique(sorted=False).tolist():
        edge_ids = graph.local_subgraph_edge_pool[node_index]
        if edge_ids.numel() > 0:
            selected_edge_ids.append(edge_ids)

    if not selected_edge_ids:
        return None

    merged_edge_ids = torch.cat(selected_edge_ids).unique(sorted=False)
    graph_mask = torch.zeros(graph.edge_index.size(0), dtype=torch.bool)
    graph_mask[merged_edge_ids] = True
    return graph_mask


def _build_batch_compact_subgraph(
    graph: Graph,
    head_index: torch.Tensor,
    tail_index: torch.Tensor,
) -> tuple[Graph, torch.Tensor, torch.Tensor]:
    if graph.local_subgraph_edge_pool is None:
        return graph, head_index, tail_index

    anchor_nodes = torch.cat([head_index, tail_index], dim=0).detach().cpu().unique(sorted=False)
    selected_edge_ids: list[torch.Tensor] = []
    for node_index in anchor_nodes.tolist():
        edge_ids = graph.local_subgraph_edge_pool[node_index]
        if edge_ids.numel() > 0:
            selected_edge_ids.append(edge_ids)

    if selected_edge_ids:
        merged_edge_ids = torch.cat(selected_edge_ids).unique(sorted=False)
        selected_edges = graph.edge_index[merged_edge_ids]
        node_ids = torch.cat([anchor_nodes, selected_edges[:, 0].cpu(), selected_edges[:, 2].cpu()]).unique(sorted=True)
        local_source = torch.searchsorted(node_ids, selected_edges[:, 0].cpu().contiguous())
        local_target = torch.searchsorted(node_ids, selected_edges[:, 2].cpu().contiguous())
        compact_edge_index = torch.stack(
            [local_source, selected_edges[:, 1].cpu(), local_target],
            dim=1,
        ).long()
    else:
        node_ids = anchor_nodes.unique(sorted=True)
        compact_edge_index = torch.empty((0, 3), dtype=torch.long)

    compact_head_index = torch.searchsorted(node_ids, head_index.detach().cpu().contiguous()).long()
    compact_tail_index = torch.searchsorted(node_ids, tail_index.detach().cpu().contiguous()).long()

    compact_graph = Graph(
        edge_index=compact_edge_index,
        num_nodes=int(node_ids.numel()),
        num_relations=graph.num_relations,
        id2relation=graph.id2relation,
        node_ids=node_ids.long(),
    )
    return compact_graph, compact_head_index, compact_tail_index


def _build_single_triple_subgraph_template(
    head_id: int,
    tail_id: int,
    graph: Graph,
) -> dict[str, torch.Tensor]:
    if graph.local_subgraph_edge_pool is None:
        node_ids = torch.tensor(sorted({int(head_id), int(tail_id)}), dtype=torch.long)
        local_head = torch.searchsorted(node_ids, torch.tensor([head_id], dtype=torch.long)).squeeze(0)
        local_tail = torch.searchsorted(node_ids, torch.tensor([tail_id], dtype=torch.long)).squeeze(0)
        return {
            "node_ids": node_ids.long(),
            "edge_index": torch.empty((0, 3), dtype=torch.long),
            "head_index": local_head.long(),
            "tail_index": local_tail.long(),
        }

    selected_edge_ids: list[torch.Tensor] = []
    for anchor_id in {int(head_id), int(tail_id)}:
        edge_ids = graph.local_subgraph_edge_pool[anchor_id]
        if edge_ids.numel() > 0:
            selected_edge_ids.append(edge_ids)

    if selected_edge_ids:
        merged_edge_ids = torch.cat(selected_edge_ids).unique(sorted=False)
        selected_edges = graph.edge_index[merged_edge_ids]
        node_ids = torch.cat(
            [
                torch.tensor([head_id, tail_id], dtype=torch.long),
                selected_edges[:, 0].cpu(),
                selected_edges[:, 2].cpu(),
            ]
        ).unique(sorted=True)
        local_source = torch.searchsorted(node_ids, selected_edges[:, 0].cpu().contiguous())
        local_target = torch.searchsorted(node_ids, selected_edges[:, 2].cpu().contiguous())
        compact_edge_index = torch.stack(
            [local_source, selected_edges[:, 1].cpu(), local_target],
            dim=1,
        ).long()
    else:
        node_ids = torch.tensor(sorted({int(head_id), int(tail_id)}), dtype=torch.long)
        compact_edge_index = torch.empty((0, 3), dtype=torch.long)

    local_head = torch.searchsorted(node_ids, torch.tensor([head_id], dtype=torch.long)).squeeze(0)
    local_tail = torch.searchsorted(node_ids, torch.tensor([tail_id], dtype=torch.long)).squeeze(0)
    return {
        "node_ids": node_ids.long(),
        "edge_index": compact_edge_index.long(),
        "head_index": local_head.long(),
        "tail_index": local_tail.long(),
    }


def _compute_triple_subgraph_templates(
    triplets: torch.Tensor,
    graph: Graph,
) -> list[dict[str, torch.Tensor]]:
    if graph.local_subgraph_edge_pool is None:
        return []

    templates: list[dict[str, torch.Tensor]] = []
    for triplet in triplets.tolist():
        head_id, _, tail_id = triplet
        templates.append(_build_single_triple_subgraph_template(int(head_id), int(tail_id), graph))

    return templates


def _load_cached_triple_subgraph_templates(
    cache_path: Path,
    triplets: torch.Tensor,
    num_edges: int,
    topk_1hop: int,
    topk_2hop: int,
    hops: int,
) -> dict[tuple[int, int, int], dict[str, torch.Tensor]] | None:
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("num_edges") != num_edges
        or payload.get("topk_1hop") != topk_1hop
        or payload.get("topk_2hop") != topk_2hop
        or payload.get("hops") != hops
    ):
        return None

    templates = payload.get("templates")
    if not isinstance(templates, list) or len(templates) != triplets.size(0):
        return None

    indexed_templates: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
    for triplet_tensor, template in zip(triplets, templates):
        if not isinstance(template, dict):
            return None
        triplet_key = tuple(int(value) for value in triplet_tensor.tolist())
        indexed_templates[triplet_key] = {
            "node_ids": template["node_ids"].long(),
            "edge_index": template["edge_index"].long(),
            "head_index": template["head_index"].long(),
            "tail_index": template["tail_index"].long(),
        }
    return indexed_templates


def _get_or_compute_triple_subgraph_templates(
    triplets: torch.Tensor,
    graph: Graph,
    cache_path: Path | None,
    use_cache: bool,
    topk_1hop: int,
    topk_2hop: int,
    hops: int,
) -> tuple[dict[tuple[int, int, int], dict[str, torch.Tensor]], str]:
    if use_cache and cache_path is not None:
        cached_templates = _load_cached_triple_subgraph_templates(
            cache_path=cache_path,
            triplets=triplets,
            num_edges=graph.edge_index.size(0),
            topk_1hop=topk_1hop,
            topk_2hop=topk_2hop,
            hops=hops,
        )
        if cached_templates is not None:
            return cached_templates, "loaded"

    templates = _compute_triple_subgraph_templates(triplets, graph)
    indexed_templates = {
        tuple(int(value) for value in triplet.tolist()): template
        for triplet, template in zip(triplets, templates)
    }

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "templates": templates,
                "num_edges": graph.edge_index.size(0),
                "topk_1hop": topk_1hop,
                "topk_2hop": topk_2hop,
                "hops": hops,
            },
            cache_path,
        )

    return indexed_templates, "computed"


class TransductiveKnowledgeGraph:
    def __init__(
        self,
        data_path: Path,
        task_type: str,
        include_valid_in_train: bool = False,
        enable_topology_prior: bool = False,
        topology_prior_cache: bool = True,
        topology_pagerank_damping: float = 0.85,
        topology_pagerank_iterations: int = 50,
        topology_pagerank_tolerance: float = 1e-6,
        enable_local_subgraph: bool = False,
        local_subgraph_cache: bool = True,
        local_subgraph_hops: int = 2,
        local_subgraph_topk_1hop: int = 32,
        local_subgraph_topk_2hop: int = 8,
    ) -> None:
        self.data_path = data_path
        self.task_type = task_type

        self.entity2id, self.id2entity = _read_mapping_file(data_path / "entities.txt")
        self.relation2id, self.id2relation = _read_mapping_file(data_path / "relations.txt")

        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)

        train = _encode_triplets(
            _read_triplets(data_path / "train.txt"),
            self.entity2id,
            self.relation2id,
        )
        valid = _encode_triplets(
            _read_triplets(data_path / "valid.txt"),
            self.entity2id,
            self.relation2id,
        )
        test = _encode_triplets(
            _read_triplets(data_path / "test.txt"),
            self.entity2id,
            self.relation2id,
        )
        train_for_training = train + valid if include_valid_in_train else train

        self.train_triplets = torch.tensor(train_for_training, dtype=torch.long)
        self.valid_triplets = torch.tensor(valid, dtype=torch.long)
        self.test_triplets = torch.tensor(test, dtype=torch.long)

        self.train_filters = _build_filters(train_for_training)
        all_filters = _build_filters(train_for_training + test)
        self.valid_filters = all_filters
        self.test_filters = all_filters

        node_prior = None
        topology_prior_status = "disabled"
        if enable_topology_prior:
            node_prior, topology_prior_status = _get_or_compute_topology_prior(
                edge_index=self.train_triplets,
                num_nodes=self.num_entity,
                num_relations=self.num_relation,
                cache_path=data_path / "topology_prior_train_graph.pt",
                use_cache=topology_prior_cache,
                damping=topology_pagerank_damping,
                num_iterations=topology_pagerank_iterations,
                tolerance=topology_pagerank_tolerance,
            )

        local_subgraph_edge_pool = None
        local_subgraph_status = "disabled"
        if enable_local_subgraph:
            local_subgraph_edge_pool, local_subgraph_status = _get_or_compute_local_subgraph_edge_pool(
                edge_index=self.train_triplets,
                num_nodes=self.num_entity,
                node_prior=node_prior,
                cache_path=_build_local_subgraph_cache_path(
                    data_path,
                    hops=local_subgraph_hops,
                    topk_1hop=local_subgraph_topk_1hop,
                    topk_2hop=local_subgraph_topk_2hop,
                ),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
        local_avg_edges, local_max_edges = _summarize_local_subgraph_edge_pool(local_subgraph_edge_pool)

        self.train_graph = Graph(
            self.train_triplets.clone(),
            self.num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=node_prior,
            local_subgraph_edge_pool=local_subgraph_edge_pool,
        )
        self.valid_graph = Graph(
            self.train_triplets.clone(),
            self.num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=node_prior,
            local_subgraph_edge_pool=local_subgraph_edge_pool,
        )
        self.test_graph = Graph(
            self.train_triplets.clone(),
            self.num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=node_prior,
            local_subgraph_edge_pool=local_subgraph_edge_pool,
        )

        self.dataset_info = DatasetInfo(
            name=data_path.name,
            task_type=task_type,
            num_relations=self.num_relation,
            train_num_entities=self.num_entity,
            eval_num_entities=self.num_entity,
            use_entity_embeddings=True,
            train_graph_num_edges=int(self.train_triplets.size(0)),
            eval_graph_num_edges=int(self.train_triplets.size(0)),
            train_topology_prior_status=topology_prior_status,
            eval_topology_prior_status=topology_prior_status,
            train_local_subgraph_status=local_subgraph_status,
            eval_local_subgraph_status=local_subgraph_status,
            train_local_subgraph_avg_edges=local_avg_edges,
            eval_local_subgraph_avg_edges=local_avg_edges,
            train_local_subgraph_max_edges=local_max_edges,
            eval_local_subgraph_max_edges=local_max_edges,
        )

    def train_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor | Graph]:
        triplets = [item.clone() for item in batch]
        batch_size = len(triplets)
        for idx in range(batch_size // 2, batch_size):
            triplets[idx] = torch.flip(triplets[idx], dims=[0])
            relation = triplets[idx][1]
            triplets[idx][1] = relation + 1 if relation % 2 == 0 else relation - 1

        filter_mask = torch.zeros(batch_size, self.train_graph.num_nodes, dtype=torch.bool)
        stacked = torch.stack(triplets, dim=0)
        h_index, r_index, t_index = stacked.unbind(-1)
        for batch_idx, query in enumerate(stacked):
            filter_mask[batch_idx, list(self.train_filters[(query[0].item(), query[1].item())])] = True

        local_subgraph_mask = _build_batch_local_subgraph_mask(self.train_graph, h_index)

        return {
            "h_index": h_index,
            "r_index": r_index,
            "t_index": t_index,
            "filter_mask": filter_mask,
            "graph": self.train_graph,
            "local_subgraph_mask": local_subgraph_mask,
        }

    def eval_collate_fn(
        self,
        batch: list[torch.Tensor],
        split: str,
    ) -> dict[str, torch.Tensor | Graph]:
        triplets = torch.stack(batch, dim=0)
        graph = self.valid_graph if split == "valid" else self.test_graph
        filters = self.valid_filters if split == "valid" else self.test_filters

        filter_mask = torch.zeros(triplets.size(0), graph.num_nodes, dtype=torch.bool)
        for batch_idx, query in enumerate(triplets):
            filter_mask[batch_idx, list(filters[(query[0].item(), query[1].item())])] = True

        h_index, r_index, t_index = triplets.unbind(-1)
        local_subgraph_mask = _build_batch_local_subgraph_mask(graph, h_index)
        return {
            "h_index": h_index,
            "r_index": r_index,
            "t_index": t_index,
            "filter_mask": filter_mask,
            "graph": graph,
            "local_subgraph_mask": local_subgraph_mask,
        }


class InductiveKnowledgeGraph:
    def __init__(
        self,
        data_path: Path,
        inductive_path: Path,
        task_type: str,
        include_valid_in_train: bool = False,
        enable_topology_prior: bool = False,
        topology_prior_cache: bool = True,
        topology_pagerank_damping: float = 0.85,
        topology_pagerank_iterations: int = 50,
        topology_pagerank_tolerance: float = 1e-6,
        enable_local_subgraph: bool = False,
        local_subgraph_cache: bool = True,
        local_subgraph_hops: int = 2,
        local_subgraph_topk_1hop: int = 32,
        local_subgraph_topk_2hop: int = 8,
    ) -> None:
        self.data_path = data_path
        self.inductive_path = inductive_path
        self.task_type = task_type

        self.entity2id, self.id2entity = _read_mapping_file(data_path / "entities.txt")
        self.ind_entity2id, self.ind_id2entity = _read_mapping_file(inductive_path / "entities.txt")
        self.relation2id, self.id2relation = _read_mapping_file(data_path / "relations.txt")

        self.num_entity = len(self.entity2id)
        self.ind_num_entity = len(self.ind_entity2id)
        self.num_relation = len(self.relation2id)

        train = _encode_triplets(
            _read_triplets(data_path / "train.txt"),
            self.entity2id,
            self.relation2id,
        )
        ind_train = _encode_triplets(
            _read_triplets(inductive_path / "train.txt"),
            self.ind_entity2id,
            self.relation2id,
        )
        ind_valid = _encode_triplets(
            _read_triplets(inductive_path / "valid.txt"),
            self.ind_entity2id,
            self.relation2id,
        )
        ind_test = _encode_triplets(
            _read_triplets(inductive_path / "test.txt"),
            self.ind_entity2id,
            self.relation2id,
        )
        ind_train_for_eval = ind_train + ind_valid if include_valid_in_train else ind_train

        self.train_triplets = torch.tensor(train, dtype=torch.long)
        self.valid_triplets = torch.tensor(ind_valid, dtype=torch.long)
        self.test_triplets = torch.tensor(ind_test, dtype=torch.long)
        self.valid_train_triplets = (
            torch.tensor(ind_valid, dtype=torch.long)
            if include_valid_in_train
            else torch.empty((0, 3), dtype=torch.long)
        )

        self.train_filters = _build_filters(train)
        self.valid_filters = _build_filters(ind_train_for_eval)
        self.test_filters = _build_filters(ind_train_for_eval + ind_test)

        train_node_prior = None
        eval_node_prior = None
        train_topology_prior_status = "disabled"
        eval_topology_prior_status = "disabled"
        train_local_subgraph_edge_pool = None
        eval_local_subgraph_edge_pool = None
        train_local_subgraph_status = "disabled"
        eval_local_subgraph_status = "disabled"
        ind_train_triplets = torch.tensor(ind_train_for_eval, dtype=torch.long)
        if enable_topology_prior:
            train_node_prior, train_topology_prior_status = _get_or_compute_topology_prior(
                edge_index=self.train_triplets,
                num_nodes=self.num_entity,
                num_relations=self.num_relation,
                cache_path=data_path / "topology_prior_train_graph.pt",
                use_cache=topology_prior_cache,
                damping=topology_pagerank_damping,
                num_iterations=topology_pagerank_iterations,
                tolerance=topology_pagerank_tolerance,
            )
            eval_node_prior, eval_topology_prior_status = _get_or_compute_topology_prior(
                edge_index=ind_train_triplets,
                num_nodes=self.ind_num_entity,
                num_relations=self.num_relation,
                cache_path=inductive_path / "topology_prior_train_graph.pt",
                use_cache=topology_prior_cache,
                damping=topology_pagerank_damping,
                num_iterations=topology_pagerank_iterations,
                tolerance=topology_pagerank_tolerance,
            )

        if enable_local_subgraph:
            train_local_subgraph_edge_pool, train_local_subgraph_status = _get_or_compute_local_subgraph_edge_pool(
                edge_index=self.train_triplets,
                num_nodes=self.num_entity,
                node_prior=train_node_prior,
                cache_path=_build_local_subgraph_cache_path(
                    data_path,
                    hops=local_subgraph_hops,
                    topk_1hop=local_subgraph_topk_1hop,
                    topk_2hop=local_subgraph_topk_2hop,
                ),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
            eval_local_subgraph_edge_pool, eval_local_subgraph_status = _get_or_compute_local_subgraph_edge_pool(
                edge_index=ind_train_triplets,
                num_nodes=self.ind_num_entity,
                node_prior=eval_node_prior,
                cache_path=_build_local_subgraph_cache_path(
                    inductive_path,
                    hops=local_subgraph_hops,
                    topk_1hop=local_subgraph_topk_1hop,
                    topk_2hop=local_subgraph_topk_2hop,
                ),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
        train_local_avg_edges, train_local_max_edges = _summarize_local_subgraph_edge_pool(
            train_local_subgraph_edge_pool
        )
        eval_local_avg_edges, eval_local_max_edges = _summarize_local_subgraph_edge_pool(
            eval_local_subgraph_edge_pool
        )

        self.train_graph = Graph(
            self.train_triplets.clone(),
            self.num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=train_node_prior,
            local_subgraph_edge_pool=train_local_subgraph_edge_pool,
        )
        self.valid_graph = Graph(
            ind_train_triplets,
            self.ind_num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=eval_node_prior,
            local_subgraph_edge_pool=eval_local_subgraph_edge_pool,
        )
        self.test_graph = Graph(
            ind_train_triplets.clone(),
            self.ind_num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_prior=eval_node_prior,
            local_subgraph_edge_pool=eval_local_subgraph_edge_pool,
        )

        self.dataset_info = DatasetInfo(
            name=data_path.name,
            task_type=task_type,
            num_relations=self.num_relation,
            train_num_entities=self.num_entity,
            eval_num_entities=self.ind_num_entity,
            use_entity_embeddings=False,
            train_graph_num_edges=int(self.train_triplets.size(0)),
            eval_graph_num_edges=int(ind_train_triplets.size(0)),
            train_topology_prior_status=train_topology_prior_status,
            eval_topology_prior_status=eval_topology_prior_status,
            train_local_subgraph_status=train_local_subgraph_status,
            eval_local_subgraph_status=eval_local_subgraph_status,
            train_local_subgraph_avg_edges=train_local_avg_edges,
            eval_local_subgraph_avg_edges=eval_local_avg_edges,
            train_local_subgraph_max_edges=train_local_max_edges,
            eval_local_subgraph_max_edges=eval_local_max_edges,
        )

    def train_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor | Graph]:
        return self._train_collate_with_graph(
            batch,
            graph=self.train_graph,
            filters=self.train_filters,
        )

    def eval_train_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor | Graph]:
        return self._train_collate_with_graph(
            batch,
            graph=self.valid_graph,
            filters=self.valid_filters,
        )

    def _train_collate_with_graph(
        self,
        batch: list[torch.Tensor],
        *,
        graph: Graph,
        filters: defaultdict[tuple[int, int], set[int]],
    ) -> dict[str, torch.Tensor | Graph]:
        triplets = [item.clone() for item in batch]
        batch_size = len(triplets)
        for idx in range(batch_size // 2, batch_size):
            triplets[idx] = torch.flip(triplets[idx], dims=[0])
            relation = triplets[idx][1]
            triplets[idx][1] = relation + 1 if relation % 2 == 0 else relation - 1

        filter_mask = torch.zeros(batch_size, graph.num_nodes, dtype=torch.bool)
        stacked = torch.stack(triplets, dim=0)
        h_index, r_index, t_index = stacked.unbind(-1)
        for batch_idx, query in enumerate(stacked):
            filter_mask[batch_idx, list(filters[(query[0].item(), query[1].item())])] = True

        local_subgraph_mask = _build_batch_local_subgraph_mask(graph, h_index)

        return {
            "h_index": h_index,
            "r_index": r_index,
            "t_index": t_index,
            "filter_mask": filter_mask,
            "graph": graph,
            "local_subgraph_mask": local_subgraph_mask,
        }

    def eval_collate_fn(
        self,
        batch: list[torch.Tensor],
        split: str,
    ) -> dict[str, torch.Tensor | Graph]:
        triplets = torch.stack(batch, dim=0)
        graph = self.valid_graph if split == "valid" else self.test_graph
        filters = self.valid_filters if split == "valid" else self.test_filters

        filter_mask = torch.zeros(triplets.size(0), graph.num_nodes, dtype=torch.bool)
        for batch_idx, query in enumerate(triplets):
            filter_mask[batch_idx, list(filters[(query[0].item(), query[1].item())])] = True

        h_index, r_index, t_index = triplets.unbind(-1)
        local_subgraph_mask = _build_batch_local_subgraph_mask(graph, h_index)
        return {
            "h_index": h_index,
            "r_index": r_index,
            "t_index": t_index,
            "filter_mask": filter_mask,
            "graph": graph,
            "local_subgraph_mask": local_subgraph_mask,
        }


class TripleClassificationKnowledgeGraph:
    def __init__(
        self,
        data_path: Path,
        task_type: str,
        enable_topology_prior: bool = False,
        topology_prior_cache: bool = True,
        topology_pagerank_damping: float = 0.85,
        topology_pagerank_iterations: int = 50,
        topology_pagerank_tolerance: float = 1e-6,
        enable_local_subgraph: bool = False,
        local_subgraph_cache: bool = True,
        local_subgraph_hops: int = 2,
        local_subgraph_topk_1hop: int = 32,
        local_subgraph_topk_2hop: int = 8,
    ) -> None:
        self.data_path = data_path
        self.task_type = task_type

        self.entity2id, self.id2entity = _read_mapping_file(data_path / "entities.txt")
        self.relation2id, self.id2relation = _read_mapping_file(data_path / "relations.txt")
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_forward_relation = _forward_relation_count(self.relation2id)

        train_raw = _read_triplets(data_path / "train.txt")
        valid_raw = _read_triplets(data_path / "valid.txt")
        test_raw = _read_triplets(data_path / "test.txt")

        self.train_triplets = torch.tensor(
            _encode_triplets_without_inverse(train_raw, self.entity2id, self.relation2id),
            dtype=torch.long,
        )
        self.valid_triplets = torch.tensor(
            _encode_triplets_without_inverse(valid_raw, self.entity2id, self.relation2id),
            dtype=torch.long,
        )
        self.test_triplets = torch.tensor(
            _encode_triplets_without_inverse(test_raw, self.entity2id, self.relation2id),
            dtype=torch.long,
        )

        self.train_labels = _read_labels(data_path / "train_labels.txt")
        self.valid_labels = _read_labels(data_path / "valid_labels.txt")
        self.test_labels = _read_labels(data_path / "test_labels.txt")

        if self.train_triplets.size(0) != self.train_labels.size(0):
            raise ValueError("train.txt and train_labels.txt have different lengths.")
        if self.valid_triplets.size(0) != self.valid_labels.size(0):
            raise ValueError("valid.txt and valid_labels.txt have different lengths.")
        if self.test_triplets.size(0) != self.test_labels.size(0):
            raise ValueError("test.txt and test_labels.txt have different lengths.")

        positive_train_raw = [
            triplet
            for triplet, label in zip(train_raw, self.train_labels.tolist())
            if int(label) == 1
        ]
        train_graph_triplets = _encode_triplets(positive_train_raw, self.entity2id, self.relation2id)
        train_graph_edge_index = torch.tensor(train_graph_triplets, dtype=torch.long)
        node_prior = None
        topology_prior_status = "disabled"
        if enable_topology_prior:
            node_prior, topology_prior_status = _get_or_compute_topology_prior(
                edge_index=train_graph_edge_index,
                num_nodes=self.num_entity,
                num_relations=self.num_relation,
                cache_path=data_path / "topology_prior_train_graph.pt",
                use_cache=topology_prior_cache,
                damping=topology_pagerank_damping,
                num_iterations=topology_pagerank_iterations,
                tolerance=topology_pagerank_tolerance,
            )

        local_subgraph_edge_pool = None
        local_subgraph_status = "disabled"
        if enable_local_subgraph:
            local_subgraph_edge_pool, local_subgraph_status = _get_or_compute_local_subgraph_edge_pool(
                edge_index=train_graph_edge_index,
                num_nodes=self.num_entity,
                node_prior=node_prior,
                cache_path=_build_local_subgraph_cache_path(
                    data_path,
                    hops=local_subgraph_hops,
                    topk_1hop=local_subgraph_topk_1hop,
                    topk_2hop=local_subgraph_topk_2hop,
                ),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
        local_avg_edges, local_max_edges = _summarize_local_subgraph_edge_pool(local_subgraph_edge_pool)

        self.graph = Graph(
            train_graph_edge_index,
            self.num_entity,
            self.num_relation,
            id2relation=self.id2relation,
            node_ids=torch.arange(self.num_entity, dtype=torch.long),
            node_prior=node_prior,
            local_subgraph_edge_pool=local_subgraph_edge_pool,
        )
        self.train_triple_subgraph_templates: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        self.valid_triple_subgraph_templates: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        self.test_triple_subgraph_templates: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        self.triple_subgraph_template_status = "disabled"
        if enable_local_subgraph:
            template_cache_path = _build_triple_subgraph_template_cache_path(
                data_path,
                hops=local_subgraph_hops,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
            )
            self.train_triple_subgraph_templates, train_template_status = _get_or_compute_triple_subgraph_templates(
                triplets=self.train_triplets,
                graph=self.graph,
                cache_path=template_cache_path.with_name(template_cache_path.stem + "_train.pt"),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
            self.valid_triple_subgraph_templates, valid_template_status = _get_or_compute_triple_subgraph_templates(
                triplets=self.valid_triplets,
                graph=self.graph,
                cache_path=template_cache_path.with_name(template_cache_path.stem + "_valid.pt"),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
            self.test_triple_subgraph_templates, test_template_status = _get_or_compute_triple_subgraph_templates(
                triplets=self.test_triplets,
                graph=self.graph,
                cache_path=template_cache_path.with_name(template_cache_path.stem + "_test.pt"),
                use_cache=local_subgraph_cache,
                topk_1hop=local_subgraph_topk_1hop,
                topk_2hop=local_subgraph_topk_2hop,
                hops=local_subgraph_hops,
            )
            self.triple_subgraph_template_status = (
                f"train={train_template_status}|valid={valid_template_status}|test={test_template_status}"
            )
        self.known_positive_triplets = {
            tuple(triplet.tolist())
            for triplet, label in zip(self.train_triplets, self.train_labels)
            if int(label.item()) == 1
        }
        positive_train_count = int(self.train_labels.sum().item())
        self.positive_train_triplets = self.train_triplets[self.train_labels.bool()]
        if self.positive_train_triplets.numel() == 0:
            self.positive_train_triplets = self.train_triplets

        self.dataset_info = DatasetInfo(
            name=data_path.name,
            task_type=task_type,
            num_relations=self.num_relation,
            train_num_entities=self.num_entity,
            eval_num_entities=self.num_entity,
            use_entity_embeddings=True,
            train_graph_num_edges=int(train_graph_edge_index.size(0)),
            eval_graph_num_edges=int(train_graph_edge_index.size(0)),
            train_topology_prior_status=topology_prior_status,
            eval_topology_prior_status=topology_prior_status,
            train_local_subgraph_status=local_subgraph_status,
            eval_local_subgraph_status=local_subgraph_status,
            train_local_subgraph_avg_edges=local_avg_edges,
            eval_local_subgraph_avg_edges=local_avg_edges,
            train_local_subgraph_max_edges=local_max_edges,
            eval_local_subgraph_max_edges=local_max_edges,
            triple_subgraph_template_status=self.triple_subgraph_template_status,
        )
        self.positive_train_count = positive_train_count

    def _merge_template_batch(
        self,
        triplets: torch.Tensor,
        template_map: dict[tuple[int, int, int], dict[str, torch.Tensor]],
    ) -> tuple[Graph, torch.Tensor, torch.Tensor] | None:
        if not template_map:
            return None

        template_records: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
        edge_chunks: list[torch.Tensor] = []
        all_node_ids: list[torch.Tensor] = []

        for triplet in triplets.tolist():
            template = template_map.get(tuple(int(value) for value in triplet))
            if template is None:
                return None

            node_ids = template["node_ids"].long()
            template_records.append((template, node_ids))
            all_node_ids.append(node_ids)

        merged_node_ids = torch.cat(all_node_ids, dim=0).unique(sorted=True)
        head_indices: list[int] = []
        tail_indices: list[int] = []

        for template, node_ids in template_records:
            edge_index = template["edge_index"].long()
            local_head = int(template["head_index"].item())
            local_tail = int(template["tail_index"].item())

            global_head_id = int(node_ids[local_head].item())
            global_tail_id = int(node_ids[local_tail].item())
            head_indices.append(int(torch.searchsorted(merged_node_ids, torch.tensor([global_head_id], dtype=torch.long)).item()))
            tail_indices.append(int(torch.searchsorted(merged_node_ids, torch.tensor([global_tail_id], dtype=torch.long)).item()))

            if edge_index.numel() > 0:
                edge_source_global = node_ids[edge_index[:, 0]].contiguous()
                edge_target_global = node_ids[edge_index[:, 2]].contiguous()
                adjusted_edge_index = torch.stack(
                    [
                        torch.searchsorted(merged_node_ids, edge_source_global),
                        edge_index[:, 1],
                        torch.searchsorted(merged_node_ids, edge_target_global),
                    ],
                    dim=1,
                ).long()
                edge_chunks.append(adjusted_edge_index)

        merged_edge_index = (
            torch.cat(edge_chunks, dim=0).unique(dim=0)
            if edge_chunks
            else torch.empty((0, 3), dtype=torch.long)
        )
        merged_graph = Graph(
            edge_index=merged_edge_index,
            num_nodes=int(merged_node_ids.numel()),
            num_relations=self.num_relation,
            id2relation=self.id2relation,
            node_ids=merged_node_ids,
        )
        return (
            merged_graph,
            torch.tensor(head_indices, dtype=torch.long),
            torch.tensor(tail_indices, dtype=torch.long),
        )

    def _collate_with_labels(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor]],
        split: str | None = None,
    ) -> dict[str, torch.Tensor | Graph]:
        triplets = torch.stack([item[0] for item in items], dim=0)
        labels = torch.stack([item[1] for item in items], dim=0).float()
        h_index, r_index, t_index = triplets.unbind(-1)
        template_map = None
        if split == "valid":
            template_map = self.valid_triple_subgraph_templates
        elif split == "test":
            template_map = self.test_triple_subgraph_templates
        elif split == "train":
            template_map = self.train_triple_subgraph_templates

        merged_template_batch = self._merge_template_batch(triplets, template_map or {})
        if merged_template_batch is not None:
            compact_graph, compact_head_index, compact_tail_index = merged_template_batch
        else:
            compact_graph, compact_head_index, compact_tail_index = _build_batch_compact_subgraph(
                self.graph,
                h_index,
                t_index,
            )
        return {
            "h_index": compact_head_index,
            "r_index": r_index,
            "t_index": compact_tail_index,
            "label": labels,
            "graph": compact_graph,
            "entity_index": compact_graph.node_ids,
        }

    def train_collate_fn(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor | Graph]:
        positive_triplets = torch.stack([item.clone() for item in batch], dim=0)
        negative_triplets = positive_triplets.clone()
        if self.num_entity > 1:
            corrupt_head_mask = torch.rand(negative_triplets.size(0)) < 0.5
            for idx in range(negative_triplets.size(0)):
                triplet = negative_triplets[idx].clone()
                original_h = int(triplet[0].item())
                original_t = int(triplet[2].item())
                corrupt_head = bool(corrupt_head_mask[idx].item())

                max_attempt = max(16, self.num_entity * 2)
                success = False
                for _ in range(max_attempt):
                    sampled_entity = int(torch.randint(low=0, high=self.num_entity, size=(1,), dtype=torch.long).item())
                    if corrupt_head:
                        if sampled_entity == original_h:
                            continue
                        candidate = (sampled_entity, int(triplet[1].item()), original_t)
                        if candidate in self.known_positive_triplets:
                            continue
                        triplet[0] = sampled_entity
                    else:
                        if sampled_entity == original_t:
                            continue
                        candidate = (original_h, int(triplet[1].item()), sampled_entity)
                        if candidate in self.known_positive_triplets:
                            continue
                        triplet[2] = sampled_entity
                    success = True
                    break

                if not success:
                    for entity_id in range(self.num_entity):
                        if corrupt_head:
                            if entity_id == original_h:
                                continue
                            candidate = (entity_id, int(triplet[1].item()), original_t)
                            if candidate in self.known_positive_triplets:
                                continue
                            triplet[0] = entity_id
                            success = True
                            break
                        else:
                            if entity_id == original_t:
                                continue
                            candidate = (original_h, int(triplet[1].item()), entity_id)
                            if candidate in self.known_positive_triplets:
                                continue
                            triplet[2] = entity_id
                            success = True
                            break

                if success:
                    negative_triplets[idx] = triplet
        triplets = torch.cat([positive_triplets, negative_triplets], dim=0)
        labels = torch.cat(
            [
                torch.ones(positive_triplets.size(0), dtype=torch.float32),
                torch.zeros(negative_triplets.size(0), dtype=torch.float32),
            ],
            dim=0,
        )
        h_index, r_index, t_index = triplets.unbind(-1)
        merged_template_batch = self._merge_template_batch(positive_triplets, self.train_triple_subgraph_templates)
        if merged_template_batch is not None:
            positive_graph, positive_head_index, positive_tail_index = merged_template_batch
            negative_graph, negative_head_index, negative_tail_index = _build_batch_compact_subgraph(
                self.graph,
                negative_triplets[:, 0],
                negative_triplets[:, 2],
            )

            node_offset = positive_graph.num_nodes
            merged_node_ids = torch.cat([positive_graph.node_ids, negative_graph.node_ids], dim=0)
            merged_edge_index_parts: list[torch.Tensor] = []
            if positive_graph.edge_index.numel() > 0:
                merged_edge_index_parts.append(positive_graph.edge_index)
            if negative_graph.edge_index.numel() > 0:
                adjusted_negative_edges = negative_graph.edge_index.clone()
                adjusted_negative_edges[:, 0] += node_offset
                adjusted_negative_edges[:, 2] += node_offset
                merged_edge_index_parts.append(adjusted_negative_edges)
            merged_edge_index = (
                torch.cat(merged_edge_index_parts, dim=0)
                if merged_edge_index_parts
                else torch.empty((0, 3), dtype=torch.long)
            )
            compact_graph = Graph(
                edge_index=merged_edge_index,
                num_nodes=int(merged_node_ids.numel()),
                num_relations=self.num_relation,
                id2relation=self.id2relation,
                node_ids=merged_node_ids,
            )
            compact_head_index = torch.cat(
                [positive_head_index, negative_head_index + node_offset],
                dim=0,
            )
            compact_tail_index = torch.cat(
                [positive_tail_index, negative_tail_index + node_offset],
                dim=0,
            )
        else:
            compact_graph, compact_head_index, compact_tail_index = _build_batch_compact_subgraph(
                self.graph,
                h_index,
                t_index,
            )
        return {
            "h_index": compact_head_index,
            "r_index": r_index,
            "t_index": compact_tail_index,
            "label": labels,
            "graph": compact_graph,
            "entity_index": compact_graph.node_ids,
        }

    def eval_collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
        split: str,
    ) -> dict[str, torch.Tensor | Graph]:
        return self._collate_with_labels(batch, split=split)


class KGDataModule:
    def __init__(
        self,
        knowledge_graph: TransductiveKnowledgeGraph | InductiveKnowledgeGraph | TripleClassificationKnowledgeGraph,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        use_test_as_validation: bool,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.dataset_info = knowledge_graph.dataset_info
        self.use_test_as_validation = use_test_as_validation

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.knowledge_graph, TripleClassificationKnowledgeGraph):
            return DataLoader(
                self.knowledge_graph.positive_train_triplets,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.knowledge_graph.train_collate_fn,
            )

        if (
            self.use_test_as_validation
            and isinstance(self.knowledge_graph, InductiveKnowledgeGraph)
            and self.knowledge_graph.valid_train_triplets.numel() > 0
        ):
            train_forward = self.knowledge_graph.train_triplets[
                self.knowledge_graph.train_triplets[:, 1] % 2 == 0
            ].clone()
            valid_forward = self.knowledge_graph.valid_train_triplets[
                self.knowledge_graph.valid_train_triplets[:, 1] % 2 == 0
            ].clone()
            return DataLoader(
                ConcatDataset(
                    [
                        _CollatedBatchDataset(
                            train_forward,
                            self.batch_size,
                            self.knowledge_graph.train_collate_fn,
                        ),
                        _CollatedBatchDataset(
                            valid_forward,
                            self.batch_size,
                            self.knowledge_graph.eval_train_collate_fn,
                        ),
                    ]
                ),
                shuffle=True,
                batch_size=None,
                num_workers=self.num_workers,
            )

        forward_triplets = self.knowledge_graph.train_triplets[
            self.knowledge_graph.train_triplets[:, 1] % 2 == 0
        ].clone()
        return DataLoader(
            forward_triplets,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.knowledge_graph.train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.knowledge_graph, TripleClassificationKnowledgeGraph):
            triplets = (
                self.knowledge_graph.test_triplets
                if self.use_test_as_validation
                else self.knowledge_graph.valid_triplets
            )
            labels = (
                self.knowledge_graph.test_labels
                if self.use_test_as_validation
                else self.knowledge_graph.valid_labels
            )
            split = "test" if self.use_test_as_validation else "valid"
            return DataLoader(
                list(zip(triplets, labels)),
                shuffle=False,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=EvalCollator(self.knowledge_graph, split),
            )

        split = "test" if self.use_test_as_validation else "valid"
        triplets = (
            self.knowledge_graph.test_triplets
            if self.use_test_as_validation
            else self.knowledge_graph.valid_triplets
        )
        return DataLoader(
            triplets,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=EvalCollator(self.knowledge_graph, split),
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.knowledge_graph, TripleClassificationKnowledgeGraph):
            return DataLoader(
                list(zip(self.knowledge_graph.test_triplets, self.knowledge_graph.test_labels)),
                shuffle=False,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=EvalCollator(self.knowledge_graph, "test"),
            )

        return DataLoader(
            self.knowledge_graph.test_triplets,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=EvalCollator(self.knowledge_graph, "test"),
        )


def build_data_module(
    root: Path,
    dataset_spec,
    task_type: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    use_test_as_validation: bool,
    enable_topology_prior: bool = False,
    topology_prior_cache: bool = True,
    topology_pagerank_damping: float = 0.85,
    topology_pagerank_iterations: int = 50,
    topology_pagerank_tolerance: float = 1e-6,
    enable_local_subgraph: bool = False,
    local_subgraph_cache: bool = True,
    local_subgraph_hops: int = 2,
    local_subgraph_topk_1hop: int = 32,
    local_subgraph_topk_2hop: int = 8,
) -> KGDataModule:
    data_path = root / dataset_spec.relative_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    if task_type == TASK_TRIPLE_CLASSIFICATION:
        knowledge_graph = TripleClassificationKnowledgeGraph(
            data_path,
            task_type,
            enable_topology_prior=enable_topology_prior,
            topology_prior_cache=topology_prior_cache,
            topology_pagerank_damping=topology_pagerank_damping,
            topology_pagerank_iterations=topology_pagerank_iterations,
            topology_pagerank_tolerance=topology_pagerank_tolerance,
            enable_local_subgraph=enable_local_subgraph,
            local_subgraph_cache=local_subgraph_cache,
            local_subgraph_hops=local_subgraph_hops,
            local_subgraph_topk_1hop=local_subgraph_topk_1hop,
            local_subgraph_topk_2hop=local_subgraph_topk_2hop,
        )
    elif task_type == TASK_INDUCTIVE:
        if dataset_spec.companion_relative_path is None:
            raise ValueError(f"Inductive dataset '{dataset_spec.name}' is missing companion data.")
        companion_path = root / dataset_spec.companion_relative_path
        if not companion_path.exists():
            raise FileNotFoundError(f"Inductive companion path does not exist: {companion_path}")
        knowledge_graph = InductiveKnowledgeGraph(
            data_path,
            companion_path,
            task_type,
            include_valid_in_train=use_test_as_validation,
            enable_topology_prior=enable_topology_prior,
            topology_prior_cache=topology_prior_cache,
            topology_pagerank_damping=topology_pagerank_damping,
            topology_pagerank_iterations=topology_pagerank_iterations,
            topology_pagerank_tolerance=topology_pagerank_tolerance,
            enable_local_subgraph=enable_local_subgraph,
            local_subgraph_cache=local_subgraph_cache,
            local_subgraph_hops=local_subgraph_hops,
            local_subgraph_topk_1hop=local_subgraph_topk_1hop,
            local_subgraph_topk_2hop=local_subgraph_topk_2hop,
        )
    else:
        knowledge_graph = TransductiveKnowledgeGraph(
            data_path,
            task_type,
            include_valid_in_train=use_test_as_validation,
            enable_topology_prior=enable_topology_prior,
            topology_prior_cache=topology_prior_cache,
            topology_pagerank_damping=topology_pagerank_damping,
            topology_pagerank_iterations=topology_pagerank_iterations,
            topology_pagerank_tolerance=topology_pagerank_tolerance,
            enable_local_subgraph=enable_local_subgraph,
            local_subgraph_cache=local_subgraph_cache,
            local_subgraph_hops=local_subgraph_hops,
            local_subgraph_topk_1hop=local_subgraph_topk_1hop,
            local_subgraph_topk_2hop=local_subgraph_topk_2hop,
        )

    return KGDataModule(
        knowledge_graph=knowledge_graph,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        use_test_as_validation=use_test_as_validation,
    )
