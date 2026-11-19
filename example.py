"""Build SICR-Graph and run a dummy forward pass (no data required)."""
import torch

from Our.common import (
    TASK_INDUCTIVE,
    DatasetInfo,
    DuetGraphConfig,
    Graph,
)
from Our.model import OurDuetGraphModel


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_entities, num_relations = 50, 4
    info = DatasetInfo(
        name="synthetic",
        task_type=TASK_INDUCTIVE,
        num_relations=num_relations,
        train_num_entities=num_entities,
        eval_num_entities=num_entities,
        use_entity_embeddings=True,
    )
    config = DuetGraphConfig(internal_coarse_to_fine_enable=True)
    model = OurDuetGraphModel(info, config).to(device).eval()

    triplets = torch.randint(0, num_entities, (200, 3), device=device)
    triplets[:, 1].remainder_(num_relations)
    batch_size = 4
    batch = {
        "graph": Graph(edge_index=triplets, num_nodes=num_entities, num_relations=num_relations),
        "h_index": torch.randint(0, num_entities, (batch_size,), device=device),
        "r_index": torch.randint(0, num_relations, (batch_size,), device=device),
    }

    with torch.no_grad():
        out = model(batch)

    n_params = sum(p.numel() for p in model.parameters())
    shapes = {k: tuple(v.shape) for k, v in out.items()} if isinstance(out, dict) else tuple(out.shape)
    print(f"params={n_params:,}")
    print("output:", shapes)


if __name__ == "__main__":
    main()