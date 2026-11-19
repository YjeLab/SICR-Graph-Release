# SICR-Graph

Official implementation of **SICR-Graph** (Structure-Aware Internalized Coarse-to-Fine Reasoning), accompanying the paper *"结构感知内化的归纳知识图谱粗到细推理方法 / Structure-Aware Internalized Coarse-to-Fine Reasoning for Inductive Knowledge Graph"* (under review).

SICR-Graph performs inductive knowledge-graph reasoning with a fully internalized coarse-to-fine pipeline on top of a dual-pathway (local message passing + global linear attention) backbone:

- **I1 – Query-aware node-level fusion**: a per-node scalar gate conditioned on the query relation and head state fuses the local and global channels, replacing the global scalar fusion of the backbone.
- **I2 – Structure-adaptive local processing**: an automatic, non-learned routing picks `dense_prune` (head-centered top-k two-hop edge pruning) on dense graphs or `sparse_route` (relation-guided learnable edge weighting that preserves connectivity) on sparse graphs, based on graph statistics.
- **I3 – Internalized coarse-to-fine reasoning**: the coarse stage is a lightweight head on the shared encoder, used at inference for a top-k + delta dynamic decision — no external coarse ranker is trained.
- **I4 – Structure-prior-driven relation-conditioned ranking correction**: at inference only, the fine score is corrected by a relation-conditioned domain-boost term and hub-suppression term whose coefficients are tuned on the validation set.

This repository contains the **model implementation only**. Training, data-loading and evaluation code are not included.

## Repository layout

```text
.
├── example.py        # build the model and run a dummy forward pass
├── Our/
│   ├── model.py      # SICR-Graph model (dual-pathway + I1–I4)
│   ├── common.py     # model-side types: config, Graph/DatasetInfo, registry, profiler
│   └── rspmm/        # relation-aware sparse message passing
│       └── source/   # optional C++/CUDA extension sources
└── requirements.txt
```

## Requirements

- Python 3.10 (other 3.9+ versions should work)
- PyTorch 2.1.x + NumPy 1.26

```bash
pip install -r requirements.txt
```

The default `rspmm` backend is the pure-PyTorch `native` path, so no CUDA toolchain or compiled extension is required. Setting the `C3R_GRAPH_RSPMM_BACKEND` environment variable to `extension` JIT-compiles the bundled C++/CUDA sources for faster message passing.

## Quick start

```bash
python example.py
```

`example.py` builds the model on a small synthetic graph, prints the parameter count and runs a forward pass on random inputs:

```python
from Our.common import TASK_INDUCTIVE, DatasetInfo, DuetGraphConfig, Graph
from Our.model import OurDuetGraphModel

info = DatasetInfo(name="synthetic", task_type=TASK_INDUCTIVE, num_relations=4,
                   train_num_entities=50, eval_num_entities=50, use_entity_embeddings=True)
model = OurDuetGraphModel(info, DuetGraphConfig())
```

`batch` is a dict with `graph` (a `Graph` holding an `[E, 3]` triplet tensor of `(head, relation, tail)` indices), `h_index` and `r_index` (query head / relation indices), and an optional `graph_mask`.

## Citation

If you use this code, please cite:

```bibtex
@article{yin2026sicrgraph,
  author  = {Yin, Jie and Li, Donghuan and Lu, Keke},
  title   = {Structure-Aware Internalized Coarse-to-Fine Reasoning for Inductive Knowledge Graph},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is released under the [MIT License](LICENSE).