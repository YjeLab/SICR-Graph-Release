# SICR-Graph

Official implementation of **SICR-Graph** (Structure-Aware Internalized Coarse-to-Fine Reasoning), accompanying the paper *"结构感知内化的归纳知识图谱粗到细推理方法 / Structure-Aware Internalized Coarse-to-Fine Reasoning for Inductive Knowledge Graph"* (under review).

SICR-Graph performs inductive knowledge-graph reasoning with a fully internalized coarse-to-fine pipeline on top of a dual-pathway (local message passing + global linear attention) backbone:

- **I1 – Query-aware node-level fusion**: a per-node scalar gate conditioned on the query relation and head state fuses the local and global channels, replacing the global scalar fusion of the backbone.
- **I2 – Structure-adaptive local processing**: an automatic, non-learned routing picks `dense_prune` (head-centered top-k two-hop edge pruning) on dense graphs or `sparse_route` (relation-guided learnable edge weighting that preserves connectivity) on sparse graphs, based on graph statistics.
- **I3 – Internalized coarse-to-fine reasoning**: the coarse stage is a lightweight head on the shared encoder, trained jointly (`L = L_fine + λ·L_coarse`) and used at inference for a top-k + delta dynamic decision — no external coarse ranker is trained.
- **I4 – Structure-prior-driven relation-conditioned ranking correction**: at inference only, the fine score is corrected by a relation-conditioned domain-boost term and hub-suppression term whose coefficients are tuned on the validation set.

## Repository layout

```text
.
├── main.py               # unified entry (config + CLI)
├── Our/
│   ├── model.py          # SICR-Graph model (dual-pathway + I1–I4)
│   └── rspmm/            # relation-aware sparse message passing
│       └── source/       # optional C++/CUDA extension sources
└── Utils/                # config, data loading, tasks, trainer, metrics
```

## Requirements

- Python 3.10 (other 3.9+ versions should work)
- PyTorch 2.1.x + NumPy 1.26 + tqdm

```bash
pip install -r requirements.txt
```

The default `rspmm` backend is the pure-PyTorch `native` path, so no CUDA toolchain or compiled extension is required. Set `runtime.rspmm_backend = "extension"` (or the `C3R_GRAPH_RSPMM_BACKEND` environment variable) to JIT-compile the bundled C++/CUDA sources for faster message passing.

## Data preparation

Datasets are **not** distributed with this repository. The loader expects the standard text layout used by the GraIL / NBFNet inductive benchmark:

```text
Data/
└── inductive/
    ├── fb15k-237_v1/          # transductive (training) graph
    │   ├── entities.txt
    │   ├── relations.txt
    │   ├── train.txt
    │   ├── valid.txt
    │   └── test.txt
    └── fb15k-237_v1_ind/      # inductive (inference) graph, same files
```

Supported inductive splits (registered in `Utils/config.py`): `fb15k-237_v1..v4`, `wn18rr_v1..v4`, `nell-995_v1..v4`. These are the widely used GraIL-style inductive splits; you can obtain them from the public releases of NBFNet / RED-GNN / GraIL and convert to the layout above (`entities.txt`/`relations.txt` are `name<TAB>id` mappings, triplet files are `head<TAB>relation<TAB>tail`).

The dataset root defaults to `./Data` and can be changed via `config.data.root`.

## Reproduce paper results

Edit `build_manual_config()` in `main.py` or pass CLI flags. A full training + evaluation run:

```bash
python main.py --task-type inductive_reasoning --dataset fb15k-237_v1 --stage train
```

Checkpoints, logs and metrics are written under `Checkpoints/` (models / logs / results).

To evaluate a released checkpoint without retraining:

```bash
python main.py --task-type inductive_reasoning --dataset fb15k-237_v1 \
    --stage test --test-checkpoint checkpoints/fb15k-237_v1.pt
```

(see `runtime.test_checkpoint` / `runtime.resume_from` in `main.py`).

## Ablation switches

Component toggles used in the paper's ablation study:

| Module | How to disable |
|---|---|
| I1 query-aware fusion | `--innovation1-disable` |
| I2 structure-adaptive local processing | `--innovation2-disable` or `--innovation2-mode off|dense_prune|sparse_route|auto` |
| I3 internalized coarse-to-fine | `--innovation3-disable` (and `--innovation3-no-joint-loss`, `--innovation3-no-delta-decision` for the sub-controls) |
| I4 ranking correction | set `use_innovation4 = False` in `build_manual_config()` |

Reported configurations: `hidden_dim = 32`, `num_layer = 3`, `num_qk_layer = 2`, `num_v_layer = 3`, `num_heads = 4`, `epochs = 20`, `seed = 42`, deterministic mode with CPU negative sampling.

## Pre-trained checkpoints

Trained `best.pt` files for the reported inductive runs are attached to the [Releases](../../releases) page (`sicr-graph-checkpoints.zip`). Each archive preserves the `dataset_split/best.pt` layout; point `--test-checkpoint` at the file you want to evaluate.

## Notes on provenance

- The dual-pathway backbone and experiment framework build upon the **DuetGraph** implementation (Li et al., NeurIPS 2025).
- `Our/rspmm/source/` is adapted from the RSPMM operator used by NBFNet/ULTRA-style codebases, which in turn is inspired by [GE-SpMM](https://github.com/hgyhungry/ge-spmm).
- Baselines (DRUM, InGram, A*Net, NBFNet, RED-GNN, AdaProp, SimKGC, ULTRA, KnowFormer, DuetGraph, MERRY, SKILL) are **not** included; see the paper for protocol details.

## Citation

If you find this code useful, please cite the paper (citation information will be updated upon publication).

## License

MIT, see [LICENSE](LICENSE).
