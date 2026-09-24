# ICRL4AHT: In-Context Reinforcement Learning for Ad Hoc Teamwork

A JAX-based benchmark and dataset-generation pipeline for evaluating In-Context Reinforcement Learning (ICRL) algorithms in cooperative multi-agent settings.

## Overview

**ICRL4AHT** addresses the challenge of *ad-hoc teamwork*: enabling an agent to cooperate with previously unseen teammates without pre-coordination. This repository provides the benchmark code, teammate-generation pipeline, and evaluation scripts used in the paper. The benchmark uses the **Overcooked-V2** cooperative cooking environment and provides a rigorous evaluation pipeline for ICRL methods.

### Key Features

- **Four ICRL Baselines**: Algorithm Distillation (AD), Decision Pretrained Transformer (DPT), AMAGO-offline, and Hybrid-AD
- **Diverse Teammate Policies**: Both RL-trained (IPPO, FCP, BRDiv, LBRDiv, CoMeDi, and etc.) and hand-crafted heuristic agents
- **Two Evaluation Tracks**: Teammate generalization and layout generalization
- **High-Performance JAX/Flax Implementation**: Vectorized training and GPU-accelerated evaluation

## Installation

```bash
# Clone the repository
git clone this_repo_url
cd icrl4aht

# Install dependencies (requires Python >= 3.11)
pip install -e .
```

**Note**: For GPU support, ensure CUDA 12.x is installed. For CPU-only, modify `pyproject.toml` to use `jax` instead of `jax[cuda12]`.

## Pipeline Overview

The benchmark follows a 6-phase pipeline:

```
Phase 1: Train Teammate Policies (IPPO/FCP/BRDiv/LBRDiv/CoMeDi/...)
    │
    ▼
Phase 2: Generate Task Manifests (JSONL)
    │
    ▼
Phase 3: Collect Learning Histories
    │
    ▼
Phase 4: Build HDF5 Dataset
    │
    ▼
Phase 5: Train ICRL Models (AD/DPT/AMAGO-offline/Hybrid-AD)
    │
    ▼
Phase 6: Evaluate on Test Teammates
```

### IPPO vs FCP Training

Running `train_fcp_overcooked_v2.py` with `--partner_pop_size=N` is equivalent to running `train_ippo_overcooked_v2.py` N times with different seeds. FCP internally vmaps N independent IPPO training runs, producing a population of N diverse policies in a single execution. This parallelization leverages JAX's vectorization for efficient population-based training.

### BRDiv / LBRDiv / CoMeDi Training

**BRDiv** (Best Response Diversity) and **LBRDiv** (Lagrangian Best Response Diversity) train populations of confederate-ego policy pairs to maximize behavioral diversity. LBRDiv extends BRDiv by using Lagrangian relaxation with learned multipliers to better balance self-play and cross-play objectives. **CoMeDi** (Cooperative Meta-Diversity) builds a population sequentially, adding one agent at a time to maximize diversity via a combined loss over cross-play, self-play, and mixed-play rollouts.

## Quick Start

### 1. Train Teammate Policies

```bash
# Train a single IPPO policy
python -m teammate_generation.train_ippo_overcooked_v2 \
    --layout grounded_coord_ring \
    --seed 0 \
    --output_dir outputs/ippo_seed0

# Or train an FCP population (10 policies at once)
python -m teammate_generation.train_fcp_overcooked_v2 \
    --layout grounded_coord_ring \
    --partner_pop_size 10 \
    --output_dir outputs/fcp_population

# Train an LBRDiv population (Lagrangian Best Response Diversity)
python -m teammate_generation.train_lbrdiv_overcooked_v2 \
    --layout grounded_coord_ring \
    --partner_pop_size 5 \
    --output_dir outputs/lbrdiv_population

# Train a CoMeDi population (Cooperative Meta-Diversity)
python -m teammate_generation.train_comedi_overcooked_v2 \
    --layout grounded_coord_ring \
    --partner_pop_size 5 \
    --output_dir outputs/comedi_population
```

### 2. Generate Task Manifests

The training manifests (`manifest_train.jsonl`) are **not** included in this repository. They must be generated from the trained teammate checkpoints produced in Phase 1:

```bash
python -m teammate_generation.generate_ippo_manifest \
    --layout grounded_coord_ring \
    --model_dir outputs/fcp_population \
    --split train \
    --track teammate
```

The script writes `manifest_train.jsonl` under `benchmarks/overcooked_icrl/` (configurable via `--output_dir`). Test manifests (`manifest_test.jsonl`) for heuristic teammates are already included.

### 3. Collect Learning Histories

```bash
python -m scripts.collect_histories \
    --manifest benchmarks/overcooked_icrl/track_teammate/grounded_coord_ring/ippo/manifest_train.jsonl \
    --out_dir outputs/histories \
    --num_workers 4
```

### 4. Build Dataset

```bash
python -m scripts.build_index outputs/histories --relabel
```

### 5. Train ICRL Models

```bash
# Algorithm Distillation
python -m benchmarks.baselines.ad.train \
    --h5_path outputs/histories/histories.h5 \
    --index_path outputs/histories/histories_index.jsonl \
    --out_dir outputs/ad_model

# Decision Pretrained Transformer
python -m benchmarks.baselines.dpt.train \
    --h5_path outputs/histories/histories.h5 \
    --index_path outputs/histories/histories_index.jsonl \
    --out_dir outputs/dpt_model

# AMAGO-offline (GRU trajectory encoder by default)
python -m benchmarks.baselines.amago_offline.train \
    --h5_path outputs/histories/histories.h5 \
    --index_path outputs/histories/histories_index.jsonl \
    --out_dir outputs/amago_offline_model

# Hybrid-AD (AD with CNN + GRU backbone)
python -m benchmarks.baselines.hybrid_ad.train \
    --h5_path outputs/histories/histories.h5 \
    --index_path outputs/histories/histories_index.jsonl \
    --out_dir outputs/hybrid_ad_model
```

### 6. Evaluate

```bash
# Evaluate all ICRL baselines
python eval_icrl.py \
    --algo ad,dpt,amago_offline,hybrid_ad,random \
    --tracks teammate,layout \
    --episodes 100 \
    --checkpoint_ad outputs/ad_model/checkpoints/checkpoint_20000 \
    --checkpoint_dpt outputs/dpt_model/checkpoints/checkpoint_20000 \
    --checkpoint_amago_offline outputs/amago_offline_model/checkpoints/checkpoint_20000 \
    --checkpoint_hybrid_ad outputs/hybrid_ad_model/checkpoints/checkpoint_20000 \
    --out results/

# Evaluate AMAGO-offline only
python eval_icrl.py \
    --algo amago_offline \
    --tracks teammate,layout \
    --episodes 100 \
    --checkpoint_amago_offline outputs/amago_offline_model/checkpoints/checkpoint_20000 \
    --out results/amago_offline/

# Evaluate Hybrid-AD only
python eval_icrl.py \
    --algo hybrid_ad \
    --tracks teammate,layout \
    --episodes 100 \
    --checkpoint_hybrid_ad outputs/hybrid_ad_model/checkpoints/checkpoint_20000 \
    --out results/hybrid_ad/
```

## Project Structure

```
icrl4aht/
├── agents/                    # Agent implementations
│   └── overcooked_v2/         # Heuristic teammates (assembly_line, territory, etc.)
├── benchmarks/
│   ├── baselines/             # ICRL algorithms
│   │   ├── ad/                # Algorithm Distillation
│   │   ├── dpt/               # Decision Pretrained Transformer
│   │   ├── amago_offline/     # AMAGO-offline (GRU trajectory encoder)
│   │   └── hybrid_ad/         # Hybrid-AD (CNN + GRU backbone)
│   └── overcooked_icrl/       # Task manifests for evaluation tracks
├── envs/                      # Environment implementations
│   └── overcooked_v2/         # JAX-native Overcooked-V2
├── marl/                      # Multi-agent RL (IPPO implementation)
├── runners/                   # Training and data collection utilities
├── scripts/                   # Data collection and preprocessing
├── teammate_generation/       # Teammate policy training scripts
├── teammate_wrapper/          # Unified teammate interface
├── eval_icrl.py               # Main evaluation script
└── pyproject.toml             # Project dependencies
```

## Evaluation Tracks

| Track | Description |
|-------|-------------|
| **Teammate** | Test adaptation to unseen heuristic teammates on familiar layouts |
| **Layout** | Test adaptation to novel layouts with unseen teammates |

### Heuristic Teammate Families

| Family | Behavior |
|--------|----------|
| **Assembly Line** | Role-based task division (runner vs. plater) |
| **Territory** | Spatial region specialization |
| **Utility Greedy** | Weighted utility-based action selection |
| **Recipe Aware Button** | Recipe-conditioned behavior |

## Acknowledgements

Parts of the JAX-based teammate generation pipeline in ICRL4AHT build on the open-source [JaxAHT](https://github.com/LARG/jax-aht) codebase. We thank the JaxAHT authors for making their implementations publicly available.

The JaxAHT MIT license and copyright notice are included in [`third_party/JaxAHT_LICENSE`](third_party/JaxAHT_LICENSE).

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{
jing2026benchmarking,
title={Benchmarking the Limits of In-Context Reinforcement Learning for Ad-Hoc Teamwork},
author={Yuheng Jing and Kai Li and Jiajun Zhang and Zeyao Ma and Jiaxi Yang and Lei Zhang and Zhe Wu and Jinmin He and Junliang Xing and Jian Cheng},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=EbkumuY3eW}
}
```

The JAX-based teammate generation components build on JaxAHT. Please also cite:

```bibtex
@misc{wang2026jaxaht,
      title={JaxAHT: A JAX-Based Library for Ad Hoc Teamwork},
      author={Caroline Wang and Rolando Fernandez and Zelal Su Mustafaoglu and Montek Kundan and Jiaxun Cui and Lingyun Xiao and Zhihan Wang and Di Yang Shi and Aditya Madhan and Johnny Liu and Arrasy Rahman and Peter Stone},
      year={2026},
      eprint={2609.13716},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.13716},
}
```
