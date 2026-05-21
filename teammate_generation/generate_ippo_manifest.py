#!/usr/bin/env python3
"""Sample IPPO teammate policies and generate evaluation manifests.

Discovers IPPO policies trained with run_ippo_seeds.sh, filters by return,
samples randomly, and generates a manifest for evaluation.

Usage:
    python scripts/sample_ippo_teammates.py \
        --layout grounded_coord_ring \
        --num_teammates 10

    python scripts/sample_ippo_teammates.py \
        --layout grounded_coord_ring \
        --num_teammates 10 \
        --sample_seed 0 \
        --min_return 50.0 \
        -v
"""

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.manifest_schema import (
    TaskEntry,
    TeammateSpec,
    generate_task_id,
    save_manifest,
    validate_manifest,
)
from envs.overcooked_v2 import overcooked_v2_layouts


@dataclass
class IPPOPolicy:
    """IPPO policy checkpoint info."""
    run_dir: str
    seed: int
    seed_idx: int
    ckpt_idx: int
    base_return: float

    def to_extra(self) -> Dict[str, Any]:
        return {
            "actor_type": "cnn_rnn",
            "activation": "relu",
            "fc_dim_size": 128,
            "gru_hidden_dim": 128,
            "use_separated_ckpt": True,
            "checkpoint_idx": self.ckpt_idx,
            "population_idx": 0,
            "seed_idx": self.seed_idx,
        }


def discover_policies(layout: str, model_dir: str = "mate_rl_models") -> List[IPPOPolicy]:
    """Find all IPPO policies for a layout."""
    pattern = os.path.join(model_dir, f"ippo_{layout}_seed*", "ippo_train_run")
    policies = []

    for run_dir in glob.glob(pattern):
        returns_path = os.path.join(run_dir, "checkpoint_returns.json")
        if not os.path.exists(returns_path):
            continue

        # Extract seed from path
        parent = os.path.basename(os.path.dirname(run_dir))
        seed = int(parent.split("_seed")[-1]) if "_seed" in parent else 0

        with open(returns_path) as f:
            data = json.load(f)

        for seed_idx, returns in enumerate(data.get("base_returns", [])):
            for ckpt_idx, base_return in enumerate(returns):
                policies.append(IPPOPolicy(
                    run_dir=run_dir,
                    seed=seed,
                    seed_idx=seed_idx,
                    ckpt_idx=ckpt_idx,
                    base_return=base_return,
                ))

    return policies


def generate_entries(
    policies: List[IPPOPolicy],
    layout: str,
    track: str,
    split: str,
    base_seed: int,
) -> List[TaskEntry]:
    """Generate manifest entries from policies."""
    entries = []
    for policy in policies:
        spec = TeammateSpec.rl(family="ippo", ckpt=policy.run_dir, extra=policy.to_extra())
        task_id = generate_task_id(track, split, layout, spec, base_seed)
        entries.append(TaskEntry(
            task_id=task_id,
            track=track,
            split=split,
            layout_name=layout,
            seed=base_seed,
            teammate=spec,
        ))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Sample IPPO policies and generate manifest.")
    parser.add_argument("--layout", required=True, help="Layout name")
    parser.add_argument("--num_teammates", type=int, default=20, help="Number to sample")
    parser.add_argument("--output_dir", default="benchmarks/overcooked_icrl", help="Output directory")
    parser.add_argument("--model_dir", default="mate_rl_models", help="Model directory")
    parser.add_argument("--sample_seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--min_return", type=float, default=20, help="Min return threshold")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--track", default="teammate", choices=["teammate", "layout"])
    parser.add_argument("--base_seed", type=int, default=0, help="Task seed")
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbose output")
    args = parser.parse_args()

    if args.layout not in overcooked_v2_layouts:
        print(f"Error: Unknown layout '{args.layout}'", file=sys.stderr)
        return 1

    # Discover and filter
    all_policies = discover_policies(args.layout, args.model_dir)
    filtered = [p for p in all_policies if p.base_return > args.min_return]

    print(f"Found {len(all_policies)} policies, {len(filtered)} after filtering (return > {args.min_return})")

    if not filtered:
        print("Error: No policies found", file=sys.stderr)
        return 1

    # Sample
    if args.sample_seed is not None:
        random.seed(args.sample_seed)
    sampled = random.sample(filtered, min(args.num_teammates, len(filtered)))

    if args.verbose:
        for p in sampled:
            print(f"  seed={p.seed}, ckpt={p.ckpt_idx}, return={p.base_return:.2f}")

    # Generate and save manifest
    entries = generate_entries(sampled, args.layout, args.track, args.split, args.base_seed)

    errors = validate_manifest(entries)
    if errors:
        print(f"Validation errors: {errors}", file=sys.stderr)
        return 1

    out_path = Path(args.output_dir) / f"track_{args.track}" / args.layout / "ippo"
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / f"manifest_{args.split}.jsonl"
    save_manifest(entries, str(manifest_path))

    print(f"Saved {len(entries)} entries to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
