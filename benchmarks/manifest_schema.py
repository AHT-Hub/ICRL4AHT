"""Manifest Schema for ICRL Benchmark Task Definitions.

This module provides dataclasses for defining task entries in benchmark manifests.
Each task entry describes a single evaluation task including:
- Layout configuration
- Teammate specification (heuristic or RL)
- Environment settings
- Track and split metadata

The schema is designed to be JSON-serializable for storage in JSONL manifest files.

Example Usage:
    >>> from benchmarks.manifest_schema import TaskEntry, TeammateSpec
    >>>
    >>> # Create a heuristic teammate task
    >>> task = TaskEntry(
    ...     task_id="teammate_train_cramped_room_assembly_line_0_s42",
    ...     track="teammate",
    ...     split="train",
    ...     layout_name="cramped_room",
    ...     seed=0,
    ...     teammate=TeammateSpec.heuristic("assembly_line", theta_id=0, base_seed=0),
    ... )
    >>>
    >>> # Serialize to JSON
    >>> json_dict = task.to_json()
    >>>
    >>> # Load from JSON
    >>> loaded = TaskEntry.from_json(json_dict)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Union
import hashlib
import json


# Valid track names
VALID_TRACKS = frozenset({"teammate", "layout"})

# Valid split names
VALID_SPLITS = frozenset({"train", "test"})


@dataclass
class TeammateSpec:
    """Specification for a teammate in a task.

    This is a unified spec that can represent either heuristic or RL teammates.

    Attributes:
        kind: "heuristic" or "rl"
        family: For heuristic: family name; for RL: algorithm name (fcp/brdiv)
        theta_id: For heuristic: theta configuration ID
        theta: For heuristic: optional expanded theta dict
        ckpt: For RL: checkpoint path or ID
        base_seed: Base seed for theta sampling
        extra: Additional configuration
    """
    kind: Literal["heuristic", "rl"]
    family: str
    theta_id: Optional[int] = None
    theta: Optional[Dict[str, Any]] = None
    ckpt: Optional[str] = None
    base_seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the spec."""
        if self.kind not in ("heuristic", "rl"):
            raise ValueError(f"kind must be 'heuristic' or 'rl', got '{self.kind}'")

        if self.kind == "heuristic":
            if self.theta_id is None:
                raise ValueError("theta_id is required for heuristic teammates")
        elif self.kind == "rl":
            if self.ckpt is None:
                raise ValueError("ckpt is required for RL teammates")

    @classmethod
    def heuristic(
        cls,
        family: str,
        theta_id: int,
        base_seed: int = 0,
        theta: Optional[Dict[str, Any]] = None,
    ) -> "TeammateSpec":
        """Create a heuristic teammate spec."""
        return cls(
            kind="heuristic",
            family=family,
            theta_id=theta_id,
            theta=theta,
            base_seed=base_seed,
        )

    @classmethod
    def rl(
        cls,
        family: str,
        ckpt: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "TeammateSpec":
        """Create an RL teammate spec.

        Args:
            family: "fcp", "brdiv", or "ippo"
            ckpt: Checkpoint path or run directory
            extra: Additional config:
                Network architecture:
                - "actor_type": "mlp" | "rnn" | "cnn_rnn" | "s5" (default: "mlp")
                - "activation": Activation function (default: "tanh")
                - "fc_dim_size": FC dimension for cnn_rnn (default: 128)
                - "gru_hidden_dim": GRU hidden dimension (default: 64/128)

                Checkpoint loading (separated format):
                - "use_separated_ckpt": Use separated checkpoint format (default: False)
                - "checkpoint_idx": Checkpoint index to load (default: -1 for last)
                - "population_idx": Population index for FCP/IPPO (default: 0)
                - "seed_idx": Seed index for multi-seed runs (default: 0)
                - "agent_type": For BRDiv, "conf" or "br" (default: "conf")
        """
        return cls(
            kind="rl",
            family=family,
            ckpt=ckpt,
            extra=extra or {},
        )

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "kind": self.kind,
            "family": self.family,
        }

        if self.kind == "heuristic":
            result["theta_id"] = self.theta_id
            result["base_seed"] = self.base_seed
            if self.theta is not None:
                result["theta"] = self.theta
        else:  # rl
            result["ckpt"] = self.ckpt
            if self.extra:
                result["extra"] = self.extra

        return result

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "TeammateSpec":
        """Create from JSON dict."""
        kind = d["kind"]
        if kind == "heuristic":
            return cls.heuristic(
                family=d["family"],
                theta_id=d["theta_id"],
                base_seed=d.get("base_seed", 0),
                theta=d.get("theta"),
            )
        else:
            return cls.rl(
                family=d["family"],
                ckpt=d["ckpt"],
                extra=d.get("extra"),
            )

    def get_display_name(self) -> str:
        """Get a human-readable display name."""
        if self.kind == "heuristic":
            return f"{self.family}[theta={self.theta_id}]"
        else:
            return f"{self.family}[{self.ckpt}]"


@dataclass
class TaskEntry:
    """A single task entry in a benchmark manifest.

    Attributes:
        task_id: Stable unique identifier for this task
        track: Benchmark track ("teammate" | "layout")
        split: Data split ("train" | "test")
        layout_name: Name of the layout to use
        seed: Random seed for this task
        teammate: Teammate specification
    """
    task_id: str
    track: Literal["teammate", "layout"]
    split: Literal["train", "test"]
    layout_name: str
    seed: int
    teammate: TeammateSpec

    def __post_init__(self):
        """Validate the entry."""
        if self.track not in VALID_TRACKS:
            raise ValueError(f"Invalid track '{self.track}'. Must be one of {VALID_TRACKS}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of {VALID_SPLITS}")

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "task_id": self.task_id,
            "track": self.track,
            "split": self.split,
            "layout_name": self.layout_name,
            "seed": self.seed,
            "teammate": self.teammate.to_json(),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "TaskEntry":
        """Create from JSON dict."""
        return cls(
            task_id=d["task_id"],
            track=d["track"],
            split=d["split"],
            layout_name=d["layout_name"],
            seed=d["seed"],
            teammate=TeammateSpec.from_json(d["teammate"]),
        )

    def to_json_line(self) -> str:
        """Convert to a single JSONL line."""
        return json.dumps(self.to_json(), sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> "TaskEntry":
        """Create from a JSONL line."""
        return cls.from_json(json.loads(line))


def generate_task_id(
    track: str,
    split: str,
    layout_name: str,
    teammate_spec: TeammateSpec,
    seed: int,
) -> str:
    """Generate a stable, deterministic task ID.

    The ID format is:
        {track}_{split}_{layout}_{teammate_key}_{seed_suffix}

    Where teammate_key is:
        - For heuristic: {family}_t{theta_id}
        - For RL: {family}_{ckpt_basename}

    Args:
        track: Track name
        split: Split name
        layout_name: Layout name
        teammate_spec: Teammate specification
        seed: Task seed

    Returns:
        A stable task ID string
    """
    # Build teammate key
    if teammate_spec.kind == "heuristic":
        teammate_key = f"{teammate_spec.family}_t{teammate_spec.theta_id}"
    else:
        # Use basename of checkpoint for RL
        import os
        ckpt_base = os.path.basename(teammate_spec.ckpt.rstrip('/\\'))
        teammate_key = f"{teammate_spec.family}_{ckpt_base}"

    # Build raw ID
    raw_id = f"{track}_{split}_{layout_name}_{teammate_key}_s{seed}"

    # Create a short hash suffix for uniqueness verification
    hash_input = f"{track}:{split}:{layout_name}:{teammate_spec.to_json()}:{seed}"
    hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    return f"{raw_id}_{hash_suffix}"


def load_manifest(filepath: str) -> List[TaskEntry]:
    """Load a manifest from a JSONL file.

    Args:
        filepath: Path to the JSONL manifest file

    Returns:
        List of TaskEntry objects
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(TaskEntry.from_json_line(line))
    return entries


def save_manifest(entries: List[TaskEntry], filepath: str) -> None:
    """Save a manifest to a JSONL file.

    Entries are sorted by task_id for deterministic output.

    Args:
        entries: List of TaskEntry objects
        filepath: Path to write the JSONL file
    """
    # Sort for deterministic output
    sorted_entries = sorted(entries, key=lambda e: e.task_id)

    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in sorted_entries:
            f.write(entry.to_json_line() + '\n')


def validate_manifest(entries: List[TaskEntry]) -> List[str]:
    """Validate a manifest for correctness.

    Checks:
    - No duplicate task_ids
    - All entries have valid tracks and splits
    - Consistent teammate specs

    Args:
        entries: List of TaskEntry objects

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check for duplicates
    task_ids = [e.task_id for e in entries]
    seen = set()
    for tid in task_ids:
        if tid in seen:
            errors.append(f"Duplicate task_id: {tid}")
        seen.add(tid)

    # Validate each entry
    for entry in entries:
        if entry.track not in VALID_TRACKS:
            errors.append(f"Invalid track '{entry.track}' in task {entry.task_id}")
        if entry.split not in VALID_SPLITS:
            errors.append(f"Invalid split '{entry.split}' in task {entry.task_id}")

    return errors


def summarize_manifest(entries: List[TaskEntry]) -> Dict[str, Any]:
    """Generate a summary of a manifest.

    Args:
        entries: List of TaskEntry objects

    Returns:
        Dict with summary statistics
    """
    from collections import defaultdict

    summary = {
        "total_tasks": len(entries),
        "by_track": defaultdict(int),
        "by_split": defaultdict(int),
        "by_track_split": defaultdict(int),
        "by_layout": defaultdict(int),
        "by_family": defaultdict(int),
        "by_kind": defaultdict(int),
    }

    for entry in entries:
        summary["by_track"][entry.track] += 1
        summary["by_split"][entry.split] += 1
        summary["by_track_split"][f"{entry.track}_{entry.split}"] += 1
        summary["by_layout"][entry.layout_name] += 1
        summary["by_family"][entry.teammate.family] += 1
        summary["by_kind"][entry.teammate.kind] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    return {
        "total_tasks": summary["total_tasks"],
        "by_track": dict(summary["by_track"]),
        "by_split": dict(summary["by_split"]),
        "by_track_split": dict(summary["by_track_split"]),
        "by_layout": dict(summary["by_layout"]),
        "by_family": dict(summary["by_family"]),
        "by_kind": dict(summary["by_kind"]),
    }
