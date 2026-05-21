#!/usr/bin/env python3
"""Collect histories: Batch runner for training tasks in a manifest.

This script runs ALL tasks in a train manifest end-to-end using TaskRunner,
producing per-task history artifacts on disk with robust resume/checkpoint
behavior (idempotent; safe to rerun).

The script:
1. Loads tasks from a JSONL manifest
2. Checks which tasks are already complete (skip them)
3. Runs incomplete tasks using TaskRunner
4. Supports parallel execution with lock files to prevent duplicates
5. Logs progress and failures to structured JSONL files

Usage:
    # Run all tasks in a manifest
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs

    # Run only specific tasks
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs \
        --task_indices "0,3,7-12"

    # Run with parallel workers
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs \
        --num_workers 4

    # Run on specific GPU
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs \
        --gpu 0

    # Distribute tasks across multiple GPUs (round-robin)
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs \
        --num_workers 4 \
        --gpu 0,1,2,3

    # Dry run (show what would be done)
    python -m scripts.collect_histories \
        --manifest path/to/manifest_train.jsonl \
        --out_dir outputs/task_runs \
        --dry_run

Output Structure:
    {out_dir}/
        {task_id}/
            history.npz
            episodes.json
            metadata.json
            final_ckpt/ (optional)
        _collect_runs/
            {timestamp}/
                run_config.json
                progress.jsonl
                failed.jsonl

Resume Semantics:
    - A task is COMPLETE if history.npz, episodes.json, and metadata.json exist
      and are readable/valid
    - Complete tasks are skipped automatically
    - Incomplete tasks are rerun from scratch (outputs deleted before rerun)
    - Lock files prevent concurrent workers from processing the same task
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np

from benchmarks.manifest_schema import TaskEntry, load_manifest

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Required artifact files for a task to be considered complete
REQUIRED_ARTIFACTS = ["history.npz", "episodes.json", "metadata.json"]

# Lock file name
LOCK_FILE_NAME = ".lock"


# =============================================================================
# Task Completeness Checking
# =============================================================================

def is_task_complete(task_dir: Union[str, Path]) -> bool:
    """Check if a task directory has all required artifacts.

    A task is considered COMPLETE if:
    1. history.npz exists and is readable as a valid npz file
    2. episodes.json exists and is valid JSON
    3. metadata.json exists and is valid JSON

    Args:
        task_dir: Path to the task output directory

    Returns:
        True if task is complete, False otherwise
    """
    task_dir = Path(task_dir)

    if not task_dir.exists():
        return False

    try:
        validate_task_outputs(task_dir)
        return True
    except TaskOutputError:
        return False


class TaskOutputError(Exception):
    """Exception raised when task outputs are invalid or incomplete."""
    pass


def validate_task_outputs(task_dir: Union[str, Path]) -> None:
    """Validate that a task directory has all required artifacts.

    Raises TaskOutputError if any artifact is missing or invalid.

    Args:
        task_dir: Path to the task output directory

    Raises:
        TaskOutputError: If any required file is missing or invalid
    """
    task_dir = Path(task_dir)

    if not task_dir.exists():
        raise TaskOutputError(f"Task directory does not exist: {task_dir}")

    # Check history.npz
    history_path = task_dir / "history.npz"
    if not history_path.exists():
        raise TaskOutputError(f"Missing required file: history.npz")
    try:
        with np.load(history_path) as f:
            # Just verify it's readable
            keys = list(f.keys())
            if not keys:
                raise TaskOutputError(f"history.npz is empty")

            # Check if this is a chunked history (incremental saving mode)
            is_chunked = "_chunked" in keys and f["_chunked"][0]

            if is_chunked:
                # For chunked histories, validate chunks directory exists
                chunks_dir = task_dir / "chunks"
                if not chunks_dir.exists():
                    raise TaskOutputError(f"Chunked history missing chunks directory")

                # Verify at least one chunk exists
                chunk_files = list(chunks_dir.glob("chunk_*.npz"))
                chunk_files = [f for f in chunk_files if "_episodes" not in f.name]
                if not chunk_files:
                    raise TaskOutputError(f"Chunked history has no chunk files")

                # Verify the first chunk has required arrays
                with np.load(chunk_files[0]) as chunk:
                    chunk_keys = list(chunk.keys())
                    for required_key in ["obs_t", "act_t", "rew_t", "done_t"]:
                        if required_key not in chunk_keys:
                            raise TaskOutputError(f"Chunk {chunk_files[0].name} missing required array: {required_key}")
            else:
                # For non-chunked histories, check required arrays exist in main file
                for required_key in ["obs_t", "act_t", "rew_t", "done_t"]:
                    if required_key not in keys:
                        raise TaskOutputError(f"history.npz missing required array: {required_key}")
    except Exception as e:
        if isinstance(e, TaskOutputError):
            raise
        raise TaskOutputError(f"Cannot read history.npz: {e}")

    # Check episodes.json
    episodes_path = task_dir / "episodes.json"
    if not episodes_path.exists():
        raise TaskOutputError(f"Missing required file: episodes.json")
    try:
        with open(episodes_path, "r") as f:
            episodes_data = json.load(f)
        if "episodes" not in episodes_data:
            raise TaskOutputError(f"episodes.json missing 'episodes' key")
    except json.JSONDecodeError as e:
        raise TaskOutputError(f"Invalid JSON in episodes.json: {e}")
    except Exception as e:
        if isinstance(e, TaskOutputError):
            raise
        raise TaskOutputError(f"Cannot read episodes.json: {e}")

    # Check metadata.json
    metadata_path = task_dir / "metadata.json"
    if not metadata_path.exists():
        raise TaskOutputError(f"Missing required file: metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if "task_spec" not in metadata:
            raise TaskOutputError(f"metadata.json missing 'task_spec' key")
    except json.JSONDecodeError as e:
        raise TaskOutputError(f"Invalid JSON in metadata.json: {e}")
    except Exception as e:
        if isinstance(e, TaskOutputError):
            raise
        raise TaskOutputError(f"Cannot read metadata.json: {e}")


# =============================================================================
# Task Index Parsing
# =============================================================================

def parse_task_indices(indices_str: str, max_idx: int) -> List[int]:
    """Parse a task indices string into a list of indices.

    Supports formats:
        - "0,3,7" -> [0, 3, 7]
        - "0-5" -> [0, 1, 2, 3, 4, 5]
        - "0,3,7-12,15" -> [0, 3, 7, 8, 9, 10, 11, 12, 15]

    Args:
        indices_str: String specifying task indices
        max_idx: Maximum valid index (exclusive)

    Returns:
        Sorted list of unique indices
    """
    indices: Set[int] = set()

    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start.strip())
            end_idx = int(end.strip())
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            for i in range(start_idx, min(end_idx + 1, max_idx)):
                if 0 <= i < max_idx:
                    indices.add(i)
        else:
            idx = int(part)
            if 0 <= idx < max_idx:
                indices.add(idx)

    return sorted(indices)


# =============================================================================
# Lock File Management
# =============================================================================

def acquire_lock(task_dir: Path, timeout: float = 0.0) -> bool:
    """Try to acquire a lock for a task directory.

    Uses a simple lock file mechanism. The lock file contains the PID
    of the process that holds the lock.

    Args:
        task_dir: Path to the task directory
        timeout: Time to wait for lock (0 = no wait)

    Returns:
        True if lock acquired, False otherwise
    """
    lock_path = task_dir / LOCK_FILE_NAME
    task_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    while True:
        try:
            # Try to create lock file exclusively
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # Lock file exists - check if still valid
            if _is_lock_stale(lock_path):
                _remove_stale_lock(lock_path)
                continue
            if timeout > 0 and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                continue
            return False
        except OSError:
            return False


def release_lock(task_dir: Path) -> None:
    """Release the lock for a task directory.

    Args:
        task_dir: Path to the task directory
    """
    lock_path = task_dir / LOCK_FILE_NAME
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _is_lock_stale(lock_path: Path, max_age_seconds: float = 3600.0) -> bool:
    """Check if a lock file is stale.

    A lock is considered stale if:
    1. The file is older than max_age_seconds, OR
    2. The PID in the file is not a running process

    Args:
        lock_path: Path to the lock file
        max_age_seconds: Maximum age before lock is considered stale

    Returns:
        True if lock is stale, False otherwise
    """
    try:
        stat = lock_path.stat()
        age = time.time() - stat.st_mtime

        # Check age
        if age > max_age_seconds:
            return True

        # Check if PID is still running
        with open(lock_path, "r") as f:
            pid_str = f.read().strip()
            if pid_str:
                pid = int(pid_str)
                # On Unix, sending signal 0 checks if process exists
                try:
                    os.kill(pid, 0)
                    return False  # Process exists, lock is valid
                except (ProcessLookupError, PermissionError):
                    return True  # Process doesn't exist, lock is stale
                except OSError:
                    # On Windows, use a different approach
                    import subprocess
                    try:
                        result = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {pid}"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if str(pid) in result.stdout:
                            return False  # Process exists
                        return True  # Process doesn't exist
                    except Exception:
                        # If we can't check, assume not stale
                        return False

        return True  # Empty lock file is stale
    except Exception:
        return True


def _remove_stale_lock(lock_path: Path) -> None:
    """Remove a stale lock file.

    Args:
        lock_path: Path to the lock file
    """
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# =============================================================================
# Progress Logging
# =============================================================================

@dataclass
class RunConfig:
    """Configuration for a collection run."""
    manifest_path: str
    out_dir: str
    num_workers: int
    max_tasks: Optional[int]
    task_indices: Optional[str]
    fail_fast: bool
    retry: int
    dry_run: bool
    started_at: str
    total_tasks: int
    tasks_to_run: int


@dataclass
class TaskProgress:
    """Progress record for a single task."""
    idx: int
    task_uid: str
    status: str  # "skipped", "success", "failed", "locked"
    duration_s: Optional[float] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


class ProgressLogger:
    """Logger for tracking collection progress."""

    def __init__(self, log_dir: Path):
        """Initialize progress logger.

        Args:
            log_dir: Directory to write log files
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = log_dir / "run_config.json"
        self.progress_path = log_dir / "progress.jsonl"
        self.failed_path = log_dir / "failed.jsonl"

        # Initialize files
        self.progress_path.touch()
        self.failed_path.touch()

    def write_config(self, config: RunConfig) -> None:
        """Write run configuration."""
        with open(self.config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

    def log_progress(self, progress: TaskProgress) -> None:
        """Log a task progress record."""
        with open(self.progress_path, "a") as f:
            f.write(json.dumps(asdict(progress)) + "\n")

        if progress.status == "failed":
            with open(self.failed_path, "a") as f:
                f.write(json.dumps(asdict(progress)) + "\n")


# =============================================================================
# Task Runner Wrapper
# =============================================================================

def default_task_runner(
    task_entry: TaskEntry,
    out_dir: str,
    manifest_path: str,
    gpu: Optional[str] = None,
) -> str:
    """Default task runner using TaskRunner.

    Args:
        task_entry: Task entry from manifest
        out_dir: Output directory
        manifest_path: Path to manifest file (for logging)
        gpu: GPU device ID to use for this task

    Returns:
        Path to task output directory
    """
    from runners.task_runner import TaskRunner, TaskRunnerConfig

    config = TaskRunnerConfig(
        manifest_path=manifest_path,
        task_id=task_entry.task_id,
        out_dir=out_dir,
        gpu=gpu,
    )

    runner = TaskRunner(config)
    result = runner.run()
    return result["task_dir"]


# Type alias for runner callable
RunnerCallable = Callable[[TaskEntry, str, str, Optional[str]], str]


# =============================================================================
# Main Collection Logic
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for the history collector."""
    manifest_path: str
    out_dir: str
    num_workers: int = 1
    max_tasks: Optional[int] = None
    task_indices: Optional[str] = None
    fail_fast: bool = False
    retry: int = 0
    dry_run: bool = False
    gpu: Optional[str] = None  # GPU device ID(s) to use


@dataclass
class CollectorStats:
    """Statistics from a collection run."""
    total_tasks: int = 0
    tasks_to_run: int = 0
    skipped: int = 0
    success: int = 0
    failed: int = 0
    locked: int = 0


def run_one_task(
    idx: int,
    task_entry: TaskEntry,
    out_dir: str,
    manifest_path: str,
    dry_run: bool = False,
    retry: int = 0,
    runner: Optional[RunnerCallable] = None,
    gpu: Optional[str] = None,
) -> TaskProgress:
    """Run a single task with retry support.

    Args:
        idx: Task index in manifest
        task_entry: Task entry from manifest
        out_dir: Output directory
        manifest_path: Path to manifest file
        dry_run: If True, don't actually run
        retry: Number of retry attempts
        runner: Optional custom runner callable
        gpu: GPU device ID to use for this task

    Returns:
        TaskProgress record
    """
    task_uid = task_entry.task_id
    task_dir = Path(out_dir) / task_uid

    # Use default runner if not provided
    if runner is None:
        runner = default_task_runner

    # Check if already complete
    if is_task_complete(task_dir):
        return TaskProgress(idx=idx, task_uid=task_uid, status="skipped")

    # Dry run - just report what would happen
    if dry_run:
        return TaskProgress(idx=idx, task_uid=task_uid, status="dry_run")

    # Try to acquire lock
    if not acquire_lock(task_dir):
        return TaskProgress(idx=idx, task_uid=task_uid, status="locked")

    start_time = time.time()
    last_error = None
    last_traceback = None

    try:
        attempts = retry + 1
        for attempt in range(attempts):
            try:
                # Clean up incomplete outputs before (re)running
                _clean_incomplete_outputs(task_dir)

                # Run the task
                runner(task_entry, out_dir, manifest_path, gpu)

                # Verify outputs
                validate_task_outputs(task_dir)

                duration = time.time() - start_time
                return TaskProgress(
                    idx=idx,
                    task_uid=task_uid,
                    status="success",
                    duration_s=duration,
                )

            except Exception as e:
                last_error = str(e)
                last_traceback = traceback.format_exc()
                log.warning(f"Task {task_uid} attempt {attempt + 1}/{attempts} failed: {e}")
                if attempt < attempts - 1:
                    time.sleep(1.0)  # Brief pause before retry

        # All retries exhausted
        duration = time.time() - start_time
        return TaskProgress(
            idx=idx,
            task_uid=task_uid,
            status="failed",
            duration_s=duration,
            error=last_error,
            traceback=last_traceback,
        )

    finally:
        release_lock(task_dir)


def _clean_incomplete_outputs(task_dir: Path) -> None:
    """Remove incomplete outputs from a task directory.

    This ensures a clean slate before (re)running a task.
    Uses atomic operations where possible.

    Args:
        task_dir: Path to the task directory
    """
    if not task_dir.exists():
        return

    # Remove artifact files (but not the lock file)
    for artifact in REQUIRED_ARTIFACTS:
        artifact_path = task_dir / artifact
        if artifact_path.exists():
            artifact_path.unlink()

    # Remove checkpoint directory
    ckpt_dir = task_dir / "final_ckpt"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


def _worker_init():
    """Initialize worker process."""
    pass


def _parse_gpu_list(gpu_str: Optional[str]) -> Optional[List[str]]:
    """Parse GPU string into a list of GPU IDs.

    Args:
        gpu_str: Comma-separated GPU IDs (e.g., "0,1,2,3") or None

    Returns:
        List of GPU ID strings, or None if no GPU specified
    """
    if gpu_str is None:
        return None
    return [g.strip() for g in gpu_str.split(",") if g.strip()]


def _worker_task(args: Tuple) -> TaskProgress:
    """Worker function for parallel execution.

    Args:
        args: Tuple of (idx, task_entry_dict, out_dir, manifest_path, dry_run, retry, gpu)

    Returns:
        TaskProgress record
    """
    idx, task_entry_dict, out_dir, manifest_path, dry_run, retry, gpu = args

    # Set GPU for this task before importing JAX (via TaskRunner)
    if gpu is not None:
        if gpu == "-1":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    task_entry = TaskEntry.from_json(task_entry_dict)
    return run_one_task(
        idx=idx,
        task_entry=task_entry,
        out_dir=out_dir,
        manifest_path=manifest_path,
        dry_run=dry_run,
        retry=retry,
        gpu=gpu,
    )


def collect_histories(
    config: CollectorConfig,
    runner: Optional[RunnerCallable] = None,
) -> CollectorStats:
    """Run history collection for all tasks in a manifest.

    Args:
        config: Collector configuration
        runner: Optional custom runner callable (for testing)

    Returns:
        Collection statistics
    """
    # Load manifest
    entries = load_manifest(config.manifest_path)
    total_tasks = len(entries)

    if total_tasks == 0:
        log.warning("Manifest is empty, nothing to do")
        return CollectorStats()

    # Determine which tasks to run
    if config.task_indices:
        indices_to_run = parse_task_indices(config.task_indices, total_tasks)
    else:
        indices_to_run = list(range(total_tasks))

    # Apply max_tasks limit
    if config.max_tasks is not None:
        indices_to_run = indices_to_run[:config.max_tasks]

    tasks_to_run = len(indices_to_run)
    log.info(f"Manifest has {total_tasks} tasks, will process {tasks_to_run}")

    # Setup output directory and logging
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "_collect_runs" / timestamp
    progress_logger = ProgressLogger(log_dir)

    run_config = RunConfig(
        manifest_path=str(config.manifest_path),
        out_dir=str(config.out_dir),
        num_workers=config.num_workers,
        max_tasks=config.max_tasks,
        task_indices=config.task_indices,
        fail_fast=config.fail_fast,
        retry=config.retry,
        dry_run=config.dry_run,
        started_at=timestamp,
        total_tasks=total_tasks,
        tasks_to_run=tasks_to_run,
    )
    progress_logger.write_config(run_config)

    # Track statistics
    stats = CollectorStats(
        total_tasks=total_tasks,
        tasks_to_run=tasks_to_run,
    )

    # Process tasks
    if config.num_workers > 1 and not config.dry_run and runner is None:
        # Parallel execution (only with default runner)
        stats = _run_parallel(
            config=config,
            entries=entries,
            indices_to_run=indices_to_run,
            progress_logger=progress_logger,
        )
    else:
        # Sequential execution
        stats = _run_sequential(
            config=config,
            entries=entries,
            indices_to_run=indices_to_run,
            progress_logger=progress_logger,
            runner=runner,
        )

    # Log summary
    log.info(
        f"Collection complete: "
        f"{stats.success} success, "
        f"{stats.skipped} skipped, "
        f"{stats.failed} failed, "
        f"{stats.locked} locked"
    )
    log.info(f"Logs written to: {log_dir}")

    return stats


def _run_sequential(
    config: CollectorConfig,
    entries: List[TaskEntry],
    indices_to_run: List[int],
    progress_logger: ProgressLogger,
    runner: Optional[RunnerCallable] = None,
) -> CollectorStats:
    """Run tasks sequentially.

    Args:
        config: Collector configuration
        entries: All manifest entries
        indices_to_run: Indices of tasks to run
        progress_logger: Progress logger
        runner: Optional custom runner callable

    Returns:
        Collection statistics
    """
    stats = CollectorStats(
        total_tasks=len(entries),
        tasks_to_run=len(indices_to_run),
    )

    for i, idx in enumerate(indices_to_run):
        entry = entries[idx]
        log.info(f"[{i + 1}/{len(indices_to_run)}] Processing task {idx}: {entry.task_id}")

        progress = run_one_task(
            idx=idx,
            task_entry=entry,
            out_dir=config.out_dir,
            manifest_path=config.manifest_path,
            dry_run=config.dry_run,
            retry=config.retry,
            runner=runner,
            gpu=config.gpu,
        )

        progress_logger.log_progress(progress)

        # Update stats
        if progress.status == "skipped":
            stats.skipped += 1
            log.info(f"  -> Skipped (already complete)")
        elif progress.status == "success":
            stats.success += 1
            log.info(f"  -> Success ({progress.duration_s:.1f}s)")
        elif progress.status == "failed":
            stats.failed += 1
            log.error(f"  -> Failed: {progress.error}")
            if config.fail_fast:
                log.error("Stopping due to --fail_fast")
                break
        elif progress.status == "locked":
            stats.locked += 1
            log.warning(f"  -> Locked by another process")
        elif progress.status == "dry_run":
            log.info(f"  -> Would run (dry run)")

    return stats


def _run_parallel(
    config: CollectorConfig,
    entries: List[TaskEntry],
    indices_to_run: List[int],
    progress_logger: ProgressLogger,
) -> CollectorStats:
    """Run tasks in parallel using multiprocessing.

    Args:
        config: Collector configuration
        entries: All manifest entries
        indices_to_run: Indices of tasks to run
        progress_logger: Progress logger

    Returns:
        Collection statistics
    """
    stats = CollectorStats(
        total_tasks=len(entries),
        tasks_to_run=len(indices_to_run),
    )

    # Parse GPU list for round-robin assignment
    gpu_list = _parse_gpu_list(config.gpu)

    # Prepare task arguments with GPU assignment
    task_args = []
    for i, idx in enumerate(indices_to_run):
        # Assign GPU in round-robin fashion
        if gpu_list is not None:
            assigned_gpu = gpu_list[i % len(gpu_list)]
        else:
            assigned_gpu = None

        task_args.append((
            idx, entries[idx].to_json(), config.out_dir, config.manifest_path,
            config.dry_run, config.retry, assigned_gpu
        ))

    log.info(f"Starting {config.num_workers} workers for {len(task_args)} tasks")
    if gpu_list is not None:
        log.info(f"Distributing tasks across GPUs: {gpu_list} (round-robin)")

    with Pool(processes=config.num_workers, initializer=_worker_init) as pool:
        for progress in pool.imap_unordered(_worker_task, task_args):
            progress_logger.log_progress(progress)

            # Update stats
            if progress.status == "skipped":
                stats.skipped += 1
            elif progress.status == "success":
                stats.success += 1
                log.info(f"Task {progress.task_uid}: success ({progress.duration_s:.1f}s)")
            elif progress.status == "failed":
                stats.failed += 1
                log.error(f"Task {progress.task_uid}: failed - {progress.error}")
                if config.fail_fast:
                    log.error("Stopping due to --fail_fast")
                    pool.terminate()
                    break
            elif progress.status == "locked":
                stats.locked += 1

            # Progress update
            completed = stats.skipped + stats.success + stats.failed + stats.locked
            log.info(f"Progress: {completed}/{stats.tasks_to_run}")

    return stats


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> CollectorConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect training histories for all tasks in a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to JSONL manifest file"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for task results"
    )

    # Task selection
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Maximum number of tasks to process"
    )
    parser.add_argument(
        "--task_indices", type=str, default=None,
        help="Specific task indices to process (e.g., '0,3,7-12')"
    )

    # Execution options
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--fail_fast", action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--retry", type=int, default=0,
        help="Number of retry attempts on failure (default: 0)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be done without actually running"
    )

    # GPU selection
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU device ID(s) to use (e.g., '0', '0,1'). Use '-1' for CPU. If not specified, uses all available GPUs."
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", type=bool, default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    return CollectorConfig(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        max_tasks=args.max_tasks,
        task_indices=args.task_indices,
        fail_fast=args.fail_fast,
        retry=args.retry,
        dry_run=args.dry_run,
        gpu=args.gpu,
    )


def main() -> int:
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = parse_args()

    # Log GPU configuration
    if config.gpu is not None:
        if config.gpu == "-1":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            log.info("Using CPU")
        else:
            gpu_list = _parse_gpu_list(config.gpu)
            if config.num_workers > 1 and gpu_list and len(gpu_list) > 1:
                log.info(f"Multi-GPU mode: {len(gpu_list)} GPUs ({config.gpu}) for {config.num_workers} workers")
                log.info("Tasks will be distributed across GPUs in round-robin fashion")
            else:
                # For sequential or single GPU, set CUDA_VISIBLE_DEVICES globally
                os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
                log.info(f"Using GPU: {config.gpu}")
    else:
        log.info("Using all available GPUs")

    if config.dry_run:
        log.info("DRY RUN MODE - no tasks will actually be executed")

    try:
        stats = collect_histories(config)

        if stats.failed > 0 and not config.dry_run:
            return 1
        return 0

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
