"""Runners module for task-based training and evaluation.

This module provides utilities for running PPO training against fixed teammates
and recording learning histories for later ICRL baseline training
(AD, DPT, AMAGO-offline, Hybrid-AD).
"""

from runners.task_runner import TaskRunner, run_task
from runners.history_recorder import HistoryRecorder

__all__ = ["TaskRunner", "run_task", "HistoryRecorder"]
