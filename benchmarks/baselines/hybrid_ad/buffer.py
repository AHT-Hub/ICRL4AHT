"""Hybrid-AD buffer: reuses AD's step-level buffer directly.

Hybrid-AD uses the same step-level context update semantics as AD,
so no buffer modification is needed.
"""

from benchmarks.baselines.ad.buffer import OnlineBuffer, VectorizedOnlineBuffer

__all__ = [
    "OnlineBuffer",
    "VectorizedOnlineBuffer",
]
