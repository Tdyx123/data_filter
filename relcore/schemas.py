"""Shared in-memory contracts for relcore stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ClipRecord:
    sample_id: str
    episode_id: int
    task_index: int
    task_name: str
    start_step: int
    end_step: int
    length: int
    previous_sample_id: str | None
    next_sample_id: str | None


@dataclass
class EdgeTable:
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    edge_type: str


@dataclass
class GraphData:
    sample_ids: list[str]
    task_indices: np.ndarray
    embeddings: np.ndarray
    reliability: np.ndarray
    prototype_indices: np.ndarray
    prototype_weights: np.ndarray
    sequence_edges: EdgeTable
    similarity_edges: EdgeTable
    transition_matrix: sparse.csr_matrix
    cooccurrence_matrix: sparse.csr_matrix


@dataclass
class ObjectiveState:
    selected_mask: np.ndarray
    task_counts: np.ndarray
    prototype_coverage: np.ndarray
    sequence_relation_counts: np.ndarray
    redundancy_sum: float
    objective_value: float
