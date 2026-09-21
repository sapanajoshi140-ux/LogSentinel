# load config/*.yaml into dataclasses
"""
Central config loading. Every pipeline script takes --config path/to/x.yaml
instead of a pile of CLI flags, so a run is fully reproducible from one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DrainConfig:
    depth: int = 4
    sim_th: float = 0.4
    max_clusters: int | None = None


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15


@dataclass
class DetectorConfig:
    ewma_alpha: float = 0.3
    ewma_threshold: float = 3.0
    markov_order: int = 1
    bilstm_hidden_dim: int = 64
    bilstm_window: int = 10
    bilstm_epochs: int = 10


@dataclass
class DAWFConfig:
    beta: float = 0.5          # Weighted Majority penalty factor
    min_weight: float = 0.01   # floor so no detector's weight hits zero


@dataclass
class PathsConfig:
    raw_log: str = "data/raw/HDFS.log"
    labels: str | None = "data/raw/anomaly_label.csv"  # None for BGL (labels are in the log lines)
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"


@dataclass
class ProjectConfig:
    dataset: str = "hdfs"
    drain: DrainConfig = field(default_factory=DrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
    dawf: DAWFConfig = field(default_factory=DAWFConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(path: str | Path) -> ProjectConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return ProjectConfig(
        dataset=raw.get("dataset", "hdfs"),
        drain=DrainConfig(**raw.get("drain", {})),
        split=SplitConfig(**raw.get("split", {})),
        detectors=DetectorConfig(**raw.get("detectors", {})),
        dawf=DAWFConfig(**raw.get("dawf", {})),
        paths=PathsConfig(**raw.get("paths", {})),
    )