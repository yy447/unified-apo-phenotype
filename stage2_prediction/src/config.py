from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class Stage2Config:
    data_path: str
    results_dir: str

    time_columns: List[str]
    event_columns: List[str]
    general_covariates: List[str]

    apo_binary_candidates: List[str]
    apo_fallback_col: str
    penalizer_values: List[float]

    cp_variants: List[str]


def load_config(path: str) -> Stage2Config:
    """Load YAML config into a Stage2Config."""
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)

    return Stage2Config(
        data_path=d["data_path"],
        results_dir=d["results_dir"],
        time_columns=d["time_columns"],
        event_columns=d["event_columns"],
        general_covariates=d["general_covariates"],
        apo_binary_candidates=d.get("apo_binary_candidates", []),
        apo_fallback_col=d.get("apo_fallback_col", "APO_unified_overall_max"),
        penalizer_values=d.get("penalizer_values", [0.01, 0.1, 1.0]),
        cp_variants=d.get(
            "cp_variants", ["CP", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"]
        ),
    )
