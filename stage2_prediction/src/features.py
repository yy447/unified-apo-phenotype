from __future__ import annotations

import re
from typing import List, Sequence

import pandas as pd


def _norm(s: str) -> str:
    """Normalize a string by keeping only [a-z0-9]."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def get_cp_covariates(df: pd.DataFrame, cp_prefix: str) -> List[str]:
    """
    Resolve standard CP component columns for a given prefix (CP/CP1/CP2/CP5).
    Tolerant to separators/spaces. Falls back to prefix match.
    """
    expected = [
        "Gestational Diabetes",
        "Gestational Hypertension",
        "Preeclampsia",
        "Preterm Delivery",
        "Small for Gestational Age",
    ]

    normmap = {_norm(c): c for c in df.columns}
    found: List[str] = []

    for name in expected:
        want1 = f"{cp_prefix}_{name}"
        want2 = f"{cp_prefix} {name}"
        if _norm(want1) in normmap:
            found.append(normmap[_norm(want1)])
        elif _norm(want2) in normmap:
            found.append(normmap[_norm(want2)])

    if len(found) == 5:
        print(f"[INFO] {cp_prefix}: matched 5 standard CP columns: {found}")
        return found

    prefix_variants = (f"{cp_prefix}_", f"{cp_prefix} ")
    fallback = [c for c in df.columns if c.startswith(prefix_variants)]
    if fallback:
        print(
            f"[INFO] {cp_prefix}: matched {len(found)} standard columns; "
            f"fallback to {len(fallback)} prefix columns. Example: {fallback[:5]}"
        )
        return fallback

    print(f"[WARN] {cp_prefix}: no CP columns found for prefix '{cp_prefix}'.")
    return []


def check_model_outputs(
    df: pd.DataFrame, model_prefix: str, outcomes: Sequence[str] = ("e_Any",)
) -> None:
    """Validate that prob/risk columns exist for a fitted model."""

    def looks_like_prob(col: str) -> bool:
        n = _norm(col)
        return ("prob" in n) or ("risk" in n)

    cols = [c for c in df.columns if (model_prefix in c and looks_like_prob(c))]
    if not cols:
        candidates = [c for c in df.columns if model_prefix in c]
        sample = "\n  ".join(candidates[:20])
        raise RuntimeError(
            f"[CHECK] No prob/risk columns found for '{model_prefix}'.\n"
            f"Columns under this prefix (sample <=20):\n  {sample if sample else '(none)'}"
        )

    for ek in outcomes:
        ok = any(
            (model_prefix in c) and (_norm(ek) in _norm(c)) and looks_like_prob(c)
            for c in df.columns
        )
        if not ok:
            print(
                f"[WARN] {model_prefix}: no prob/risk column detected for outcome '{ek}'."
            )

    print(f"[OK] {model_prefix}: prob/risk columns = {len(cols)}; example: {cols[:3]}")
