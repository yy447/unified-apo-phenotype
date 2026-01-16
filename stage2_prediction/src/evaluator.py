# stage2_prediction/src/evaluator.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a series to numeric; invalid parsing becomes NaN."""
    return pd.to_numeric(s, errors="coerce")


@dataclass
class CoxFitResult:
    """Container for model performance metrics for one outcome."""

    validation_c_index: float
    final_c_index: float
    isotonic_auc: float
    best_penalizer: float


class CoxRiskModelEvaluator:
    """
    Train Cox PH models with penalizer tuning (GroupKFold), calibrate risk scores
    via isotonic regression, and support reclassification & subgroup analyses.

    This class writes prediction columns back to `self.df`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_columns: List[str],
        event_columns: List[str],
        general_covariates: List[str],
        apo_covariates: Optional[List[str]],
        penalizer_values: List[float],
        prediction_horizon: int = 365 * 5,
    ):
        self.df = df
        self.time_columns = time_columns
        self.event_columns = event_columns
        self.general_covariates = general_covariates
        self.apo_covariates = apo_covariates or []
        self.penalizer_values = penalizer_values
        self.prediction_horizon = prediction_horizon

        self.predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.cox_models: Dict[Tuple[str, str, str], CoxPHFitter] = {}

        # Ensure APO_ind exists
        self._ensure_apo_indicator()

    # ---------------------------------------------------------------------
    # APO indicator helpers
    # ---------------------------------------------------------------------
    def _ensure_apo_indicator(self) -> None:
        """
        Create:
        - apo_count: sum of APO component indicators (best-effort)
        - APO_ind: 1 if any APO component > 0 else 0
        """
        cols = [c for c in (self.apo_covariates or []) if c in self.df.columns]
        if cols:
            self.df["apo_count"] = self.df[cols].fillna(0).sum(axis=1)
        else:
            if "Any APO_True" in self.df.columns:
                any_true = (
                    pd.to_numeric(self.df["Any APO_True"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                self.df["apo_count"] = any_true
            else:
                apo_like = [c for c in self.df.columns if c.endswith("_1")]
                if apo_like:
                    self.df["apo_count"] = (
                        self.df[apo_like]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0)
                        .sum(axis=1)
                    )
                else:
                    self.df["apo_count"] = 0

        self.df["APO_ind"] = (self.df["apo_count"] > 0).astype(int)

    def _oof_iso_apo0(
        self,
        prob_col: str,
        y_col: str,
        n_splits: int = 5,
        seed: int = 42,
        min_n: int = 50,
    ) -> None:
        """
        Create `{prob_col}_calOOF_APO0` for all rows:
        - If APO_ind == 0: replace with out-of-fold isotonic-calibrated probabilities
        - If APO_ind  > 0: keep the original probability
        If APO=0 sample size < min_n, fallback to original probability for all.
        """
        if prob_col not in self.df.columns or y_col not in self.df.columns:
            return

        out_col = f"{prob_col}_calOOF_APO0"
        self.df[out_col] = self.df[prob_col].astype(float)

        mask = (
            (self.df["APO_ind"] == 0)
            & self.df[prob_col].notna()
            & self.df[y_col].notna()
        )
        idx = self.df.index[mask]
        n = len(idx)
        if n < min_n:
            return

        x = self.df.loc[idx, prob_col].astype(float).values
        y = self.df.loc[idx, y_col].astype(float).values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = np.empty_like(x, dtype=float)

        for tr, va in kf.split(x):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x[tr], y[tr])
            pred[va] = np.clip(ir.predict(x[va]), 1e-8, 1 - 1e-8)

        self.df.loc[idx, out_col] = pred

    # ---------------------------------------------------------------------
    # Training + evaluation
    # ---------------------------------------------------------------------
    def fit_and_evaluate(
        self,
        model_name: str,
        additional_covariates: Optional[List[str]] = None,
        include_apo: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit one model for each (time_col, event_col), tune penalizer by 5-fold
        GroupKFold on row index, then fit final model on full subset.

        Writes columns:
        - {model}_{event}_raw_prob: partial hazard (risk score)
        - {model}_{event}_prob: isotonic-calibrated probability
        - {model}_{event}_prob_calOOF_APO0: optional OOF isotonic for APO=0
        - {model}_{event}_rank: percentile rank of prob
        """
        additional_covariates = additional_covariates or []

        results: Dict[str, Dict[str, Any]] = {}

        for time_col, event_col in zip(self.time_columns, self.event_columns):
            print(f"\nProcessing: {model_name} - {time_col} / {event_col}")

            covariates = list(self.general_covariates)
            if include_apo:
                covariates += list(self.apo_covariates)
            covariates += list(additional_covariates)
            covariates = list(dict.fromkeys(covariates))  # de-duplicate

            required = [time_col, event_col] + covariates
            subset = self.df[required].dropna()
            if len(subset) < 10:
                print(f"Skipping {event_col} (n<10).")
                continue

            subset[event_col] = (
                subset[event_col]
                .astype(str)
                .replace({"True": 1, "False": 0, "1.0": 1, "0.0": 0})
                .astype(float)
                .astype(int)
            )

            # Use index as group id (keeps your original behavior)
            groups = subset.index.astype(str)
            gkf = GroupKFold(n_splits=5)

            best_pen, best_c = None, 0.0
            for pen in self.penalizer_values:
                scores = []
                for tr, va in gkf.split(subset, groups=groups):
                    trd = subset.iloc[tr].copy()
                    vad = subset.iloc[va].copy()
                    m = CoxPHFitter(penalizer=pen)
                    m.fit(trd, duration_col=time_col, event_col=event_col)
                    c = concordance_index(
                        vad[time_col],
                        -m.predict_partial_hazard(vad),
                        vad[event_col],
                    )
                    scores.append(c)
                mean_c = float(np.mean(scores))
                if mean_c > best_c:
                    best_c, best_pen = mean_c, pen

            final = CoxPHFitter(penalizer=float(best_pen))
            final.fit(subset, duration_col=time_col, event_col=event_col)
            key = (model_name, time_col, event_col)
            self.cox_models[key] = final

            raw_scores = final.predict_partial_hazard(subset)

            # Global isotonic calibration on the same subset
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(
                raw_scores.values.astype(float), subset[event_col].values.astype(float)
            )
            prob = ir.predict(raw_scores.values.astype(float))
            auc = roc_auc_score(subset[event_col], prob)

            results[event_col] = {
                "Validation C-index (5-Fold)": float(best_c),
                "Final C-index": float(
                    concordance_index(
                        subset[time_col],
                        -final.predict_partial_hazard(subset),
                        subset[event_col],
                    )
                ),
                "Isotonic AUC": float(auc),
                "Best Penalizer": float(best_pen),
            }

            raw_prob_col = f"{model_name}_{event_col}_raw_prob"
            prob_col = f"{model_name}_{event_col}_prob"

            self.df[raw_prob_col] = np.nan
            self.df[prob_col] = np.nan
            self.df.loc[subset.index, raw_prob_col] = raw_scores
            self.df.loc[subset.index, prob_col] = prob

            # OOF isotonic on APO=0 subset, if applicable
            self._oof_iso_apo0(prob_col=prob_col, y_col=event_col)

            rank_col = f"{model_name}_{event_col}_rank"
            self.df[rank_col] = self.df[prob_col].rank(pct=True)

        self.predictions[model_name] = results
        return results

    # ---------------------------------------------------------------------
    # Simple flags
    # ---------------------------------------------------------------------
    def _compute_naive_apo_count(self) -> None:
        """Compute apo_count and apo_group bins using self.apo_covariates."""
        if not self.apo_covariates:
            self.df["apo_count"] = 0
            self.df["apo_group"] = 0
            return

        self.df["apo_count"] = self.df[self.apo_covariates].fillna(0).sum(axis=1)
        self.df["apo_group"] = pd.cut(
            self.df["apo_count"],
            bins=[-1, 0, 1, 2, 3, 4, 5],
            labels=[0, 1, 2, 3, 4, 5],
        )

    def _compute_cvd_flag(self) -> None:
        """Flag whether any CVD event column is positive."""
        self.df["cvd_flag"] = (self.df[self.event_columns].sum(axis=1) > 0).astype(int)

    # ---------------------------------------------------------------------
    # HR table
    # ---------------------------------------------------------------------
    def get_hr_table(
        self, model_name: str, time_col: str, event_col: str
    ) -> pd.DataFrame:
        """
        Return HR table with 95% CI for a fitted Cox model.
        """
        key = (model_name, time_col, event_col)
        if key not in self.cox_models:
            raise KeyError(f"No Cox model found for {key}. Run fit_and_evaluate first.")

        m = self.cox_models[key]
        coefs = m.params_
        ci = m.confidence_intervals_

        hr = np.exp(coefs)
        hr_ci_lower = np.exp(ci.iloc[:, 0])
        hr_ci_upper = np.exp(ci.iloc[:, 1])

        return pd.DataFrame(
            {
                "variable": coefs.index,
                "coef": coefs.values,
                "HR": hr.values,
                "HR_lower": hr_ci_lower.values,
                "HR_upper": hr_ci_upper.values,
            }
        )

    # ---------------------------------------------------------------------
    # Reclassification (NRI / net-up)
    # ---------------------------------------------------------------------
    def _compute_reclassification_full(
        self,
        df_subset: pd.DataFrame,
        base_model_col: str,
        new_model_col: str,
        event_col: str,
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Compute 2x2 reclassification tables for cases and non-cases,
        and compute NRI and net-up counts.
        """
        for c in [base_model_col, new_model_col, event_col]:
            if c not in df_subset.columns:
                empty = pd.DataFrame(
                    columns=["New Low", "New High"], index=["Base Low", "Base High"]
                ).fillna(0)
                return {
                    "cases_table": empty.copy(),
                    "noncases_table": empty.copy(),
                    "metrics": {
                        "cases_up_pct": 0.0,
                        "cases_down_pct": 0.0,
                        "noncases_down_pct": 0.0,
                        "noncases_up_pct": 0.0,
                        "NRI": 0.0,
                        "net_up": 0,
                    },
                }

        def _tab(df: pd.DataFrame) -> pd.DataFrame:
            base_high = df[base_model_col] > threshold
            new_high = df[new_model_col] > threshold
            t = pd.crosstab(base_high, new_high)
            t = t.reindex(index=[False, True], columns=[False, True], fill_value=0)
            t = t.rename(
                index={False: "Base Low", True: "Base High"},
                columns={False: "New Low", True: "New High"},
            )
            return t

        cases = df_subset[df_subset[event_col] == 1]
        noncs = df_subset[df_subset[event_col] == 0]

        cases_tab = _tab(cases) if len(cases) else _tab(df_subset.iloc[0:0])
        non_tab = _tab(noncs) if len(noncs) else _tab(df_subset.iloc[0:0])

        def _pct(a: float, b: float) -> float:
            return (a / b * 100.0) if b > 0 else 0.0

        cases_den = cases_tab.to_numpy().sum()
        cases_up_pct = _pct(cases_tab.loc["Base Low", "New High"], cases_den)
        cases_down_pct = _pct(cases_tab.loc["Base High", "New Low"], cases_den)

        non_den = non_tab.to_numpy().sum()
        non_down_pct = _pct(non_tab.loc["Base High", "New Low"], non_den)  # good
        non_up_pct = _pct(non_tab.loc["Base Low", "New High"], non_den)  # bad

        nri = (cases_up_pct - cases_down_pct) / 100.0 + (
            non_down_pct - non_up_pct
        ) / 100.0

        all_tab = _tab(df_subset)
        net_up = int(
            all_tab.loc["Base Low", "New High"] - all_tab.loc["Base High", "New Low"]
        )

        return {
            "cases_table": cases_tab,
            "noncases_table": non_tab,
            "metrics": {
                "cases_up_pct": round(float(cases_up_pct), 2),
                "cases_down_pct": round(float(cases_down_pct), 2),
                "noncases_down_pct": round(float(non_down_pct), 2),
                "noncases_up_pct": round(float(non_up_pct), 2),
                "NRI": round(float(nri), 3),
                "net_up": net_up,
            },
        }

    # ---------------------------------------------------------------------
    # Group-by summary + reclassification on "low group"
    # ---------------------------------------------------------------------
    def analyze_group_by(
        self,
        group_col: str,
        low_group_selector=None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
        """
        For each model/outcome:
        - Summarize avg prob/rank by (group_col, cvd_flag)
        - Compute reclassification vs RP1 within the "low group"
        Threshold is outcome prevalence in full cohort.
        """
        self._compute_naive_apo_count()
        self._compute_cvd_flag()

        if group_col not in self.df.columns:
            raise KeyError(
                f"[analyze_group_by] group_col '{group_col}' not found in df"
            )

        results: Dict[str, pd.DataFrame] = {}
        reclass_tables: Dict[str, Dict[str, Any]] = {}

        for model_name in self.predictions:
            for event_col in self.event_columns:
                raw_prob_col = f"{model_name}_{event_col}_raw_prob"
                prob_col = f"{model_name}_{event_col}_prob"
                rank_col = f"{model_name}_{event_col}_rank"
                if prob_col not in self.df.columns:
                    continue

                grouped = (
                    self.df.groupby([group_col, "cvd_flag"])
                    .agg(
                        avg_raw_prob=(raw_prob_col, "mean"),
                        avg_recalibrated_prob=(prob_col, "mean"),
                        avg_rank=(rank_col, "mean"),
                    )
                    .reset_index()
                )
                results[f"{model_name}_{event_col}"] = grouped

                if model_name != "RP1":
                    if low_group_selector is None:
                        low_mask = self._default_low_mask(self.df, group_col)
                    else:
                        low_mask = low_group_selector(self.df)
                        if (
                            not isinstance(low_mask, pd.Series)
                            or low_mask.dtype != bool
                        ):
                            raise ValueError(
                                "low_group_selector must return a boolean Series"
                            )

                    df_low = self.df[low_mask].copy()
                    thr = float(self.df[event_col].mean())

                    tbl = self._compute_reclassification_full(
                        df_subset=df_low,
                        base_model_col=f"RP1_{event_col}_prob",
                        new_model_col=prob_col,
                        event_col=event_col,
                        threshold=thr,
                    )
                    reclass_tables[f"{model_name}_{event_col}"] = tbl

        return results, reclass_tables

    def _default_low_mask(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        """Default low group: smallest value in group_col (supports numeric or sortable categories)."""
        s = df[group_col]
        if pd.api.types.is_numeric_dtype(s):
            return s == s.min()

        s_num = _coerce_numeric_series(s)
        if s_num.notna().any():
            return s_num == s_num.min()

        non_na = s.dropna()
        if non_na.empty:
            return pd.Series(False, index=df.index)

        low_val = sorted(non_na.unique())[0]
        return s == low_val

    # ---------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------
    def summarize_all_results(self) -> pd.DataFrame:
        """Summarize stored metrics across all fitted models/outcomes."""
        rows = []
        for model_name, model_results in self.predictions.items():
            for outcome, m in model_results.items():
                rows.append(
                    {
                        "Model": model_name,
                        "Outcome": outcome,
                        "Validation C-index": m.get(
                            "Validation C-index (5-Fold)", np.nan
                        ),
                        "Final C-index": m.get("Final C-index", np.nan),
                        "Isotonic AUC": m.get("Isotonic AUC", np.nan),
                        "Best Penalizer": m.get("Best Penalizer", np.nan),
                    }
                )
        return pd.DataFrame(rows)
