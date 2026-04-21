"""Backtest and classification metrics engine for MarketTrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class BacktestEngine:
    threshold: float = 70.0

    def run_backtest(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate predictions against trap labels.

        Prediction rule:
        - pred = risk_score >= 70
        """
        if df is None or df.empty:
            return {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "false_positive_rate": 0.0,
                "hit_rate": 0.0,
                "total": 0,
            }

        if "risk_score" not in df.columns:
            raise ValueError("run_backtest requires 'risk_score' column")
        if "trap_label" not in df.columns:
            raise ValueError("run_backtest requires 'trap_label' column")

        frame = df.copy()
        pred = (frame["risk_score"].astype(float) >= float(self.threshold)).astype(int)
        truth = frame["trap_label"].astype(int)

        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())

        def _safe_div(num: float, den: float) -> float:
            return float(num / den) if den else 0.0

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1_score = _safe_div(2 * precision * recall, precision + recall)
        accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
        false_positive_rate = _safe_div(fp, fp + tn)
        hit_rate = _safe_div(tp, tp + fn)

        return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "false_positive_rate": false_positive_rate,
            "hit_rate": hit_rate,
            "total": int(tp + tn + fp + fn),
        }
