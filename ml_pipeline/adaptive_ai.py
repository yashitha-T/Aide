"""Adaptive AI utilities: blend learning, calibration, and dynamic thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass
class AdaptiveAIConfig:
    model_path: str = "models/adaptive_ai.joblib"


class AdaptiveAIScorer:
    """
    Lightweight adaptive scorer with:
    - learned blending (LinearRegression)
    - Platt scaling calibration per symbol
    - dynamic threshold selection per symbol
    """

    def __init__(self, config: Optional[AdaptiveAIConfig] = None):
        self.config = config or AdaptiveAIConfig()
        self.blend_model: Optional[LinearRegression] = None
        self.calibration_params: Dict[str, Dict[str, float]] = {}
        self.dynamic_thresholds: Dict[str, float] = {}
        self.is_fitted = False

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    @staticmethod
    def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["rule_score", "sequence_score", "anomaly_score", "phase_confidence"]
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[cols].astype(float).to_numpy()

    def fit(self, symbol: str, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train adaptive blend + calibration + dynamic threshold.

        Expected columns:
        - rule_score [0-100]
        - sequence_score [0-100]
        - anomaly_score [0-100 or 0-1]
        - phase_confidence [0-1]
        - trap_label [0/1]
        """
        if train_df is None or train_df.empty or "trap_label" not in train_df.columns:
            return {"fitted": 0}

        df = train_df.copy().reset_index(drop=True)

        y = df["trap_label"].astype(int).to_numpy()
        X = self._build_X(df)

        # Normalize anomaly input to 0-100 if needed.
        anomaly_col = X[:, 2]
        if np.nanmax(np.abs(anomaly_col)) <= 1.0:
            X[:, 2] = anomaly_col * 100.0

        # Learned blending model.
        blend_model = LinearRegression()
        blend_model.fit(X, y * 100.0)
        self.blend_model = blend_model

        raw_pred = np.clip(blend_model.predict(X), 0.0, 100.0)

        # Platt scaling (logistic calibration): P(y=1 | raw_pred).
        if len(np.unique(y)) >= 2:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(raw_pred.reshape(-1, 1), y)
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])
        else:
            # Fallback: weak identity-like calibration.
            a, b = 0.08, -4.0

        self.calibration_params[symbol] = {"a": a, "b": b}

        calibrated = self._sigmoid(a * raw_pred + b)
        calibrated_pct = np.clip(calibrated * 100.0, 0.0, 100.0)

        # Auto threshold per asset: maximize F1 on [40..90]
        best_thr = 70.0
        best_f1 = -1.0
        for thr in range(40, 91):
            y_hat = (calibrated_pct >= float(thr)).astype(int)
            f1 = self._f1(y, y_hat)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        self.dynamic_thresholds[symbol] = best_thr
        self.is_fitted = True
        self.save(self.config.model_path)

        return {
            "fitted": 1,
            "best_threshold": best_thr,
            "best_f1": float(best_f1),
        }

    def predict_raw(
        self,
        rule_score: float,
        sequence_score: float,
        anomaly_score: float,
        phase_confidence: float,
        fallback_rule_weight: float = 0.6,
        fallback_seq_weight: float = 0.4,
    ) -> float:
        x = np.array([[rule_score, sequence_score, anomaly_score, phase_confidence]], dtype=float)
        if x[0, 2] <= 1.0:
            x[0, 2] = x[0, 2] * 100.0

        if self.blend_model is not None:
            pred = float(self.blend_model.predict(x)[0])
            return float(np.clip(pred, 0.0, 100.0))

        fallback = (fallback_rule_weight * float(rule_score)) + (fallback_seq_weight * float(sequence_score))
        return float(np.clip(fallback, 0.0, 100.0))

    def calibrate(self, symbol: str, raw_risk: float) -> float:
        """Return calibrated probability in [0, 1]."""
        params = self.calibration_params.get(symbol)
        if not params:
            # Probability-like fallback.
            return float(np.clip(raw_risk / 100.0, 0.0, 1.0))

        a = float(params.get("a", 0.08))
        b = float(params.get("b", -4.0))
        p = float(self._sigmoid(a * float(raw_risk) + b))
        return float(np.clip(p, 0.0, 1.0))

    def get_threshold(self, symbol: str, default: float = 70.0) -> float:
        return float(self.dynamic_thresholds.get(symbol, default))

    def save(self, path: Optional[str] = None) -> None:
        save_path = path or self.config.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "blend_model": self.blend_model,
            "calibration_params": self.calibration_params,
            "dynamic_thresholds": self.dynamic_thresholds,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, save_path)

    @classmethod
    def load(cls, path: str) -> "AdaptiveAIScorer":
        state = joblib.load(path)
        obj = cls(AdaptiveAIConfig(model_path=path))
        obj.blend_model = state.get("blend_model")
        obj.calibration_params = state.get("calibration_params", {})
        obj.dynamic_thresholds = state.get("dynamic_thresholds", {})
        obj.is_fitted = bool(state.get("is_fitted", False))
        return obj
