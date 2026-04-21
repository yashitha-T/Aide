"""Lightweight multivariate sequence model for trap probability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class SequenceModelConfig:
    window_size: int = 30
    conv_filters: int = 6
    kernel_size: int = 3
    random_seed: int = 42
    epochs: int = 50
    lr: float = 0.05


class SequenceTrapModel:
    """
    Lightweight NumPy sequence model.

    Design:
    - per-feature temporal convolutions + max pooling
    - logistic readout
    - stored feature ordering + normalization stats for stable inference
    """

    def __init__(self, config: Optional[SequenceModelConfig] = None):
        self.config = config or SequenceModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        self.feature_order: List[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        self.kernels = self.rng.normal(
            0.0,
            0.15,
            size=(self.config.conv_filters, self.config.kernel_size),
        )
        self.kernel_bias = np.zeros(self.config.conv_filters)

        self.readout: Optional[np.ndarray] = None
        self.readout_bias = 0.0
        self.is_trained = False

    def set_feature_order(self, feature_names: List[str]):
        self.feature_order = list(feature_names)

    def _align_window(
        self,
        window_features: np.ndarray | pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        if isinstance(window_features, pd.DataFrame):
            incoming_names = list(window_features.columns)
            arr = window_features.to_numpy(dtype=float)
        else:
            arr = np.asarray(window_features, dtype=float)
            incoming_names = feature_names or []

        if arr.ndim != 2:
            raise ValueError("window_features must be shape [window, features]")

        if not self.feature_order:
            if incoming_names:
                self.feature_order = list(incoming_names)
            else:
                self.feature_order = [f"f{i}" for i in range(arr.shape[1])]

        if incoming_names:
            name_to_idx = {n: i for i, n in enumerate(incoming_names)}
            aligned = np.zeros((arr.shape[0], len(self.feature_order)), dtype=float)
            for j, name in enumerate(self.feature_order):
                if name in name_to_idx:
                    aligned[:, j] = arr[:, name_to_idx[name]]
            arr = aligned
        else:
            if arr.shape[1] < len(self.feature_order):
                pad = np.zeros((arr.shape[0], len(self.feature_order) - arr.shape[1]))
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > len(self.feature_order):
                arr = arr[:, : len(self.feature_order)]

        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_window(self, arr: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            mu = np.mean(arr, axis=0)
            sd = np.std(arr, axis=0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            return (arr - mu) / sd

        sd = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        return (arr - self.feature_mean) / sd

    def _conv_pool_1d(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.config.kernel_size:
            signal = np.pad(signal, (self.config.kernel_size - len(signal), 0), mode="edge")

        acts = []
        for kernel, bias in zip(self.kernels, self.kernel_bias):
            conv = np.convolve(signal, kernel[::-1], mode="valid") + bias
            relu = np.maximum(conv, 0.0)
            acts.append(float(np.max(relu)) if len(relu) else 0.0)
        return np.array(acts, dtype=float)

    def _embed_window(self, arr_norm: np.ndarray) -> np.ndarray:
        # Convolve each feature independently and concatenate pooled responses.
        feats = []
        for f in range(arr_norm.shape[1]):
            feats.append(self._conv_pool_1d(arr_norm[:, f]))

        conv_emb = np.concatenate(feats, axis=0) if feats else np.zeros(self.config.conv_filters)

        # Add lightweight summary stats (last-step context).
        last_vals = arr_norm[-1, :] if arr_norm.shape[0] > 0 else np.zeros(arr_norm.shape[1])
        mean_vals = np.mean(arr_norm, axis=0) if arr_norm.shape[0] > 0 else np.zeros(arr_norm.shape[1])

        return np.concatenate([conv_emb, last_vals, mean_vals], axis=0)

    def _predict_logit_from_emb(self, emb: np.ndarray) -> float:
        if self.readout is None:
            # Neutral fallback until trained.
            return 0.0
        return float(np.dot(emb, self.readout) + self.readout_bias)

    def fit_windows(
        self,
        X_windows: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        if X_windows.ndim != 3:
            raise ValueError("X_windows must be shape [batch, window, features]")

        if feature_names:
            self.set_feature_order(feature_names)
        elif not self.feature_order:
            self.feature_order = [f"f{i}" for i in range(X_windows.shape[2])]

        # Fit normalization stats globally.
        stacked = X_windows.reshape(-1, X_windows.shape[2]).astype(float)
        self.feature_mean = np.mean(stacked, axis=0)
        self.feature_std = np.std(stacked, axis=0)
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)

        embeddings = []
        for i in range(X_windows.shape[0]):
            aligned = self._align_window(X_windows[i], feature_names=self.feature_order)
            emb = self._embed_window(self._normalize_window(aligned))
            embeddings.append(emb)
        E = np.array(embeddings)
        y = y.astype(float)

        # Initialize readout once embedding size is known.
        if self.readout is None or len(self.readout) != E.shape[1]:
            self.readout = self.rng.normal(0.0, 0.05, size=E.shape[1])
            self.readout_bias = 0.0

        for _ in range(self.config.epochs):
            logits = E @ self.readout + self.readout_bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            err = probs - y

            grad_w = (E.T @ err) / len(E)
            grad_b = float(np.mean(err))

            self.readout -= self.config.lr * grad_w
            self.readout_bias -= self.config.lr * grad_b

        self.is_trained = True
        return self

    def fit_from_frame(
        self,
        df: pd.DataFrame,
        label_col: str = "trap_label",
        feature_columns: Optional[List[str]] = None,
        window_size: Optional[int] = None,
    ):
        if df is None or df.empty or label_col not in df.columns:
            return self

        win = int(window_size or self.config.window_size)
        feature_columns = feature_columns or [c for c in df.columns if c != label_col]

        frame = df[feature_columns + [label_col]].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X, y = [], []
        for i in range(win - 1, len(frame)):
            X.append(frame[feature_columns].iloc[i - win + 1 : i + 1].to_numpy(dtype=float))
            y.append(float(frame[label_col].iloc[i]))

        if not X:
            return self

        return self.fit_windows(np.array(X), np.array(y), feature_names=feature_columns)

    def predict(
        self,
        window_features: np.ndarray | pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> float:
        aligned = self._align_window(window_features, feature_names=feature_names)
        emb = self._embed_window(self._normalize_window(aligned))
        logit = self._predict_logit_from_emb(emb)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return float(np.clip(prob, 0.0, 1.0))

    def feature_perturbation_importance(
        self,
        window_features: np.ndarray | pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        epsilon_scale: float = 0.05,
        max_features: int = 25,
    ) -> Dict[str, float]:
        aligned = self._align_window(window_features, feature_names=feature_names)
        names = self.feature_order if self.feature_order else [f"f{i}" for i in range(aligned.shape[1])]

        baseline = self.predict(aligned, feature_names=names)
        impacts: Dict[str, float] = {}

        # Limit perturbation count for latency safety.
        n_feats = min(aligned.shape[1], max_features)
        for j in range(n_feats):
            feat_std = float(np.std(aligned[:, j]))
            eps = epsilon_scale * (feat_std if feat_std > 1e-6 else 1.0)

            perturbed = aligned.copy()
            perturbed[:, j] += eps
            new_prob = self.predict(perturbed, feature_names=names)
            impacts[names[j]] = float(abs(new_prob - baseline))

        return impacts

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "feature_order": self.feature_order,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "kernels": self.kernels,
            "kernel_bias": self.kernel_bias,
            "readout": self.readout,
            "readout_bias": self.readout_bias,
            "is_trained": self.is_trained,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "SequenceTrapModel":
        state = joblib.load(path)
        model = cls(config=state.get("config", SequenceModelConfig()))
        model.feature_order = state.get("feature_order", [])
        model.feature_mean = state.get("feature_mean")
        model.feature_std = state.get("feature_std")
        model.kernels = state.get("kernels", model.kernels)
        model.kernel_bias = state.get("kernel_bias", model.kernel_bias)
        model.readout = state.get("readout")
        model.readout_bias = state.get("readout_bias", 0.0)
        model.is_trained = bool(state.get("is_trained", False))
        return model
