"""MarketTrap Unified Risk Engine — adaptive, calibrated, explainable AI."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestEngine
from feature_engineering.liquidity_features import compute_all_liquidity_features
from feature_engineering.retail_features import compute_all_retail_features
from ml_pipeline.adaptive_ai import AdaptiveAIConfig, AdaptiveAIScorer
from ml_pipeline.anomaly_model import IsolationForestModel
from ml_pipeline.sequence_model import SequenceModelConfig, SequenceTrapModel
from ml_pipeline.trap_labeling import generate_trap_labels
from risk_inference.phase_engine import TrapPhaseEngine
from risk_inference.realtime_trap_engine import (
    build_component_scores,
    buyer_seller_control,
    classify_trap_type,
    extract_trap_reasons,
)

logger = logging.getLogger(__name__)


class MarketTrapEngine:
    """Unified engine for real-time market trap detection with adaptive AI."""

    def __init__(
        self,
        model_path: str = "models/isolation_forest.pkl",
        sequence_model_path: str = "models/sequence_model.joblib",
        adaptive_ai_path: str = "models/adaptive_ai.joblib",
    ):
        # Isolation Forest -------------------------------------------------------
        try:
            self.model = IsolationForestModel.load(model_path)
            logger.info("MarketTrapEngine: Anomaly model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "MarketTrapEngine: Could not load anomaly model from %s. Error: %s",
                model_path,
                exc,
            )
            self.model = None

        # Phase engine -----------------------------------------------------------
        self.phase_engine = TrapPhaseEngine()
        self.futures_client = None

        # Sequence model ---------------------------------------------------------
        self.sequence_model_path = sequence_model_path
        try:
            if Path(sequence_model_path).exists():
                self.sequence_model = SequenceTrapModel.load(sequence_model_path)
                logger.info("MarketTrapEngine: Sequence model loaded from %s", sequence_model_path)
            else:
                self.sequence_model = SequenceTrapModel(SequenceModelConfig(window_size=30))
                logger.info("MarketTrapEngine: Sequence model initialized (untrained).")
        except Exception as exc:
            logger.warning("Sequence model initialization failed: %s", exc)
            self.sequence_model = None

        # Adaptive blend / calibration / thresholds -----------------------------
        self.adaptive_ai_path = adaptive_ai_path
        try:
            if Path(adaptive_ai_path).exists():
                self.adaptive_ai = AdaptiveAIScorer.load(adaptive_ai_path)
                logger.info("MarketTrapEngine: Adaptive AI model loaded from %s", adaptive_ai_path)
            else:
                self.adaptive_ai = AdaptiveAIScorer(AdaptiveAIConfig(model_path=adaptive_ai_path))
                logger.info("MarketTrapEngine: Adaptive AI model initialized.")
        except Exception as exc:
            logger.warning("Adaptive AI initialization failed: %s", exc)
            self.adaptive_ai = AdaptiveAIScorer(AdaptiveAIConfig(model_path=adaptive_ai_path))

        # Runtime state ----------------------------------------------------------
        self._bar_counters: Dict[str, int] = {}
        self._feature_importance_cache: Dict[str, List[str]] = {}

        # Lightweight online learning -------------------------------------------
        self._online_samples = deque(maxlen=5000)
        self._online_lock = threading.Lock()
        self._online_executor = ThreadPoolExecutor(max_workers=1)
        self._online_training_inflight = False
        self._last_online_retrain_ts = 0.0
        self._online_retrain_interval_sec = 4 * 3600  # every few hours

    # ------------------------------------------------------------------
    # Core feature prep
    # ------------------------------------------------------------------

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "price" not in frame.columns and "close" in frame.columns:
            frame["price"] = frame["close"]
        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        frame["price"] = pd.to_numeric(frame["price"])
        frame["volume"] = pd.to_numeric(frame["volume"])

        frame["price_return"] = frame["price"].pct_change().fillna(0.0)
        frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        frame["volatility"] = frame["price_return"].rolling(window=10, min_periods=1).std().fillna(0.0)

        rolling_max = (
            frame["price"]
            .rolling(window=min(20, len(frame)), min_periods=1)
            .max()
            .shift(1)
            .fillna(frame["price"].iloc[0])
        )
        frame["breakout_strength"] = (frame["price"] - rolling_max) / rolling_max.replace(0, np.nan)
        frame["is_breakout"] = (frame["price"] > rolling_max).astype(int)
        frame["pv_divergence"] = ((frame["price_return"] > 0) & (frame["volume_change"] < 0)).astype(int)

        return frame.replace([np.inf, -np.inf], 0).fillna(0)

    def _get_derivatives_data(self, symbol: str):
        oi_series, funding_series, ls_series = [], [], []
        if self.futures_client is None:
            return oi_series, funding_series, ls_series

        try:
            oi_series = self.futures_client.get_oi_series(symbol, n=60)
        except Exception:
            pass
        try:
            funding_series = self.futures_client.get_funding_series(symbol, n=10)
        except Exception:
            pass
        try:
            ls_series = self.futures_client.get_ls_ratio_series(symbol, n=30)
        except Exception:
            pass
        return oi_series, funding_series, ls_series

    # ------------------------------------------------------------------
    # Phase-aware rule scoring
    # ------------------------------------------------------------------

    def phase_weights(self, symbol: str) -> Tuple[float, ...]:
        return self.phase_engine.get_phase_weights(symbol)

    @staticmethod
    def phase_multiplier(phase_confidence: float) -> float:
        return float(0.30 + (0.70 * float(np.clip(phase_confidence, 0.0, 1.0))))

    @staticmethod
    def spike_boost(components: Dict[str, float], anomaly_component: float) -> float:
        active = sum(1 for v in components.values() if v > 0.6)
        if anomaly_component > 0.7:
            active += 1
        if active >= 4:
            return 1.30
        if active >= 3:
            return 1.20
        if active >= 2:
            return 1.10
        return 1.00

    def _rule_score_phase_aware(
        self,
        symbol: str,
        components: Dict[str, float],
        anomaly_component: float,
        phase_confidence: float,
    ) -> float:
        w_struct, w_vol, w_mom, w_anom, w_liq, w_retail = self.phase_weights(symbol)

        raw_rule = (
            w_struct * components.get("structure_failure", 0.0)
            + w_vol * components.get("volume_behavior", 0.0)
            + w_mom * components.get("momentum_exhaustion", 0.0)
            + w_anom * anomaly_component
            + w_liq * components.get("liquidity_intelligence", 0.0)
            + w_retail * components.get("retail_behavior", 0.0)
        )

        scored = raw_rule * self.phase_multiplier(phase_confidence) * self.spike_boost(components, anomaly_component)
        return float(np.clip(scored * 100.0, 0.0, 100.0))

    @staticmethod
    def _apply_false_positive_guards(risk_score: float, components: Dict[str, float], phase: str, phase_age: int) -> float:
        min_dwell = {"ACCUMULATION": 8, "MANIPULATION": 2, "DISTRIBUTION": 3, "REVERSAL": 1}
        if phase in min_dwell and phase_age < min_dwell[phase]:
            risk_score *= 0.5

        active = sum(1 for v in components.values() if v > 0.4)
        if active < 2 and phase != "REVERSAL":
            risk_score = min(risk_score, 45.0)

        return float(np.clip(round(risk_score, 1), 0.0, 100.0))

    @staticmethod
    def _phase_to_numeric(phase: str) -> float:
        mapping = {
            "NEUTRAL": 0.0,
            "ACCUMULATION": 0.25,
            "MANIPULATION": 0.50,
            "DISTRIBUTION": 0.75,
            "REVERSAL": 1.00,
        }
        return float(mapping.get(phase, 0.0))

    # ------------------------------------------------------------------
    # Sequence + explainability
    # ------------------------------------------------------------------

    def _get_sequence_feature_columns(self, features: pd.DataFrame) -> List[str]:
        # Include all numeric engineered features with stable ordering.
        return [c for c in features.columns if np.issubdtype(features[c].dtype, np.number)]

    def _sequence_probability(self, features: pd.DataFrame, feature_cols: List[str]) -> float:
        if self.sequence_model is None or len(features) < 10 or not feature_cols:
            return 0.0

        window_df = features[feature_cols].tail(min(30, len(features))).copy()
        try:
            return float(np.clip(self.sequence_model.predict(window_df, feature_names=feature_cols), 0.0, 1.0))
        except Exception as exc:
            logger.debug("Sequence model predict failed: %s", exc)
            return 0.0

    def _ai_feature_importance(
        self,
        symbol: str,
        features: pd.DataFrame,
        seq_feature_cols: List[str],
        anomaly_component: float,
        sequence_probability: float,
        force: bool = False,
    ) -> List[str]:
        # Skip most bars for latency safety; refresh cache periodically.
        bar_idx = self._bar_counters.get(symbol, 0)
        if not force and bar_idx % 5 != 0:
            return self._feature_importance_cache.get(symbol, [])

        impacts: Dict[str, float] = {}

        # Candidate features limited for runtime safety.
        candidates = seq_feature_cols[:25]

        # 1) Isolation Forest perturbation importance.
        if self.model is not None and candidates:
            try:
                baseline_anom = self.model.predict_as_feature(features, symbol=symbol, dynamic_feature_columns=seq_feature_cols)
                last_idx = features.index[-1]
                for name in candidates:
                    std = float(features[name].tail(60).std()) if name in features.columns else 0.0
                    eps = 0.05 * (std if std > 1e-6 else 1.0)

                    pert = features.copy()
                    pert.at[last_idx, name] = float(pert.at[last_idx, name]) + eps
                    anom_pert = self.model.predict_as_feature(pert, symbol=symbol, dynamic_feature_columns=seq_feature_cols)
                    impacts[name] = impacts.get(name, 0.0) + abs(anom_pert - baseline_anom)
            except Exception as exc:
                logger.debug("Anomaly perturbation importance failed: %s", exc)

        # 2) Sequence perturbation importance.
        if self.sequence_model is not None and seq_feature_cols:
            try:
                window_df = features[seq_feature_cols].tail(min(30, len(features))).copy()
                seq_impacts = self.sequence_model.feature_perturbation_importance(
                    window_df,
                    feature_names=seq_feature_cols,
                    epsilon_scale=0.05,
                    max_features=25,
                )
                for name, val in seq_impacts.items():
                    impacts[name] = impacts.get(name, 0.0) + float(val)
            except Exception as exc:
                logger.debug("Sequence perturbation importance failed: %s", exc)

        ranked = sorted(impacts.items(), key=lambda kv: kv[1], reverse=True)
        top_features = [name for name, _ in ranked[:5]]
        self._feature_importance_cache[symbol] = top_features
        return top_features

    # ------------------------------------------------------------------
    # Online adaptive retraining
    # ------------------------------------------------------------------

    def _enqueue_online_samples(self, symbol: str, train_df: pd.DataFrame):
        if train_df is None or train_df.empty:
            return
        with self._online_lock:
            for _, row in train_df.tail(400).iterrows():
                self._online_samples.append(row.to_dict())

            enough_data = len(self._online_samples) >= 300
            interval_ok = (time.time() - self._last_online_retrain_ts) >= self._online_retrain_interval_sec
            if enough_data and interval_ok and not self._online_training_inflight:
                data_snapshot = pd.DataFrame(list(self._online_samples))
                self._online_training_inflight = True
                self._last_online_retrain_ts = time.time()
                self._online_executor.submit(self._online_retrain_task, symbol, data_snapshot)

    def _online_retrain_task(self, symbol: str, online_df: pd.DataFrame):
        try:
            # Adaptive blend/calibration retrain
            needed = ["rule_score", "sequence_score", "anomaly_score", "phase_confidence", "trap_label"]
            if all(c in online_df.columns for c in needed):
                adaptive_df = online_df[needed].dropna()
                if len(adaptive_df) >= 50:
                    self.adaptive_ai.fit(symbol, adaptive_df)

            # Sequence retrain (expanded feature set + labels)
            if self.sequence_model is not None and "trap_label" in online_df.columns:
                seq_df = online_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                seq_feature_cols = [c for c in seq_df.columns if c not in {"trap_label"}]
                if len(seq_df) >= 120 and len(seq_feature_cols) > 0:
                    win = int(np.clip(self.sequence_model.config.window_size, 20, 30))
                    self.sequence_model.fit_from_frame(
                        seq_df[seq_feature_cols + ["trap_label"]],
                        label_col="trap_label",
                        feature_columns=seq_feature_cols,
                        window_size=win,
                    )
                    self.sequence_model.save(self.sequence_model_path)

            logger.info("Online adaptive retrain complete for %s (n=%d)", symbol, len(online_df))
        except Exception as exc:
            logger.warning("Online adaptive retrain failed for %s: %s", symbol, exc)
        finally:
            with self._online_lock:
                self._online_training_inflight = False

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def get_risk_snapshot(self, symbol: str, df_1m: pd.DataFrame, return_feature_row: bool = False) -> Dict:
        symbol = str(symbol).upper()
        if df_1m is None or len(df_1m) < 10:
            out = {
                "risk_score": 0.0,
                "risk_level": "LOW",
                "trap_type": "Neutral",
                "reasons": [],
                "control": "Neutral",
                "components": {},
                "phase": "NEUTRAL",
                "phase_confidence": 0.0,
                "trap_direction": None,
                "raw_risk": 0.0,
                "calibrated_risk": 0.0,
                "dynamic_threshold": 70.0,
                "ai_feature_importance": [],
                "sequence_score": 0.0,
                "anomaly_score": 0.0,
            }
            if return_feature_row:
                out["_feature_row"] = {}
            return out

        bar_idx = self._bar_counters.get(symbol, 0) + 1
        self._bar_counters[symbol] = bar_idx

        # Pipeline preserved:
        # features -> liquidity -> retail -> anomaly -> sequence -> phase_update -> phase-aware scoring
        features = self.compute_features(df_1m)
        oi_series, funding_series, ls_series = self._get_derivatives_data(symbol)

        try:
            features = compute_all_liquidity_features(
                features,
                oi_series=oi_series,
                funding_series=funding_series,
                ls_series=ls_series,
                price_col="price",
            )
        except Exception as exc:
            logger.debug("Liquidity feature computation failed: %s", exc)

        try:
            features = compute_all_retail_features(
                features,
                ls_series=ls_series,
                funding_series=funding_series,
                oi_series=oi_series,
                price_col="price",
                volume_col="volume",
            )
        except Exception as exc:
            logger.debug("Retail feature computation failed: %s", exc)

        components, diagnostics = build_component_scores(features)

        # Phase context before transition (used as AI features).
        prev_phase = self.phase_engine.get_phase(symbol)
        prev_phase_conf = self.phase_engine.get_trap_confidence(symbol)
        features["phase_signal"] = self._phase_to_numeric(prev_phase)
        features["phase_conf_feature"] = prev_phase_conf

        # Dynamic anomaly score as feature in [0,1].
        anomaly_component = 0.0
        if self.model:
            try:
                dynamic_cols = features.select_dtypes(include=[np.number]).columns.tolist()
                anomaly_component = float(
                    np.clip(self.model.predict_as_feature(features, symbol=symbol, dynamic_feature_columns=dynamic_cols), 0.0, 1.0)
                )
            except Exception as exc:
                logger.error("Error in anomaly prediction: %s", exc)
                anomaly_component = 0.0

        features["anomaly_feature"] = anomaly_component

        # Sequence prediction over expanded feature set.
        seq_feature_cols = self._get_sequence_feature_columns(features)
        sequence_probability = self._sequence_probability(features, seq_feature_cols)
        sequence_score = float(np.clip(sequence_probability * 100.0, 0.0, 100.0))

        # Phase engine update after anomaly+sequence.
        phase_info = self.phase_engine.update(symbol, features, bar_idx)
        phase = phase_info["phase"]
        phase_confidence = phase_info["phase_confidence"]
        trap_direction = phase_info["trap_direction"]

        # Rule score.
        rule_score = self._rule_score_phase_aware(symbol, components, anomaly_component, phase_confidence)

        # Adaptive blend + fallback.
        model_usage = "fallback_0.6_0.4"
        try:
            raw_risk = self.adaptive_ai.predict_raw(
                rule_score=rule_score,
                sequence_score=sequence_score,
                anomaly_score=anomaly_component * 100.0,
                phase_confidence=phase_confidence,
                fallback_rule_weight=0.6,
                fallback_seq_weight=0.4,
            )
            if self.adaptive_ai.blend_model is not None:
                model_usage = "adaptive_blend_model"
        except Exception as exc:
            logger.warning("Adaptive blend predict failed: %s", exc)
            raw_risk = float(np.clip((0.6 * rule_score) + (0.4 * sequence_score), 0.0, 100.0))

        raw_risk = self._apply_false_positive_guards(raw_risk, components, phase, phase_info["phase_age"])

        # Calibration.
        calibration_applied = symbol in self.adaptive_ai.calibration_params
        try:
            calibrated_risk = float(np.clip(self.adaptive_ai.calibrate(symbol, raw_risk), 0.0, 1.0))
        except Exception:
            calibrated_risk = float(np.clip(raw_risk / 100.0, 0.0, 1.0))
            calibration_applied = False

        risk_score = float(np.clip((calibrated_risk * 100.0) if calibration_applied else raw_risk, 0.0, 100.0))

        # Per-asset threshold.
        dynamic_threshold = float(self.adaptive_ai.get_threshold(symbol, default=70.0))

        # Explainability (top 3–5 features).
        ai_feature_importance = self._ai_feature_importance(
            symbol=symbol,
            features=features,
            seq_feature_cols=seq_feature_cols,
            anomaly_component=anomaly_component,
            sequence_probability=sequence_probability,
            force=risk_score >= (dynamic_threshold * 0.85),
        )

        trap_type = classify_trap_type(components, anomaly_component, phase_direction=trap_direction)
        control = buyer_seller_control(features)
        reasons = extract_trap_reasons(components, diagnostics, anomaly_component, max_reasons=3)

        risk_level = "LOW"
        if risk_score >= dynamic_threshold:
            risk_level = "CRITICAL"
        elif risk_score >= 40:
            risk_level = "ELEVATED"

        if risk_score < 40:
            reasons = []

        logger.debug(
            "AI scoring [%s] model=%s calibration=%s threshold=%.1f raw=%.1f risk=%.1f",
            symbol,
            model_usage,
            calibration_applied,
            dynamic_threshold,
            raw_risk,
            risk_score,
        )

        out = {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "trap_type": trap_type,
            "reasons": reasons,
            "control": control,
            "components": {
                **components,
                "anomaly": anomaly_component,
                "sequence": sequence_probability,
                "rule_score": rule_score / 100.0,
            },
            "diagnostics": diagnostics,
            "phase": phase,
            "phase_confidence": phase_confidence,
            "trap_direction": trap_direction,
            # Backward-compatible scores
            "rule_score": round(rule_score, 1),
            "sequence_score": round(sequence_score, 1),
            # New adaptive AI outputs
            "raw_risk": round(raw_risk, 1),
            "calibrated_risk": round(calibrated_risk, 4),
            "dynamic_threshold": round(dynamic_threshold, 1),
            "ai_feature_importance": ai_feature_importance,
            "anomaly_score": round(anomaly_component * 100.0, 2),
            "model_usage": model_usage,
            "calibration_applied": calibration_applied,
            "threshold_used": round(dynamic_threshold, 1),
        }
        if return_feature_row:
            out["_feature_row"] = features.tail(1).to_dict("records")[0]
        return out

    # ------------------------------------------------------------------
    # Historical evaluation + training
    # ------------------------------------------------------------------

    def evaluate_on_historical(self, df: pd.DataFrame, symbol: str = "HIST_EVAL"):
        """
        Generate labels, evaluate, and update adaptive AI models.

        This method keeps the existing architecture but now also:
        - trains adaptive blend model
        - fits calibration params
        - learns per-asset threshold
        - retrains sequence model on expanded features
        """
        symbol = str(symbol).upper()
        if df is None or df.empty:
            return BacktestEngine().run_backtest(pd.DataFrame({"risk_score": [], "trap_label": []}))

        frame = df.copy().reset_index(drop=True)
        if "price" not in frame.columns and "close" in frame.columns:
            frame["price"] = frame["close"]
        if "volume" not in frame.columns:
            frame["volume"] = 0.0

        self._bar_counters[symbol] = 0

        records = []
        sequence_rows = []
        for i in range(len(frame)):
            if i < 20:
                records.append(
                    {
                        "risk_score": 0.0,
                        "raw_risk": 0.0,
                        "rule_score": 0.0,
                        "sequence_score": 0.0,
                        "anomaly_score": 0.0,
                        "phase_confidence": 0.0,
                    }
                )
                sequence_rows.append({})
                continue

            window = frame.iloc[: i + 1].copy()
            snap = self.get_risk_snapshot(symbol, window, return_feature_row=True)
            records.append(
                {
                    "risk_score": float(snap.get("risk_score", 0.0)),
                    "raw_risk": float(snap.get("raw_risk", 0.0)),
                    "rule_score": float(snap.get("rule_score", 0.0)),
                    "sequence_score": float(snap.get("sequence_score", 0.0)),
                    "anomaly_score": float(snap.get("anomaly_score", 0.0)),
                    "phase_confidence": float(snap.get("phase_confidence", 0.0)),
                }
            )
            sequence_rows.append(snap.get("_feature_row", {}))

        scored = pd.concat([frame, pd.DataFrame(records)], axis=1)
        scored = generate_trap_labels(scored)

        # Train adaptive blend/calibration/threshold model.
        train_cols = ["rule_score", "sequence_score", "anomaly_score", "phase_confidence", "trap_label"]
        train_df = scored[train_cols].dropna().copy()
        if len(train_df) >= 30:
            try:
                self.adaptive_ai.fit(symbol, train_df)
            except Exception as exc:
                logger.warning("Adaptive AI training failed: %s", exc)

        # Sequence model retrain with expanded features + phase/anomaly context.
        if self.sequence_model is not None and len(scored) >= 80:
            try:
                seq_feature_df = pd.DataFrame(sequence_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                seq_train_df = pd.concat([seq_feature_df, scored[["trap_label"]]], axis=1)
                seq_feature_cols = [c for c in seq_train_df.columns if c != "trap_label"]
                win = int(np.clip(self.sequence_model.config.window_size, 20, 30))
                self.sequence_model.fit_from_frame(
                    seq_train_df,
                    label_col="trap_label",
                    feature_columns=seq_feature_cols,
                    window_size=win,
                )
                self.sequence_model.save(self.sequence_model_path)
            except Exception as exc:
                logger.warning("Sequence retraining failed: %s", exc)

        # Online-learning queue update (async, non-blocking).
        online_df = pd.concat(
            [
                seq_train_df.reset_index(drop=True) if 'seq_train_df' in locals() else pd.DataFrame(),
                scored[["rule_score", "sequence_score", "anomaly_score", "phase_confidence"]].reset_index(drop=True),
            ],
            axis=1,
        )
        self._enqueue_online_samples(symbol, online_df)

        # Backtest with learned threshold when available.
        learned_threshold = float(self.adaptive_ai.get_threshold(symbol, default=70.0))
        metrics = BacktestEngine(threshold=learned_threshold).run_backtest(scored[["risk_score", "trap_label"]])
        metrics["dynamic_threshold"] = learned_threshold
        return metrics
