"""
Isolation Forest Model for Anomaly Detection

This module implements an Isolation Forest-based anomaly detection model
that can process multiple cryptocurrency symbols with shared model weights.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class IsolationForestModel:
    """
    Wrapper for scikit-learn's IsolationForest with additional functionality
    for cryptocurrency market data.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        window_size: int = 100
    ):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
            random_state: Random seed for reproducibility
            window_size: Size of the rolling window for min-max normalization
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.score_window = {}  # Store score windows per symbol
        self.feature_columns = None
        self.is_fitted = False
        
    def _prepare_features(
        self,
        df: pd.DataFrame,
        dynamic_feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for model input.
        
        Dynamically aligns input features to the columns the model was
        trained on.  If the model has not been fitted yet, all numeric
        columns are used (first-fit captures the column set).
        
        Args:
            df: Input DataFrame with OHLCV and derived features
            
        Returns:
            Tuple of (features_array, feature_columns)
        """
        # Select only numeric columns and drop any remaining NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if dynamic_feature_columns:
            numeric_cols = [c for c in dynamic_feature_columns if c in numeric_cols]
        features = df[numeric_cols].dropna(axis=1)

        if self.feature_columns is None:
            # First fit — capture whatever columns are available
            self.feature_columns = features.columns.tolist()
        else:
            # Align features to the saved training columns
            missing = [col for col in self.feature_columns if col not in features.columns]
            for col in missing:
                features[col] = 0
            extra = [col for col in features.columns if col not in self.feature_columns]
            if extra:
                features = features.drop(columns=extra)
            features = features[self.feature_columns]

        return features.values, self.feature_columns
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> 'IsolationForestModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Training data (DataFrame or numpy array)
            y: Ignored (for scikit-learn compatibility)
            
        Returns:
            self: Returns an instance of self
        """
        if isinstance(X, pd.DataFrame):
            X, _ = self._prepare_features(X)
            
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict_anomaly_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        dynamic_feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict anomaly scores for the input data.
        
        Args:
            X: Input data (DataFrame or numpy array)
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        if isinstance(X, pd.DataFrame):
            X, _ = self._prepare_features(X, dynamic_feature_columns=dynamic_feature_columns)
            
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)  # Convert to positive (higher = more anomalous)

    def anomaly_score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        dynamic_feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """Backward-compatible alias for predict_anomaly_scores."""
        return self.predict_anomaly_scores(X, dynamic_feature_columns=dynamic_feature_columns)
    
    def compute_risk_percentage(
        self,
        scores: np.ndarray,
        symbol: str
    ) -> np.ndarray:
        """
        Convert anomaly scores to risk percentages using rolling window min-max normalization.
        
        Args:
            scores: Array of anomaly scores
            symbol: Cryptocurrency symbol for maintaining separate windows
            
        Returns:
            Array of risk percentages (0-100)
        """
        if symbol not in self.score_window:
            self.score_window[symbol] = []
            
        risk_scores = []
        
        for score in scores:
            # Update score window
            self.score_window[symbol].append(score)
            if len(self.score_window[symbol]) > self.window_size:
                self.score_window[symbol].pop(0)
                
            # Get current window
            window = np.array(self.score_window[symbol])
            
            # Calculate min and max in the window
            min_score = np.min(window)
            max_score = np.max(window)
            score_range = max_score - min_score
            
            # Avoid division by zero
            if score_range > 0:
                normalized = (score - min_score) / score_range
            else:
                normalized = 0.5  # Default to middle if no variation
                
            # Convert to percentage (0-100)
            risk_scores.append(min(max(normalized * 100, 0), 100))
            
        return np.array(risk_scores)

    def risk_percentage(self, scores: np.ndarray, symbol: str = "default") -> np.ndarray:
        """Backward-compatible risk percentage helper with default symbol."""
        return self.compute_risk_percentage(scores, symbol)
    
    def predict_latest_risk(
        self,
        df: pd.DataFrame,
        symbol: str = "default",
        dynamic_feature_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Convenience method to get the risk score and percentage for the latest data point.
        
        Args:
            df: Input DataFrame with features
            symbol: Cryptocurrency symbol
            
        Returns:
            Dict containing 'anomaly_score' and 'risk_percentage'
        """
        if not self.is_fitted:
            return {"anomaly_score": 0.0, "risk_percentage": 0.0}
            
        # Get anomaly scores for the whole batch to keep window consistent
        anomaly_scores = self.predict_anomaly_scores(df, dynamic_feature_columns=dynamic_feature_columns)
        risk_percentages = self.compute_risk_percentage(anomaly_scores, symbol)
        
        return {
            'anomaly_score': float(anomaly_scores[-1]),
            'risk_percentage': float(risk_percentages[-1])
        }

    def predict_as_feature(
        self,
        df: pd.DataFrame,
        symbol: str = "default",
        dynamic_feature_columns: Optional[List[str]] = None
    ) -> float:
        """
        Return anomaly as a stable normalized feature in [0, 1].

        Uses the same rolling normalization path as risk_percentage for
        consistency across assets and runtime sessions.
        """
        if not self.is_fitted:
            return 0.0

        try:
            latest = self.predict_latest_risk(
                df=df,
                symbol=symbol,
                dynamic_feature_columns=dynamic_feature_columns,
            )
            return float(np.clip(latest.get("risk_percentage", 0.0) / 100.0, 0.0, 1.0))
        except Exception as exc:
            logger.debug("predict_as_feature failed: %s", exc)
            return 0.0
    
    def save(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Convert score_window to a regular dict for serialization
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size': self.window_size,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'score_window': {k: v for k, v in self.score_window.items()}
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the state
        joblib.dump(state, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'IsolationForestModel':
        """
        Load a saved model from disk. Handles both dictionary state and direct object pickling.
        """
        try:
            state = joblib.load(filepath)
            
            # If state is already an instance of cls, return it
            if isinstance(state, cls):
                logger.info(f"Loaded {cls.__name__} instance directly from {filepath}")
                return state
                
            # If state is a dictionary (new format)
            if isinstance(state, dict):
                model = cls(
                    contamination=state['model'].contamination,
                    n_estimators=len(state['model'].estimators_),
                    random_state=state['model'].random_state,
                    window_size=state['window_size']
                )
                model.model = state['model']
                model.scaler = state['scaler']
                model.feature_columns = state.get('feature_columns')
                model.is_fitted = state['is_fitted']
                model.score_window = state.get('score_window', {})
                logger.info(f"Loaded {cls.__name__} state dict from {filepath}")
                return model
                
            raise TypeError(f"Unknown model state type: {type(state)}")
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = IsolationForestModel(contamination=0.1, window_size=100)
    
    # Example: Train on some data
    # model.fit(X_train)
    
    # Example: Make predictions
    # results = model.predict(X_test, "BTC-USD")
    
    # Example: Save model
    # model.save("models/isolation_forest.pkl")
    
    # Example: Load model
    # loaded_model = IsolationForestModel.load("models/isolation_forest.pkl")
