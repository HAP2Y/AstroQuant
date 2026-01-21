"""
LSTM Predictor - Deep Learning for Time Series
===============================================
Uses LSTM neural networks combined with astrological
features for market prediction.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config.settings import DEFAULT_PREDICTION_CONFIG, PredictionConfig

# TensorFlow imports with graceful degradation
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class LSTMPrediction:
    """LSTM prediction result."""
    dates: List[datetime]
    predicted_values: List[float]
    confidence_scores: List[float]
    prediction_intervals: List[Tuple[float, float]]
    model_loss: float


class LSTMPredictor:
    """
    LSTM-based neural network predictor.

    Uses multiple features including price data and astrological
    sentiment scores for prediction.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize the LSTM predictor.

        Args:
            config: Optional prediction configuration.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM predictions. "
                            "Install with: pip install tensorflow")

        self.config = config or DEFAULT_PREDICTION_CONFIG
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 60  # Days of lookback
        self.feature_columns = []
        self._is_fitted = False

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Input data array.
            sequence_length: Length of each sequence.

        Returns:
            Tuple of (X, y) arrays.
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])  # Predict first column (typically price)
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, n_features).

        Returns:
            Compiled Keras model.
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )

        return model

    def _normalize_data(self, data: pd.DataFrame) -> np.ndarray:
        """Normalize data for training."""
        from sklearn.preprocessing import MinMaxScaler

        if self.scaler_X is None:
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            normalized = self.scaler_X.fit_transform(data)
        else:
            normalized = self.scaler_X.transform(data)

        return normalized

    def _inverse_normalize(self, data: np.ndarray, column_idx: int = 0) -> np.ndarray:
        """Inverse normalize predictions."""
        # Create dummy array with zeros for other features
        dummy = np.zeros((len(data), self.scaler_X.n_features_in_))
        dummy[:, column_idx] = data.flatten()
        inverse = self.scaler_X.inverse_transform(dummy)
        return inverse[:, column_idx]

    def prepare_features(self, price_data: pd.DataFrame,
                        sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare feature set for model training.

        Args:
            price_data: DataFrame with OHLCV data.
            sentiment_data: Optional DataFrame with sentiment scores.

        Returns:
            DataFrame with all features.
        """
        features = pd.DataFrame(index=price_data.index)

        # Price features
        features['close'] = price_data['Close']
        features['open'] = price_data['Open']
        features['high'] = price_data['High']
        features['low'] = price_data['Low']
        features['volume'] = price_data['Volume']

        # Technical indicators
        features['returns'] = features['close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()
        features['rsi'] = self._calculate_rsi(features['close'])

        # Price momentum
        features['momentum_5'] = features['close'].pct_change(5)
        features['momentum_10'] = features['close'].pct_change(10)

        # High-Low range
        features['hl_range'] = (features['high'] - features['low']) / features['close']

        # Add sentiment features if available
        if sentiment_data is not None:
            # Align indices
            common_idx = features.index.intersection(sentiment_data.index)
            features = features.loc[common_idx]

            if 'sentiment_score' in sentiment_data.columns:
                features['sentiment'] = sentiment_data.loc[common_idx, 'sentiment_score']

            # Sentiment-derived features
            if 'sentiment' in features.columns:
                features['sentiment_ma5'] = features['sentiment'].rolling(window=5).mean()
                features['sentiment_change'] = features['sentiment'].diff()

        # Drop NaN rows
        features = features.dropna()

        self.feature_columns = features.columns.tolist()

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def fit(self, features: pd.DataFrame, epochs: Optional[int] = None,
           batch_size: Optional[int] = None, validation_split: float = 0.2) -> Dict:
        """
        Fit the LSTM model.

        Args:
            features: DataFrame with prepared features.
            epochs: Training epochs (default from config).
            batch_size: Batch size (default from config).
            validation_split: Validation data fraction.

        Returns:
            Training history dictionary.
        """
        epochs = epochs or self.config.lstm_epochs
        batch_size = batch_size or self.config.lstm_batch_size

        # Normalize data
        normalized = self._normalize_data(features)

        # Create sequences
        X, y = self._create_sequences(normalized, self.sequence_length)

        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build model
        self.model = self._build_model((self.sequence_length, features.shape[1]))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )

        self._is_fitted = True
        self._last_sequence = normalized[-self.sequence_length:]
        self._last_date = features.index[-1]

        return {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }

    def predict(self, steps: int,
               future_sentiment: Optional[List[float]] = None) -> LSTMPrediction:
        """
        Generate predictions for future time steps.

        Args:
            steps: Number of steps to predict.
            future_sentiment: Optional sentiment scores for future dates.

        Returns:
            LSTMPrediction with forecasted values.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")

        predictions = []
        confidence_scores = []
        current_sequence = self._last_sequence.copy()

        for i in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, -1)

            # Predict
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)

            # Calculate confidence based on prediction stability
            # Use dropout at inference for uncertainty estimation
            mc_predictions = []
            for _ in range(10):
                mc_pred = self.model(X_pred, training=True)
                mc_predictions.append(mc_pred.numpy()[0, 0])

            confidence = 1.0 - np.std(mc_predictions) / (np.mean(mc_predictions) + 1e-8)
            confidence_scores.append(max(0, min(1, confidence)))

            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[0] = pred  # Update price

            # Update sentiment if provided
            if future_sentiment and i < len(future_sentiment) and 'sentiment' in self.feature_columns:
                sent_idx = self.feature_columns.index('sentiment')
                new_row[sent_idx] = (future_sentiment[i] - 50) / 50  # Normalize

            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Inverse normalize predictions
        predictions_original = self._inverse_normalize(np.array(predictions))

        # Generate future dates
        future_dates = [self._last_date + timedelta(days=i+1) for i in range(steps)]

        # Calculate prediction intervals
        intervals = []
        for pred, conf in zip(predictions_original, confidence_scores):
            margin = pred * 0.05 * (2 - conf)  # Wider interval for lower confidence
            intervals.append((pred - margin, pred + margin))

        return LSTMPrediction(
            dates=future_dates,
            predicted_values=predictions_original.tolist(),
            confidence_scores=confidence_scores,
            prediction_intervals=intervals,
            model_loss=float(self.model.evaluate(
                self._last_sequence.reshape(1, self.sequence_length, -1),
                np.array([predictions[0]]),
                verbose=0
            )[0])
        )

    def evaluate(self, test_features: pd.DataFrame) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_features: DataFrame with test features.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Normalize
        normalized = self.scaler_X.transform(test_features)

        # Create sequences
        X_test, y_test = self._create_sequences(normalized, self.sequence_length)

        # Predict
        predictions = self.model.predict(X_test, verbose=0).flatten()

        # Inverse normalize
        predictions_original = self._inverse_normalize(predictions)
        y_test_original = self._inverse_normalize(y_test)

        # Calculate metrics
        mse = np.mean((y_test_original - predictions_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_original - predictions_original))
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100

        # Direction accuracy
        actual_direction = np.sign(np.diff(y_test_original))
        predicted_direction = np.sign(np.diff(predictions_original))
        direction_accuracy = np.mean(actual_direction == predicted_direction)

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.model:
            self.model.save(path)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.model = load_model(path)
        self._is_fitted = True


class EnsemblePredictor:
    """
    Ensemble predictor combining ARIMA and LSTM.

    Uses weighted averaging of predictions from multiple models
    for improved accuracy.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        from .time_series import ARIMAPredictor

        self.config = config or DEFAULT_PREDICTION_CONFIG
        self.arima = ARIMAPredictor(config)
        self.lstm = LSTMPredictor(config) if TF_AVAILABLE else None

        # Weights for ensemble (can be tuned)
        self.arima_weight = 0.4
        self.lstm_weight = 0.6

    def fit(self, price_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Fit all models in the ensemble.

        Args:
            price_data: DataFrame with OHLCV data.
            sentiment_data: Optional sentiment data.

        Returns:
            Training results dictionary.
        """
        results = {}

        # Prepare data
        arima_data = price_data[['Close']].copy()
        arima_data.columns = ['close']

        if sentiment_data is not None:
            arima_data = arima_data.join(sentiment_data[['sentiment_score']], how='inner')

        # Fit ARIMA
        self.arima.fit(arima_data, target_column='close',
                      sentiment_column='sentiment_score' if sentiment_data is not None else None)
        results['arima'] = {'fitted': True}

        # Fit LSTM if available
        if self.lstm:
            features = self.lstm.prepare_features(price_data, sentiment_data)
            lstm_results = self.lstm.fit(features)
            results['lstm'] = lstm_results

        return results

    def predict(self, steps: int, future_sentiment: Optional[List[float]] = None) -> Dict:
        """
        Generate ensemble predictions.

        Args:
            steps: Number of steps to predict.
            future_sentiment: Optional future sentiment scores.

        Returns:
            Dictionary with ensemble and individual predictions.
        """
        results = {}

        # ARIMA prediction
        arima_pred = self.arima.predict(steps, future_sentiment)
        results['arima'] = arima_pred

        # LSTM prediction
        if self.lstm:
            lstm_pred = self.lstm.predict(steps, future_sentiment)
            results['lstm'] = lstm_pred

            # Ensemble average
            ensemble_values = []
            ensemble_confidence = []

            for i in range(steps):
                arima_val = arima_pred.predicted_values[i]
                lstm_val = lstm_pred.predicted_values[i]
                lstm_conf = lstm_pred.confidence_scores[i]

                # Weighted average
                ensemble_val = (
                    arima_val * self.arima_weight +
                    lstm_val * self.lstm_weight
                )
                ensemble_values.append(ensemble_val)

                # Confidence is based on agreement between models
                agreement = 1.0 - abs(arima_val - lstm_val) / max(arima_val, lstm_val)
                ensemble_confidence.append(agreement * lstm_conf)

            results['ensemble'] = {
                'dates': arima_pred.dates,
                'predicted_values': ensemble_values,
                'confidence_scores': ensemble_confidence
            }
        else:
            results['ensemble'] = {
                'dates': arima_pred.dates,
                'predicted_values': arima_pred.predicted_values,
                'confidence_scores': [0.5] * steps  # Default confidence without LSTM
            }

        return results
