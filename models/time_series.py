"""
Time Series Predictor - ARIMA-based forecasting
================================================
Uses ARIMA models combined with astrological features
for time series prediction.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

from config.settings import DEFAULT_PREDICTION_CONFIG, PredictionConfig


@dataclass
class ARIMAPrediction:
    """ARIMA prediction result."""
    dates: List[datetime]
    predicted_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_params: Tuple[int, int, int]
    aic: float
    rmse: Optional[float]


class ARIMAPredictor:
    """
    ARIMA-based time series predictor.

    Combines traditional ARIMA modeling with astrological sentiment
    features for enhanced prediction.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize the ARIMA predictor.

        Args:
            config: Optional prediction configuration.
        """
        self.config = config or DEFAULT_PREDICTION_CONFIG
        self.model = None
        self.fitted_model = None
        self.last_train_data = None

    def _check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        Check if a series is stationary using Augmented Dickey-Fuller test.

        Args:
            series: Time series to test.

        Returns:
            Tuple of (is_stationary, p_value).
        """
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        return is_stationary, p_value

    def _difference_series(self, series: pd.Series, d: int = 1) -> pd.Series:
        """Apply differencing to make series stationary."""
        result = series.copy()
        for _ in range(d):
            result = result.diff().dropna()
        return result

    def _find_optimal_params(self, series: pd.Series,
                            max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA parameters using AIC.

        Args:
            series: Time series data.
            max_p: Maximum AR order.
            max_d: Maximum differencing order.
            max_q: Maximum MA order.

        Returns:
            Tuple of (p, d, q) parameters.
        """
        best_aic = float('inf')
        best_params = self.config.arima_order

        # Check stationarity to determine d
        is_stationary, _ = self._check_stationarity(series)
        d_range = [0] if is_stationary else [1, 2]

        warnings.filterwarnings('ignore')

        for d in d_range:
            if d > max_d:
                continue
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                    except Exception:
                        continue

        warnings.filterwarnings('default')
        return best_params

    def fit(self, data: pd.DataFrame, target_column: str = 'close',
           sentiment_column: Optional[str] = 'sentiment_score',
           auto_params: bool = False) -> None:
        """
        Fit the ARIMA model to historical data.

        Args:
            data: DataFrame with datetime index and price data.
            target_column: Column name for target variable.
            sentiment_column: Optional column for sentiment scores.
            auto_params: If True, automatically find optimal parameters.
        """
        series = data[target_column].dropna()

        if auto_params:
            params = self._find_optimal_params(series)
        else:
            params = self.config.arima_order

        # If sentiment column exists, use as exogenous variable
        exog = None
        if sentiment_column and sentiment_column in data.columns:
            exog = data[sentiment_column].dropna()
            # Align indices
            common_idx = series.index.intersection(exog.index)
            series = series.loc[common_idx]
            exog = exog.loc[common_idx]

        self.model = ARIMA(series, order=params, exog=exog)
        self.fitted_model = self.model.fit()
        self.last_train_data = series
        self._order = params
        self._exog_used = exog is not None

    def predict(self, steps: int, future_sentiment: Optional[List[float]] = None,
               confidence_level: float = 0.95) -> ARIMAPrediction:
        """
        Generate predictions for future time steps.

        Args:
            steps: Number of steps to predict.
            future_sentiment: Optional sentiment scores for future dates.
            confidence_level: Confidence level for prediction intervals.

        Returns:
            ARIMAPrediction with forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting")

        exog_future = None
        if self._exog_used and future_sentiment:
            exog_future = np.array(future_sentiment[:steps])

        # Get forecast with confidence intervals
        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog_future)
        predicted = forecast.predicted_mean

        # Get confidence intervals
        conf_int = forecast.conf_int(alpha=1 - confidence_level)

        # Generate future dates
        last_date = self.last_train_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]

        return ARIMAPrediction(
            dates=future_dates,
            predicted_values=predicted.tolist(),
            confidence_lower=conf_int.iloc[:, 0].tolist(),
            confidence_upper=conf_int.iloc[:, 1].tolist(),
            model_params=self._order,
            aic=self.fitted_model.aic,
            rmse=None
        )

    def evaluate(self, test_data: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Evaluate model performance on test data.

        Args:
            test_data: DataFrame with test data.
            target_column: Column name for target variable.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")

        actual = test_data[target_column].values
        predicted = self.fitted_model.forecast(steps=len(actual))

        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Direction accuracy
        if len(actual) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            direction_accuracy = np.mean(actual_direction == predicted_direction)
        else:
            direction_accuracy = None

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic
        }

    def rolling_forecast(self, data: pd.DataFrame, target_column: str = 'close',
                        window_size: int = 252, forecast_horizon: int = 5) -> pd.DataFrame:
        """
        Perform rolling window forecast for backtesting.

        Args:
            data: Full dataset.
            target_column: Target column name.
            window_size: Training window size.
            forecast_horizon: Steps to forecast.

        Returns:
            DataFrame with actual vs predicted values.
        """
        results = []
        series = data[target_column]

        for i in range(window_size, len(series) - forecast_horizon):
            train = series.iloc[i-window_size:i]

            try:
                model = ARIMA(train, order=self.config.arima_order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=forecast_horizon)

                for j, pred in enumerate(forecast):
                    if i + j < len(series):
                        results.append({
                            'date': series.index[i + j],
                            'actual': series.iloc[i + j],
                            'predicted': pred,
                            'horizon': j + 1
                        })
            except Exception:
                continue

        return pd.DataFrame(results)


class SentimentEnhancedARIMA(ARIMAPredictor):
    """
    ARIMA model enhanced with astrological sentiment features.

    Combines price-based ARIMA with sentiment scores to improve
    prediction accuracy.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        super().__init__(config)
        self.sentiment_weight = 0.3  # Weight for sentiment adjustment

    def fit_with_sentiment(self, price_data: pd.DataFrame,
                          sentiment_data: pd.DataFrame,
                          target_column: str = 'close') -> None:
        """
        Fit model combining price and sentiment data.

        Args:
            price_data: DataFrame with price data.
            sentiment_data: DataFrame with sentiment scores.
            target_column: Target column name.
        """
        # Merge data
        merged = price_data.join(sentiment_data, how='inner')
        merged = merged.dropna()

        # Fit base ARIMA
        self.fit(merged, target_column, sentiment_column='sentiment_score')

        # Store sentiment statistics for adjustment
        self._sentiment_mean = merged['sentiment_score'].mean()
        self._sentiment_std = merged['sentiment_score'].std()

    def predict_with_sentiment(self, steps: int,
                              future_sentiment: List[float]) -> ARIMAPrediction:
        """
        Generate predictions using both ARIMA and sentiment.

        Args:
            steps: Number of steps to predict.
            future_sentiment: Sentiment scores for future dates.

        Returns:
            ARIMAPrediction with sentiment-adjusted forecasts.
        """
        # Get base ARIMA prediction
        base_pred = self.predict(steps, future_sentiment)

        # Adjust based on sentiment deviation from mean
        adjusted_values = []
        for i, (pred, sent) in enumerate(zip(base_pred.predicted_values, future_sentiment)):
            # Calculate sentiment deviation
            sent_deviation = (sent - self._sentiment_mean) / self._sentiment_std

            # Apply adjustment (sentiment above mean increases prediction)
            adjustment = pred * sent_deviation * self.sentiment_weight * 0.01
            adjusted_values.append(pred + adjustment)

        return ARIMAPrediction(
            dates=base_pred.dates,
            predicted_values=adjusted_values,
            confidence_lower=base_pred.confidence_lower,
            confidence_upper=base_pred.confidence_upper,
            model_params=base_pred.model_params,
            aic=base_pred.aic,
            rmse=base_pred.rmse
        )
