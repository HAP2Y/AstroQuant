"""
Signal Generator - Trading Signal Generation
=============================================
Generates buy/sell signals by combining ML predictions
with astrological sentiment analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from config.settings import DEFAULT_PREDICTION_CONFIG, PredictionConfig
from core.sentiment_scorer import MarketSentimentScorer, MarketSentiment
from core.sector_mapper import SectorMapper, SectorPhase


class SignalType(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


@dataclass
class TradingSignal:
    """A trading signal for a specific date."""
    date: datetime
    signal_type: SignalType
    confidence: float  # 0-1
    price_target: Optional[float]
    stop_loss: Optional[float]
    sentiment_score: float  # 0-100
    ml_prediction: float  # Predicted price change %
    key_factors: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    time_horizon: str  # 'short', 'medium', 'long'


@dataclass
class SignalCalendar:
    """Calendar of trading signals for a date range."""
    ticker: str
    signals: List[TradingSignal]
    summary: Dict
    best_buy_dates: List[datetime]
    best_sell_dates: List[datetime]


class SignalGenerator:
    """
    Generates trading signals by combining:
    - Machine learning price predictions
    - Astrological sentiment scores
    - Sector-specific planetary influences
    - Technical indicators
    """

    # Signal thresholds
    STRONG_BUY_THRESHOLD = 75
    BUY_THRESHOLD = 60
    SELL_THRESHOLD = 40
    STRONG_SELL_THRESHOLD = 25

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.75
    MEDIUM_CONFIDENCE = 0.5

    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize the signal generator.

        Args:
            config: Optional prediction configuration.
        """
        self.config = config or DEFAULT_PREDICTION_CONFIG
        self.sentiment_scorer = MarketSentimentScorer()
        self.sector_mapper = SectorMapper()

    def _calculate_composite_score(self, sentiment: MarketSentiment,
                                  ml_prediction: float,
                                  sector_score: Optional[float] = None) -> float:
        """
        Calculate composite score from multiple signals.

        Args:
            sentiment: Astrological sentiment.
            ml_prediction: ML model prediction (% change).
            sector_score: Optional sector-specific score.

        Returns:
            Composite score (0-100).
        """
        # Base weights
        sentiment_weight = 0.4
        ml_weight = 0.4
        sector_weight = 0.2

        if sector_score is None:
            # Redistribute weights
            sentiment_weight = 0.5
            ml_weight = 0.5
            sector_weight = 0.0

        # Normalize ML prediction to 0-100 scale
        # Assume prediction range is roughly -10% to +10%
        ml_normalized = 50 + (ml_prediction * 500)  # Scale to 0-100
        ml_normalized = max(0, min(100, ml_normalized))

        composite = (
            sentiment.overall_score * sentiment_weight +
            ml_normalized * ml_weight +
            (sector_score or 0) * sector_weight
        )

        return composite

    def _determine_signal_type(self, score: float) -> SignalType:
        """Determine signal type based on composite score."""
        if score >= self.STRONG_BUY_THRESHOLD:
            return SignalType.STRONG_BUY
        elif score >= self.BUY_THRESHOLD:
            return SignalType.BUY
        elif score <= self.STRONG_SELL_THRESHOLD:
            return SignalType.STRONG_SELL
        elif score <= self.SELL_THRESHOLD:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _calculate_confidence(self, sentiment: MarketSentiment,
                            ml_confidence: float,
                            score: float) -> float:
        """
        Calculate signal confidence.

        Args:
            sentiment: Astrological sentiment.
            ml_confidence: ML model confidence.
            score: Composite score.

        Returns:
            Confidence level (0-1).
        """
        # Base confidence from ML
        base_confidence = ml_confidence

        # Sentiment alignment boost
        # If sentiment and ML agree, boost confidence
        sentiment_signal = sentiment.overall_score > 50
        ml_signal = score > 50

        if sentiment_signal == ml_signal:
            alignment_boost = 0.1
        else:
            alignment_boost = -0.1

        # Clarity boost (strong signals are more confident)
        clarity = abs(score - 50) / 50
        clarity_boost = clarity * 0.15

        # Volatility penalty
        if sentiment.volatility_forecast in ['high', 'extreme']:
            volatility_penalty = 0.1
        else:
            volatility_penalty = 0.0

        confidence = base_confidence + alignment_boost + clarity_boost - volatility_penalty

        return max(0.1, min(0.95, confidence))

    def _determine_risk_level(self, sentiment: MarketSentiment,
                             confidence: float) -> str:
        """Determine risk level for the signal."""
        if confidence >= self.HIGH_CONFIDENCE:
            if sentiment.volatility_forecast == 'low':
                return 'low'
            elif sentiment.volatility_forecast == 'moderate':
                return 'medium'
            else:
                return 'high'
        elif confidence >= self.MEDIUM_CONFIDENCE:
            if sentiment.volatility_forecast in ['low', 'moderate']:
                return 'medium'
            else:
                return 'high'
        else:
            return 'high'

    def _calculate_targets(self, current_price: float, signal_type: SignalType,
                          ml_prediction: float, confidence: float) -> Tuple[float, float]:
        """
        Calculate price target and stop loss.

        Args:
            current_price: Current price.
            signal_type: Signal type.
            ml_prediction: ML prediction (% change).
            confidence: Signal confidence.

        Returns:
            Tuple of (price_target, stop_loss).
        """
        # Adjust target based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)

        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            # Bullish targets
            target_move = abs(ml_prediction) * confidence_multiplier
            target_move = max(0.02, min(0.15, target_move))  # 2-15% range

            price_target = current_price * (1 + target_move)
            stop_loss = current_price * (1 - target_move * 0.5)

        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            # Bearish targets
            target_move = abs(ml_prediction) * confidence_multiplier
            target_move = max(0.02, min(0.15, target_move))

            price_target = current_price * (1 - target_move)
            stop_loss = current_price * (1 + target_move * 0.5)

        else:
            # Hold - no targets
            price_target = current_price
            stop_loss = current_price * 0.95

        return round(price_target, 2), round(stop_loss, 2)

    def generate_signal(self, ticker: str, date: datetime,
                       current_price: float,
                       ml_prediction: float,
                       ml_confidence: float = 0.5) -> TradingSignal:
        """
        Generate a trading signal for a specific ticker and date.

        Args:
            ticker: Stock ticker symbol.
            date: Date for signal.
            current_price: Current stock price.
            ml_prediction: ML model prediction (% change).
            ml_confidence: ML model confidence.

        Returns:
            TradingSignal with all details.
        """
        # Get astrological sentiment
        sentiment = self.sentiment_scorer.calculate_sentiment(date)

        # Check if ticker has sector mapping
        sector = self.sector_mapper.get_sector_for_ticker(ticker)
        sector_score = None

        if sector:
            sector_analysis = self.sector_mapper.analyze_sector(sector, date)
            sector_score = sector_analysis.score

        # Calculate composite score
        composite_score = self._calculate_composite_score(
            sentiment, ml_prediction, sector_score
        )

        # Determine signal type
        signal_type = self._determine_signal_type(composite_score)

        # Calculate confidence
        confidence = self._calculate_confidence(sentiment, ml_confidence, composite_score)

        # Calculate targets
        price_target, stop_loss = self._calculate_targets(
            current_price, signal_type, ml_prediction, confidence
        )

        # Determine risk level
        risk_level = self._determine_risk_level(sentiment, confidence)

        # Determine time horizon based on volatility
        if sentiment.volatility_forecast in ['low', 'moderate']:
            time_horizon = 'medium'
        else:
            time_horizon = 'short'

        # Compile key factors
        key_factors = sentiment.key_factors.copy()
        key_factors.append(f"ML Prediction: {ml_prediction*100:+.1f}%")
        if sector:
            key_factors.append(f"Sector ({sector}): {sector_score:.0f}")

        return TradingSignal(
            date=date,
            signal_type=signal_type,
            confidence=round(confidence, 3),
            price_target=price_target,
            stop_loss=stop_loss,
            sentiment_score=sentiment.overall_score,
            ml_prediction=ml_prediction,
            key_factors=key_factors[:8],
            risk_level=risk_level,
            time_horizon=time_horizon
        )

    def generate_signal_calendar(self, ticker: str,
                                start_date: datetime,
                                days: int,
                                current_price: float,
                                predictions: Optional[Dict] = None) -> SignalCalendar:
        """
        Generate a calendar of trading signals.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for calendar.
            days: Number of days to forecast.
            current_price: Current stock price.
            predictions: Optional ML predictions dictionary.

        Returns:
            SignalCalendar with all signals.
        """
        signals = []
        buy_dates = []
        sell_dates = []

        for i in range(days):
            date = start_date + timedelta(days=i)

            # Get ML prediction if available
            if predictions and 'predicted_values' in predictions:
                if i < len(predictions['predicted_values']):
                    # Calculate predicted change
                    pred_price = predictions['predicted_values'][i]
                    ml_prediction = (pred_price - current_price) / current_price

                    if 'confidence_scores' in predictions:
                        ml_confidence = predictions['confidence_scores'][i]
                    else:
                        ml_confidence = 0.5
                else:
                    ml_prediction = 0.0
                    ml_confidence = 0.3
            else:
                # Use sentiment-only prediction
                sentiment = self.sentiment_scorer.calculate_sentiment(date)
                ml_prediction = (sentiment.overall_score - 50) / 500  # Convert to % prediction
                ml_confidence = 0.4

            signal = self.generate_signal(
                ticker, date, current_price, ml_prediction, ml_confidence
            )
            signals.append(signal)

            # Track best dates
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                if signal.confidence >= self.MEDIUM_CONFIDENCE:
                    buy_dates.append(date)
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                if signal.confidence >= self.MEDIUM_CONFIDENCE:
                    sell_dates.append(date)

        # Generate summary
        summary = self._generate_summary(signals)

        return SignalCalendar(
            ticker=ticker,
            signals=signals,
            summary=summary,
            best_buy_dates=buy_dates[:10],
            best_sell_dates=sell_dates[:10]
        )

    def _generate_summary(self, signals: List[TradingSignal]) -> Dict:
        """Generate summary statistics for signals."""
        if not signals:
            return {}

        signal_counts = {}
        for signal in signals:
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        avg_sentiment = sum(s.sentiment_score for s in signals) / len(signals)

        # Find highest confidence signals
        best_buy = max(
            (s for s in signals if s.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]),
            key=lambda x: x.confidence,
            default=None
        )

        best_sell = max(
            (s for s in signals if s.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]),
            key=lambda x: x.confidence,
            default=None
        )

        return {
            'total_signals': len(signals),
            'signal_distribution': signal_counts,
            'average_confidence': round(avg_confidence, 3),
            'average_sentiment': round(avg_sentiment, 2),
            'best_buy_date': best_buy.date if best_buy else None,
            'best_buy_confidence': best_buy.confidence if best_buy else None,
            'best_sell_date': best_sell.date if best_sell else None,
            'best_sell_confidence': best_sell.confidence if best_sell else None,
            'bullish_bias': sum(1 for s in signals if s.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]) / len(signals),
            'bearish_bias': sum(1 for s in signals if s.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]) / len(signals)
        }

    def find_pivot_points(self, ticker: str, start_date: datetime,
                         days: int = 90) -> List[Dict]:
        """
        Find high-probability pivot points (potential market turns).

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for search.
            days: Number of days to search.

        Returns:
            List of pivot point dictionaries.
        """
        pivots = []

        # Get sentiment scores for the range
        sentiments = self.sentiment_scorer.get_sentiment_range(
            start_date,
            start_date + timedelta(days=days)
        )

        # Look for significant sentiment changes (potential pivots)
        for i in range(1, len(sentiments) - 1):
            prev = sentiments[i-1]
            curr = sentiments[i]
            next_s = sentiments[i+1]

            # Check for local minimum (potential buy pivot)
            if curr.overall_score < prev.overall_score and curr.overall_score < next_s.overall_score:
                if curr.overall_score < 40:  # Significant low
                    confidence = (50 - curr.overall_score) / 50
                    pivots.append({
                        'date': curr.timestamp,
                        'type': 'potential_bottom',
                        'sentiment_score': curr.overall_score,
                        'confidence': round(confidence, 3),
                        'action': 'Watch for BUY opportunity',
                        'factors': curr.key_factors[:3]
                    })

            # Check for local maximum (potential sell pivot)
            elif curr.overall_score > prev.overall_score and curr.overall_score > next_s.overall_score:
                if curr.overall_score > 60:  # Significant high
                    confidence = (curr.overall_score - 50) / 50
                    pivots.append({
                        'date': curr.timestamp,
                        'type': 'potential_top',
                        'sentiment_score': curr.overall_score,
                        'confidence': round(confidence, 3),
                        'action': 'Watch for SELL opportunity',
                        'factors': curr.key_factors[:3]
                    })

            # Check for sentiment crossovers
            if (prev.overall_score < 50 <= curr.overall_score):
                pivots.append({
                    'date': curr.timestamp,
                    'type': 'bullish_crossover',
                    'sentiment_score': curr.overall_score,
                    'confidence': 0.6,
                    'action': 'Sentiment turning bullish',
                    'factors': curr.key_factors[:3]
                })
            elif (prev.overall_score >= 50 > curr.overall_score):
                pivots.append({
                    'date': curr.timestamp,
                    'type': 'bearish_crossover',
                    'sentiment_score': curr.overall_score,
                    'confidence': 0.6,
                    'action': 'Sentiment turning bearish',
                    'factors': curr.key_factors[:3]
                })

        # Sort by confidence
        pivots.sort(key=lambda x: x['confidence'], reverse=True)

        return pivots


class MarketIndexSignalGenerator(SignalGenerator):
    """
    Signal generator specialized for market indices.

    Uses broader astrological factors without sector-specific weighting.
    """

    def generate_market_signals(self, index_name: str, start_date: datetime,
                               days: int = 30) -> SignalCalendar:
        """
        Generate signals for a market index.

        Args:
            index_name: Name of index (e.g., 'S&P 500').
            start_date: Start date.
            days: Number of days.

        Returns:
            SignalCalendar for the index.
        """
        # Use index as ticker
        from config.settings import MARKET_INDICES
        ticker = MARKET_INDICES.get(index_name, index_name)

        # For indices, we rely more heavily on overall sentiment
        signals = []

        for i in range(days):
            date = start_date + timedelta(days=i)
            sentiment = self.sentiment_scorer.calculate_sentiment(date)

            # ML prediction based on sentiment for indices
            ml_prediction = (sentiment.overall_score - 50) / 1000  # Conservative

            signal = TradingSignal(
                date=date,
                signal_type=self._determine_signal_type(sentiment.overall_score),
                confidence=0.5 + (abs(sentiment.overall_score - 50) / 100),
                price_target=None,  # Not applicable for indices
                stop_loss=None,
                sentiment_score=sentiment.overall_score,
                ml_prediction=ml_prediction,
                key_factors=sentiment.key_factors,
                risk_level='medium',
                time_horizon='medium'
            )
            signals.append(signal)

        summary = self._generate_summary(signals)
        buy_dates = [s.date for s in signals if s.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]]
        sell_dates = [s.date for s in signals if s.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]]

        return SignalCalendar(
            ticker=index_name,
            signals=signals,
            summary=summary,
            best_buy_dates=buy_dates[:10],
            best_sell_dates=sell_dates[:10]
        )
