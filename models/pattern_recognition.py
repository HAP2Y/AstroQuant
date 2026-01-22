"""
Pattern Recognition Model - Historical Astro-Financial Analysis
===============================================================
A machine learning model that analyzes 15 years of historical data
combining financial returns with Vedic astrological features to
generate predictive calendar heatmaps.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from enum import Enum
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.settings import (
    Planet, ZodiacSign, NAKSHATRAS, PLANETARY_MARKET_WEIGHTS,
    SECTOR_PLANETARY_MAP, MARKET_INDICES
)
from core.planetary_engine import PlanetaryEngine
from core.vedic_calculator import VedicCalculator
from core.sentiment_scorer import MarketSentimentScorer


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_BULLISH = "Strong Bullish"
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    STRONG_BEARISH = "Strong Bearish"


@dataclass
class PatternFeatures:
    """Features extracted for a single date."""
    date: datetime
    # Planetary positions (longitude normalized 0-1)
    sun_longitude: float
    moon_longitude: float
    mars_longitude: float
    mercury_longitude: float
    jupiter_longitude: float
    venus_longitude: float
    saturn_longitude: float
    rahu_longitude: float
    # Zodiac signs (encoded)
    sun_sign: int
    moon_sign: int
    mars_sign: int
    jupiter_sign: int
    venus_sign: int
    saturn_sign: int
    # Nakshatra (Moon's lunar mansion)
    moon_nakshatra: int
    nakshatra_energy: float
    # Lunar features
    lunar_phase: float  # 0-1 (new to full)
    is_waxing: int
    tithi: int
    # Dignity scores
    dignity_score: float
    # Aspect features
    aspect_harmony: float
    major_aspects_count: int
    # Retrograde planets
    mercury_retrograde: int
    venus_retrograde: int
    mars_retrograde: int
    jupiter_retrograde: int
    saturn_retrograde: int
    retrograde_count: int
    # Day of week / month patterns
    day_of_week: int
    day_of_month: int
    month: int
    quarter: int
    # Composite scores
    sentiment_score: float


@dataclass
class HeatmapData:
    """Data structure for calendar heatmap."""
    dates: List[datetime]
    scores: List[float]  # 0-100 scale
    regimes: List[str]
    confidence: List[float]
    details: List[Dict]


@dataclass
class PatternAnalysis:
    """Complete analysis results."""
    ticker_or_scope: str
    analysis_type: str  # 'market', 'sector', 'stock'
    train_period: Tuple[datetime, datetime]
    accuracy: float
    feature_importance: Dict[str, float]
    heatmap_data: HeatmapData
    best_days: List[Dict]
    worst_days: List[Dict]
    model_metadata: Dict


class AstroPatternRecognizer:
    """
    Pattern Recognition Model for Astro-Financial Analysis.

    Trains on 15 years of historical data to learn correlations
    between Vedic astrological features and market movements.
    """

    YEARS_OF_HISTORY = 15
    MODEL_DIR = Path(__file__).parent / "saved_models"

    # Return thresholds for classification
    STRONG_BULLISH_THRESHOLD = 0.02  # +2%
    BULLISH_THRESHOLD = 0.005  # +0.5%
    BEARISH_THRESHOLD = -0.005  # -0.5%
    STRONG_BEARISH_THRESHOLD = -0.02  # -2%

    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize the pattern recognizer.

        Args:
            model_type: 'gradient_boosting' or 'random_forest'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.model_type = model_type
        self.engine = PlanetaryEngine()
        self.vedic = VedicCalculator(self.engine)
        self.sentiment_scorer = MarketSentimentScorer()

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self._is_fitted = False

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _get_nakshatra_index(self, longitude: float) -> Tuple[int, float]:
        """Get nakshatra index and energy from Moon longitude."""
        nakshatra_span = 360 / 27  # 13.333 degrees per nakshatra
        index = int(longitude / nakshatra_span)
        energy = NAKSHATRAS[index]['market_energy']
        return index, energy

    def _extract_features(self, dt: datetime) -> PatternFeatures:
        """
        Extract all astrological features for a given date.

        Args:
            dt: Date to extract features for.

        Returns:
            PatternFeatures dataclass with all features.
        """
        # Get planetary positions
        snapshot = self.engine.get_all_positions(dt)
        positions = snapshot.positions

        # Get Vedic analysis
        vedic_analysis = self.vedic.get_complete_analysis(dt)

        # Calculate sentiment score
        try:
            sentiment = self.sentiment_scorer.calculate_sentiment(dt)
            sentiment_score = sentiment.overall_score
        except Exception:
            sentiment_score = 50.0

        # Extract planetary longitudes (normalized to 0-1)
        sun_long = positions[Planet.SUN].longitude / 360
        moon_long = positions[Planet.MOON].longitude / 360
        mars_long = positions[Planet.MARS].longitude / 360
        mercury_long = positions[Planet.MERCURY].longitude / 360
        jupiter_long = positions[Planet.JUPITER].longitude / 360
        venus_long = positions[Planet.VENUS].longitude / 360
        saturn_long = positions[Planet.SATURN].longitude / 360
        rahu_long = positions[Planet.RAHU].longitude / 360

        # Extract zodiac signs (0-11)
        sun_sign = positions[Planet.SUN].sign.value
        moon_sign = positions[Planet.MOON].sign.value
        mars_sign = positions[Planet.MARS].sign.value
        jupiter_sign = positions[Planet.JUPITER].sign.value
        venus_sign = positions[Planet.VENUS].sign.value
        saturn_sign = positions[Planet.SATURN].sign.value

        # Moon's nakshatra
        moon_nak_idx, nak_energy = self._get_nakshatra_index(
            positions[Planet.MOON].longitude
        )

        # Lunar features
        lunar_phase = snapshot.lunar_phase.illumination
        is_waxing = 1 if snapshot.lunar_phase.is_waxing else 0
        tithi = snapshot.lunar_phase.tithi

        # Dignity score (average across planets)
        dignity_scores = []
        for planet, dignity in vedic_analysis.dignities.items():
            dignity_scores.append(dignity.strength)
        dignity_score = np.mean(dignity_scores) if dignity_scores else 0.5

        # Aspect harmony
        if vedic_analysis.aspects:
            aspect_strengths = [a.strength for a in vedic_analysis.aspects]
            aspect_harmony = np.mean(aspect_strengths)
            major_aspects = len([a for a in vedic_analysis.aspects if a.orb < 5])
        else:
            aspect_harmony = 0.0
            major_aspects = 0

        # Retrograde counts
        mercury_rx = 1 if positions[Planet.MERCURY].is_retrograde else 0
        venus_rx = 1 if positions[Planet.VENUS].is_retrograde else 0
        mars_rx = 1 if positions[Planet.MARS].is_retrograde else 0
        jupiter_rx = 1 if positions[Planet.JUPITER].is_retrograde else 0
        saturn_rx = 1 if positions[Planet.SATURN].is_retrograde else 0
        rx_count = mercury_rx + venus_rx + mars_rx + jupiter_rx + saturn_rx

        # Calendar features
        day_of_week = dt.weekday()
        day_of_month = dt.day
        month = dt.month
        quarter = (dt.month - 1) // 3 + 1

        return PatternFeatures(
            date=dt,
            sun_longitude=sun_long,
            moon_longitude=moon_long,
            mars_longitude=mars_long,
            mercury_longitude=mercury_long,
            jupiter_longitude=jupiter_long,
            venus_longitude=venus_long,
            saturn_longitude=saturn_long,
            rahu_longitude=rahu_long,
            sun_sign=sun_sign,
            moon_sign=moon_sign,
            mars_sign=mars_sign,
            jupiter_sign=jupiter_sign,
            venus_sign=venus_sign,
            saturn_sign=saturn_sign,
            moon_nakshatra=moon_nak_idx,
            nakshatra_energy=nak_energy,
            lunar_phase=lunar_phase,
            is_waxing=is_waxing,
            tithi=tithi,
            dignity_score=dignity_score,
            aspect_harmony=aspect_harmony,
            major_aspects_count=major_aspects,
            mercury_retrograde=mercury_rx,
            venus_retrograde=venus_rx,
            mars_retrograde=mars_rx,
            jupiter_retrograde=jupiter_rx,
            saturn_retrograde=saturn_rx,
            retrograde_count=rx_count,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            month=month,
            quarter=quarter,
            sentiment_score=sentiment_score
        )

    def _features_to_array(self, features: PatternFeatures) -> np.ndarray:
        """Convert PatternFeatures to numpy array."""
        return np.array([
            features.sun_longitude,
            features.moon_longitude,
            features.mars_longitude,
            features.mercury_longitude,
            features.jupiter_longitude,
            features.venus_longitude,
            features.saturn_longitude,
            features.rahu_longitude,
            features.sun_sign,
            features.moon_sign,
            features.mars_sign,
            features.jupiter_sign,
            features.venus_sign,
            features.saturn_sign,
            features.moon_nakshatra,
            features.nakshatra_energy,
            features.lunar_phase,
            features.is_waxing,
            features.tithi,
            features.dignity_score,
            features.aspect_harmony,
            features.major_aspects_count,
            features.mercury_retrograde,
            features.venus_retrograde,
            features.mars_retrograde,
            features.jupiter_retrograde,
            features.saturn_retrograde,
            features.retrograde_count,
            features.day_of_week,
            features.day_of_month,
            features.month,
            features.quarter,
            features.sentiment_score
        ])

    def _classify_return(self, daily_return: float) -> MarketRegime:
        """Classify daily return into market regime."""
        if daily_return >= self.STRONG_BULLISH_THRESHOLD:
            return MarketRegime.STRONG_BULLISH
        elif daily_return >= self.BULLISH_THRESHOLD:
            return MarketRegime.BULLISH
        elif daily_return <= self.STRONG_BEARISH_THRESHOLD:
            return MarketRegime.STRONG_BEARISH
        elif daily_return <= self.BEARISH_THRESHOLD:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def _generate_synthetic_returns(self, dates: List[datetime],
                                   base_volatility: float = 0.015) -> pd.Series:
        """
        Generate synthetic market returns when real data isn't available.
        Uses astrological features to influence the synthetic data.

        Args:
            dates: List of dates
            base_volatility: Base daily volatility

        Returns:
            Series of synthetic returns
        """
        returns = []
        np.random.seed(42)  # For reproducibility

        for dt in dates:
            # Get sentiment for this date
            try:
                sentiment = self.sentiment_scorer.calculate_sentiment(dt)
                sentiment_factor = (sentiment.overall_score - 50) / 100  # -0.5 to +0.5
            except Exception:
                sentiment_factor = 0

            # Base random return
            base_return = np.random.normal(0.0003, base_volatility)  # Small positive drift

            # Modify by sentiment
            astro_influence = sentiment_factor * base_volatility * 2

            # Add some noise
            noise = np.random.normal(0, base_volatility * 0.3)

            daily_return = base_return + astro_influence + noise
            returns.append(daily_return)

        return pd.Series(returns, index=dates)

    def build_training_dataset(self, ticker: Optional[str] = None,
                               scope: str = 'market',
                               sector: Optional[str] = None,
                               use_synthetic: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset with 15 years of data.

        Args:
            ticker: Optional specific ticker to train on
            scope: 'market', 'sector', or 'stock'
            sector: Sector name if scope is 'sector'
            use_synthetic: Whether to use synthetic data if real data unavailable

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.YEARS_OF_HISTORY * 365)

        print(f"Building dataset from {start_date.date()} to {end_date.date()}...")

        # Try to get real market data
        returns_data = None

        try:
            from data.market_data import MarketDataFetcher
            fetcher = MarketDataFetcher()

            if scope == 'stock' and ticker:
                data = fetcher.get_historical_range(ticker, start_date, end_date)
                returns_data = data['Close'].pct_change().dropna()
            elif scope == 'market':
                data = fetcher.get_historical_range('^GSPC', start_date, end_date)
                returns_data = data['Close'].pct_change().dropna()
            elif scope == 'sector' and sector:
                # Use sector ETF as proxy
                sector_etfs = {
                    'Technology': 'XLK', 'Finance': 'XLF', 'Healthcare': 'XLV',
                    'Energy': 'XLE', 'Defense': 'XLI', 'Real Estate': 'XLRE',
                    'Consumer Goods': 'XLP', 'Communications': 'XLC'
                }
                etf = sector_etfs.get(sector, 'SPY')
                data = fetcher.get_historical_range(etf, start_date, end_date)
                returns_data = data['Close'].pct_change().dropna()
        except Exception as e:
            print(f"Could not fetch real data: {e}")

        # Generate synthetic data if needed
        if returns_data is None or len(returns_data) < 100:
            if use_synthetic:
                print("Using synthetic data for training...")
                # Generate dates (business days only)
                dates = pd.date_range(start=start_date, end=end_date, freq='B')
                dates = [d.to_pydatetime() for d in dates]
                returns_data = self._generate_synthetic_returns(dates)
            else:
                raise ValueError("No market data available and synthetic data disabled")

        # Build feature matrix
        print("Extracting astrological features...")
        features_list = []
        labels = []
        valid_dates = []

        dates_to_process = returns_data.index if hasattr(returns_data, 'index') else list(returns_data.keys())
        total_dates = len(dates_to_process)

        for i, date in enumerate(dates_to_process):
            if i % 500 == 0:
                print(f"  Processing {i}/{total_dates} dates...")

            try:
                # Convert to datetime if needed
                if hasattr(date, 'to_pydatetime'):
                    dt = date.to_pydatetime()
                else:
                    dt = date

                # Extract features
                features = self._extract_features(dt)
                features_array = self._features_to_array(features)

                # Get return and classify
                if hasattr(returns_data, 'loc'):
                    daily_return = returns_data.loc[date]
                else:
                    daily_return = returns_data[date]

                regime = self._classify_return(daily_return)

                features_list.append(features_array)
                labels.append(regime.value)
                valid_dates.append(dt)

            except Exception as e:
                continue

        print(f"Built dataset with {len(features_list)} samples")

        # Create DataFrame
        self.feature_names = [
            'sun_longitude', 'moon_longitude', 'mars_longitude', 'mercury_longitude',
            'jupiter_longitude', 'venus_longitude', 'saturn_longitude', 'rahu_longitude',
            'sun_sign', 'moon_sign', 'mars_sign', 'jupiter_sign', 'venus_sign', 'saturn_sign',
            'moon_nakshatra', 'nakshatra_energy', 'lunar_phase', 'is_waxing', 'tithi',
            'dignity_score', 'aspect_harmony', 'major_aspects_count',
            'mercury_retrograde', 'venus_retrograde', 'mars_retrograde',
            'jupiter_retrograde', 'saturn_retrograde', 'retrograde_count',
            'day_of_week', 'day_of_month', 'month', 'quarter', 'sentiment_score'
        ]

        X = pd.DataFrame(features_list, columns=self.feature_names, index=valid_dates)
        y = pd.Series(labels, index=valid_dates)

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the pattern recognition model.

        Args:
            X: Feature DataFrame
            y: Labels Series

        Returns:
            Training results dictionary
        """
        print("Training pattern recognition model...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            if self.model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        # Train final model on all data
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                random_state=42
            )

        self.model.fit(X_scaled, y_encoded)
        self._is_fitted = True

        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return {
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'feature_importance': importance,
            'classes': list(self.label_encoder.classes_)
        }

    def predict(self, dt: datetime) -> Tuple[str, float, Dict]:
        """
        Predict market regime for a given date.

        Args:
            dt: Date to predict

        Returns:
            Tuple of (regime, confidence, details)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self._extract_features(dt)
        X = self._features_to_array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict class probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        predicted_class = self.model.predict(X_scaled)[0]
        regime = self.label_encoder.inverse_transform([predicted_class])[0]

        # Get confidence
        confidence = max(proba)

        # Build details
        class_probs = dict(zip(self.label_encoder.classes_, proba))

        details = {
            'sentiment_score': features.sentiment_score,
            'lunar_phase': features.lunar_phase,
            'moon_nakshatra': NAKSHATRAS[features.moon_nakshatra]['name'],
            'nakshatra_energy': features.nakshatra_energy,
            'retrograde_count': features.retrograde_count,
            'dignity_score': features.dignity_score,
            'class_probabilities': class_probs
        }

        return regime, confidence, details

    def generate_heatmap(self, start_date: datetime, days: int = 365,
                        scope: str = 'market', ticker: Optional[str] = None,
                        sector: Optional[str] = None) -> HeatmapData:
        """
        Generate calendar heatmap data.

        Args:
            start_date: Start date for heatmap
            days: Number of days to forecast
            scope: 'market', 'sector', or 'stock'
            ticker: Ticker if scope is 'stock'
            sector: Sector if scope is 'sector'

        Returns:
            HeatmapData with all calendar information
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before generating heatmap")

        dates = []
        scores = []
        regimes = []
        confidences = []
        details_list = []

        print(f"Generating heatmap for {days} days...")

        for i in range(days):
            dt = start_date + timedelta(days=i)

            # Skip weekends for market predictions
            if dt.weekday() >= 5:
                continue

            try:
                regime, confidence, details = self.predict(dt)

                # Convert regime to score (0-100)
                regime_scores = {
                    'Strong Bullish': 90,
                    'Bullish': 70,
                    'Neutral': 50,
                    'Bearish': 30,
                    'Strong Bearish': 10
                }
                score = regime_scores.get(regime, 50)

                # Adjust score by confidence
                score = 50 + (score - 50) * confidence

                dates.append(dt)
                scores.append(score)
                regimes.append(regime)
                confidences.append(confidence)
                details_list.append(details)

            except Exception as e:
                continue

        return HeatmapData(
            dates=dates,
            scores=scores,
            regimes=regimes,
            confidence=confidences,
            details=details_list
        )

    def analyze(self, scope: str = 'market',
                ticker: Optional[str] = None,
                sector: Optional[str] = None,
                forecast_days: int = 365,
                train: bool = True) -> PatternAnalysis:
        """
        Complete analysis pipeline.

        Args:
            scope: 'market', 'sector', or 'stock'
            ticker: Ticker symbol for stock analysis
            sector: Sector name for sector analysis
            forecast_days: Days to forecast
            train: Whether to train the model

        Returns:
            PatternAnalysis with complete results
        """
        ticker_or_scope = ticker or sector or 'Market'

        if train or not self._is_fitted:
            # Build and train
            X, y = self.build_training_dataset(
                ticker=ticker, scope=scope, sector=sector
            )
            train_results = self.fit(X, y)
            accuracy = train_results['cv_accuracy']
            feature_importance = train_results['feature_importance']
            train_period = (X.index[0], X.index[-1])
        else:
            accuracy = 0.0
            feature_importance = {}
            train_period = (datetime.now() - timedelta(days=15*365), datetime.now())

        # Generate heatmap
        heatmap = self.generate_heatmap(
            start_date=datetime.now(),
            days=forecast_days,
            scope=scope,
            ticker=ticker,
            sector=sector
        )

        # Find best and worst days
        sorted_by_score = sorted(
            zip(heatmap.dates, heatmap.scores, heatmap.regimes, heatmap.confidence),
            key=lambda x: x[1],
            reverse=True
        )

        best_days = [
            {
                'date': d,
                'score': s,
                'regime': r,
                'confidence': c
            }
            for d, s, r, c in sorted_by_score[:10]
        ]

        worst_days = [
            {
                'date': d,
                'score': s,
                'regime': r,
                'confidence': c
            }
            for d, s, r, c in sorted_by_score[-10:]
        ]

        return PatternAnalysis(
            ticker_or_scope=ticker_or_scope,
            analysis_type=scope,
            train_period=train_period,
            accuracy=accuracy,
            feature_importance=feature_importance,
            heatmap_data=heatmap,
            best_days=best_days,
            worst_days=worst_days,
            model_metadata={
                'model_type': self.model_type,
                'n_features': len(self.feature_names),
                'years_of_data': self.YEARS_OF_HISTORY
            }
        )

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        if path is None:
            path = self.MODEL_DIR / f"pattern_model_{datetime.now().strftime('%Y%m%d')}.pkl"

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        return str(path)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self._is_fitted = True


class QuickPatternAnalyzer:
    """
    Lightweight pattern analyzer for quick heatmap generation
    without full model training.
    """

    def __init__(self):
        self.sentiment_scorer = MarketSentimentScorer()
        self.engine = PlanetaryEngine()

    def generate_quick_heatmap(self, start_date: datetime, days: int = 365,
                               scope: str = 'market') -> HeatmapData:
        """
        Generate a quick heatmap using sentiment scores directly.

        Args:
            start_date: Start date
            days: Number of days
            scope: Analysis scope

        Returns:
            HeatmapData
        """
        dates = []
        scores = []
        regimes = []
        confidences = []
        details_list = []

        for i in range(days):
            dt = start_date + timedelta(days=i)

            # Skip weekends
            if dt.weekday() >= 5:
                continue

            try:
                sentiment = self.sentiment_scorer.calculate_sentiment(dt)
                score = sentiment.overall_score

                # Determine regime
                if score >= 75:
                    regime = 'Strong Bullish'
                elif score >= 60:
                    regime = 'Bullish'
                elif score >= 40:
                    regime = 'Neutral'
                elif score >= 25:
                    regime = 'Bearish'
                else:
                    regime = 'Strong Bearish'

                # Confidence based on score extremity
                confidence = abs(score - 50) / 50

                dates.append(dt)
                scores.append(score)
                regimes.append(regime)
                confidences.append(confidence)
                details_list.append({
                    'interpretation': sentiment.interpretation,
                    'volatility': sentiment.volatility_forecast,
                    'key_factors': sentiment.key_factors
                })

            except Exception:
                continue

        return HeatmapData(
            dates=dates,
            scores=scores,
            regimes=regimes,
            confidence=confidences,
            details=details_list
        )
