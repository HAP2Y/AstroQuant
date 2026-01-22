# AstroQuant - Market Intelligence Platform

A sophisticated market analysis platform combining **Deep Vedic Astrological cycles** with **Machine Learning** for predictive financial insights.

## Overview

AstroQuant fuses ancient Vedic wisdom with modern quantitative analysis to provide actionable financial foresight. The platform calculates planetary positions with high precision, incorporates complex Vedic factors, and synthesizes them into a dynamic "Market Sentiment Score" that reflects the cosmic weather affecting financial markets.

## Features

### Deep-Astro Quant Engine
- **High-precision planetary calculations** using astronomical algorithms
- **Vedic astrology integration**: Nakshatras (27 lunar mansions), Planetary Dignities, complex aspects
- **Market Sentiment Score (0-100)**: Dynamic score reflecting cosmic influences on markets
- **Real-time and forecasted planetary positions**

### Predictive Modeling & Signals
- **ARIMA time-series analysis** for trend prediction
- **LSTM neural networks** for pattern recognition
- **Ensemble predictions** combining multiple models
- **Buy/Sell signals** with confidence levels
- **Pivot point detection** for market turns

### Sector Intelligence
- **Planetary-to-sector mapping**: Mars (Defense/Energy), Mercury (Tech/Comms), etc.
- **Golden Phase/Stress Phase identification** for each sector
- **Sector rotation timing** based on planetary transits

### User Experience
- **Clean Streamlit interface** with interactive visualizations
- **Signal Finder**: Enter any ticker for predicted pivot points
- **Personal Relevance**: Birth chart analysis for personalized timing
- **Confidence gauges and signal calendars**

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AstroQuant.git
cd AstroQuant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure

```
AstroQuant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration and constants
├── core/
│   ├── __init__.py
│   ├── planetary_engine.py     # Planetary position calculations
│   ├── vedic_calculator.py     # Vedic astrology calculations
│   ├── sentiment_scorer.py     # Market Sentiment Score algorithm
│   └── sector_mapper.py        # Sector-planetary mapping
├── models/
│   ├── __init__.py
│   ├── time_series.py          # ARIMA predictor
│   ├── lstm_predictor.py       # LSTM neural network
│   └── signal_generator.py     # Buy/Sell signal generation
├── data/
│   ├── __init__.py
│   ├── market_data.py          # Market data fetcher
│   └── cache/                  # Data caching
├── ui/
│   ├── __init__.py
│   ├── components.py           # UI components
│   ├── visualizations.py       # Charts and graphs
│   └── pages/                  # Page modules
└── utils/
    ├── __init__.py
    └── helpers.py              # Utility functions
```

## Core Components

### Market Sentiment Score

The sentiment score (0-100) is calculated using weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Planetary Dignity | 20% | Strength of planets in their zodiac signs |
| Nakshatra Influence | 15% | Lunar mansion energies |
| Aspect Harmony | 20% | Planetary geometric relationships |
| Transit Strength | 25% | Current planetary movements |
| Lunar Phase | 10% | Moon phase and Tithi |
| Retrograde Impact | 10% | Effect of retrograde planets |

### Sector-Planetary Mapping

| Sector | Primary Planet | Secondary Planets |
|--------|---------------|-------------------|
| Technology | Mercury | Uranus, Rahu |
| Finance | Jupiter | Venus, Mercury |
| Healthcare | Moon | Jupiter, Neptune |
| Energy | Sun | Mars, Pluto |
| Defense | Mars | Saturn, Pluto |
| Real Estate | Saturn | Moon, Venus |
| Consumer Goods | Venus | Moon, Mercury |

### Signal Interpretation

| Score Range | Signal | Action |
|-------------|--------|--------|
| 75-100 | Strong Buy | Aggressive accumulation |
| 60-74 | Buy | Gradual position building |
| 40-59 | Neutral | Hold positions |
| 25-39 | Sell | Reduce exposure |
| 0-24 | Strong Sell | Defensive positioning |

## Usage

### Quick Start

1. Launch the application: `streamlit run app.py`
2. View the Dashboard for current cosmic overview
3. Use Signal Finder to analyze specific stocks
4. Check Sector Intelligence for sector rotation timing

### Signal Finder

Enter any stock ticker (e.g., AAPL, RELIANCE.NS, TSLA) to:
- View predicted buy/sell dates
- See confidence levels for each signal
- Identify high-probability pivot points
- Get price targets and stop-loss levels

### Personal Relevance

Enter your birth details to see:
- How current transits affect your personal timing
- Your Moon-Jupiter activation days
- Personalized financial timing recommendations

## API Reference

### PlanetaryEngine

```python
from core.planetary_engine import PlanetaryEngine

engine = PlanetaryEngine()

# Get all planetary positions
snapshot = engine.get_all_positions(datetime.now())

# Get lunar phase
lunar = engine.get_lunar_phase(datetime.now())

# Find next aspect
next_conjunction = engine.find_next_aspect(
    Planet.JUPITER, Planet.SATURN,
    aspect_degrees=0,  # Conjunction
    start_dt=datetime.now()
)
```

### MarketSentimentScorer

```python
from core.sentiment_scorer import MarketSentimentScorer

scorer = MarketSentimentScorer()

# Get current sentiment
sentiment = scorer.calculate_sentiment(datetime.now())
print(f"Score: {sentiment.overall_score}")
print(f"Signal: {sentiment.signal}")

# Get sentiment range
sentiments = scorer.get_sentiment_range(
    start_dt=datetime.now(),
    end_dt=datetime.now() + timedelta(days=30)
)
```

### SignalGenerator

```python
from models.signal_generator import SignalGenerator

generator = SignalGenerator()

# Generate signal calendar
calendar = generator.generate_signal_calendar(
    ticker="AAPL",
    start_date=datetime.now(),
    days=90,
    current_price=150.0
)

# Find pivot points
pivots = generator.find_pivot_points(
    ticker="AAPL",
    start_date=datetime.now(),
    days=90
)
```

## Deployment

### Streamlit Community Cloud (Free)

The easiest way to deploy AstroQuant for free:

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app" and select:
   - Repository: `your-username/AstroQuant`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

### Alternative Deployment Options

- **Render**: Use the included `Procfile` for deployment
- **Hugging Face Spaces**: Create a Streamlit Space and upload the code
- **Railway**: Connect your GitHub repo for automatic deployment

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Linting**: Code quality checks with flake8
- **Import Validation**: Ensures all modules load correctly
- **Testing**: Runs pytest test suite
- **Security**: Dependency vulnerability scanning

Workflows run automatically on push to `main` and on pull requests.

## Disclaimer

**Important:** This platform is for educational and research purposes only. Astrological analysis should not be considered financial advice. Always:

- Conduct your own due diligence
- Consult qualified financial advisors
- Never invest more than you can afford to lose
- Past performance does not guarantee future results

The creators assume no responsibility for financial decisions made using this tool.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Swiss Ephemeris for astronomical calculations
- Vedic astrology traditions for timing wisdom
- Modern quantitative finance for ML integration
