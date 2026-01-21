"""
UI Components - Reusable Streamlit Components
=============================================
Provides reusable UI components for the AstroQuant dashboard.
"""

import streamlit as st
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from config.settings import Planet, MARKET_INDICES, SECTOR_PLANETARY_MAP
from core.sentiment_scorer import MarketSentiment
from core.sector_mapper import SectorAnalysis, SectorPhase
from models.signal_generator import TradingSignal, SignalType


def render_sentiment_gauge(sentiment: MarketSentiment) -> None:
    """
    Render a sentiment gauge with score and interpretation.

    Args:
        sentiment: MarketSentiment object to display.
    """
    score = sentiment.overall_score

    # Determine color based on score
    if score >= 70:
        color = "#00c853"  # Green
        status = "Bullish"
    elif score >= 55:
        color = "#64dd17"  # Light green
        status = "Mildly Bullish"
    elif score >= 45:
        color = "#ffc107"  # Yellow
        status = "Neutral"
    elif score >= 30:
        color = "#ff9800"  # Orange
        status = "Mildly Bearish"
    else:
        color = "#f44336"  # Red
        status = "Bearish"

    # Create gauge-like display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; font-weight: bold; color: {color};">
                {score:.1f}
            </div>
            <div style="font-size: 18px; color: {color}; margin-top: 10px;">
                {status}
            </div>
            <div style="font-size: 14px; color: #888; margin-top: 5px;">
                Market Sentiment Score
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar visualization
    st.progress(score / 100)

    # Signal badge
    signal_colors = {
        'strong_buy': '#00c853',
        'buy': '#64dd17',
        'neutral': '#ffc107',
        'sell': '#ff9800',
        'strong_sell': '#f44336'
    }

    st.markdown(f"""
    <div style="text-align: center; margin-top: 15px;">
        <span style="background-color: {signal_colors[sentiment.signal]};
                     color: white; padding: 5px 15px; border-radius: 15px;
                     font-weight: bold;">
            {sentiment.signal.upper().replace('_', ' ')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Volatility forecast
    volatility_colors = {
        'low': '#00c853',
        'moderate': '#ffc107',
        'high': '#ff9800',
        'extreme': '#f44336'
    }

    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px; font-size: 14px;">
        Volatility Forecast:
        <span style="color: {volatility_colors[sentiment.volatility_forecast]};
                     font-weight: bold;">
            {sentiment.volatility_forecast.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_signal_card(signal: TradingSignal, show_targets: bool = True) -> None:
    """
    Render a trading signal card.

    Args:
        signal: TradingSignal to display.
        show_targets: Whether to show price targets.
    """
    signal_colors = {
        SignalType.STRONG_BUY: ('#00c853', 'Strong Buy'),
        SignalType.BUY: ('#64dd17', 'Buy'),
        SignalType.HOLD: ('#ffc107', 'Hold'),
        SignalType.SELL: ('#ff9800', 'Sell'),
        SignalType.STRONG_SELL: ('#f44336', 'Strong Sell'),
    }

    color, label = signal_colors[signal.signal_type]

    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px;
                    padding: 15px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between;
                        align-items: center;">
                <div>
                    <span style="font-size: 20px; font-weight: bold;">
                        {signal.date.strftime('%b %d, %Y')}
                    </span>
                </div>
                <div>
                    <span style="background-color: {color}; color: white;
                                 padding: 5px 15px; border-radius: 15px;
                                 font-weight: bold;">
                        {label}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Confidence", f"{signal.confidence*100:.1f}%")

        with col2:
            st.metric("Sentiment", f"{signal.sentiment_score:.1f}")

        with col3:
            st.metric("Risk Level", signal.risk_level.capitalize())

        if show_targets and signal.price_target:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price Target", f"${signal.price_target:.2f}")
            with col2:
                st.metric("Stop Loss", f"${signal.stop_loss:.2f}")

        # Key factors
        if signal.key_factors:
            st.markdown("**Key Factors:**")
            for factor in signal.key_factors[:4]:
                st.markdown(f"- {factor}")


def render_sector_card(analysis: SectorAnalysis) -> None:
    """
    Render a sector analysis card.

    Args:
        analysis: SectorAnalysis to display.
    """
    phase_colors = {
        SectorPhase.GOLDEN: ('#00c853', 'Golden Phase'),
        SectorPhase.FAVORABLE: ('#64dd17', 'Favorable'),
        SectorPhase.NEUTRAL: ('#ffc107', 'Neutral'),
        SectorPhase.STRESS: ('#ff9800', 'Stress Phase'),
        SectorPhase.CRITICAL: ('#f44336', 'Critical'),
    }

    color, label = phase_colors[analysis.phase]

    with st.container():
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding-left: 15px;
                    margin: 10px 0;">
            <div style="display: flex; justify-content: space-between;
                        align-items: center;">
                <div>
                    <span style="font-size: 18px; font-weight: bold;">
                        {analysis.sector_name}
                    </span>
                </div>
                <div>
                    <span style="background-color: {color}; color: white;
                                 padding: 3px 10px; border-radius: 10px;
                                 font-size: 12px; font-weight: bold;">
                        {label}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Score", f"{analysis.score:.1f}")
            st.caption(f"Confidence: {analysis.confidence*100:.0f}%")

        with col2:
            st.markdown(f"**Primary Planet Status:**")
            st.caption(analysis.primary_planet_status)

        # Recommended action
        st.info(analysis.recommended_action)


def render_ticker_input() -> Tuple[str, bool]:
    """
    Render a ticker input field with validation.

    Returns:
        Tuple of (ticker, is_valid).
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g., AAPL, RELIANCE.NS, TSLA",
            help="Enter a valid stock ticker symbol. For Indian stocks, add .NS (NSE) or .BO (BSE)"
        ).upper().strip()

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    # Quick selection buttons
    st.markdown("**Quick Select:**")
    quick_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "RELIANCE.NS", "TCS.NS"]

    cols = st.columns(len(quick_tickers))
    for i, t in enumerate(quick_tickers):
        with cols[i]:
            if st.button(t, key=f"quick_{t}", use_container_width=True):
                ticker = t

    return ticker, analyze_btn and bool(ticker)


def render_date_selector(default_days: int = 90) -> Tuple[datetime, int]:
    """
    Render date range selector.

    Returns:
        Tuple of (start_date, forecast_days).
    """
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today(),
            help="Select the starting date for analysis"
        )

    with col2:
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=default_days,
            help="Number of days to forecast"
        )

    return datetime.combine(start_date, datetime.min.time()), forecast_days


def render_personal_info_form() -> Optional[Dict]:
    """
    Render birth details form for personal relevance feature.

    Returns:
        Dictionary with birth details or None if not submitted.
    """
    st.markdown("### Personal Cosmic Profile")
    st.caption("Enter your birth details for personalized market timing insights")

    with st.form("birth_details"):
        col1, col2 = st.columns(2)

        with col1:
            birth_date = st.date_input(
                "Birth Date",
                value=date(1990, 1, 1),
                min_value=date(1920, 1, 1),
                max_value=date.today()
            )

            birth_time = st.time_input(
                "Birth Time",
                value=datetime.strptime("12:00", "%H:%M").time(),
                help="Enter your birth time if known"
            )

        with col2:
            birth_city = st.text_input(
                "Birth City",
                placeholder="e.g., New York, Mumbai",
                help="Enter your birth city for location-based calculations"
            )

            time_known = st.checkbox(
                "Birth time is accurate",
                value=False,
                help="Check if you know your exact birth time"
            )

        submitted = st.form_submit_button("Calculate Personal Profile", type="primary")

        if submitted:
            return {
                'birth_date': birth_date,
                'birth_time': birth_time if time_known else None,
                'birth_city': birth_city,
                'time_known': time_known
            }

    return None


def render_market_selector() -> str:
    """
    Render market/index selector.

    Returns:
        Selected market index name.
    """
    st.markdown("### Select Market")

    selected = st.selectbox(
        "Choose Market Index",
        options=list(MARKET_INDICES.keys()),
        index=0,
        help="Select a major market index to analyze"
    )

    return selected


def render_sector_selector(allow_multiple: bool = False) -> List[str]:
    """
    Render sector selector.

    Args:
        allow_multiple: Whether to allow multiple selections.

    Returns:
        List of selected sector names.
    """
    sectors = list(SECTOR_PLANETARY_MAP.keys())

    if allow_multiple:
        selected = st.multiselect(
            "Select Sectors",
            options=sectors,
            default=sectors[:3],
            help="Choose sectors to analyze"
        )
    else:
        selected = [st.selectbox(
            "Select Sector",
            options=sectors,
            help="Choose a sector to analyze"
        )]

    return selected


def render_component_scores(components) -> None:
    """
    Render sentiment component breakdown.

    Args:
        components: SentimentComponents object.
    """
    st.markdown("### Score Components")

    components_data = {
        "Planetary Dignity": components.planetary_dignity_score,
        "Nakshatra Influence": components.nakshatra_influence_score,
        "Aspect Harmony": components.aspect_harmony_score,
        "Transit Strength": components.transit_strength_score,
        "Lunar Phase": components.lunar_phase_score,
        "Retrograde Impact": components.retrograde_impact_score,
    }

    for name, score in components_data.items():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.progress(score)

        with col2:
            st.write(f"{score*100:.0f}%")

        st.caption(name)


def render_planetary_positions(positions: Dict) -> None:
    """
    Render current planetary positions.

    Args:
        positions: Dictionary of planet positions.
    """
    st.markdown("### Current Planetary Positions")

    cols = st.columns(4)

    planet_symbols = {
        Planet.SUN: "Sun",
        Planet.MOON: "Moon",
        Planet.MARS: "Mars",
        Planet.MERCURY: "Mercury",
        Planet.JUPITER: "Jupiter",
        Planet.VENUS: "Venus",
        Planet.SATURN: "Saturn",
        Planet.RAHU: "Rahu",
        Planet.KETU: "Ketu",
    }

    for i, (planet, pos) in enumerate(positions.items()):
        if planet in planet_symbols:
            with cols[i % 4]:
                retro_marker = " (R)" if pos.is_retrograde else ""
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;
                            border: 1px solid #ddd; border-radius: 5px;
                            margin: 5px 0;">
                    <div style="font-weight: bold;">{planet_symbols[planet]}</div>
                    <div style="font-size: 12px; color: #666;">
                        {pos.sign.name}{retro_marker}
                    </div>
                    <div style="font-size: 11px; color: #888;">
                        {pos.sign_degree:.1f}°
                    </div>
                </div>
                """, unsafe_allow_html=True)
