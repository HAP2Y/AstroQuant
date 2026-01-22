"""
AstroQuant - Market Intelligence Platform
==========================================
A sophisticated market analysis platform combining Deep Vedic Astrological
cycles with Machine Learning for predictive financial insights.

Run with: streamlit run app.py
"""

import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import MARKET_INDICES, SECTOR_PLANETARY_MAP, Planet
from core.planetary_engine import PlanetaryEngine
from core.vedic_calculator import VedicCalculator
from core.sentiment_scorer import MarketSentimentScorer
from core.sector_mapper import SectorMapper, SectorPhase
from models.signal_generator import SignalGenerator, MarketIndexSignalGenerator
from data.market_data import MarketDataFetcher
from ui.components import (
    render_sentiment_gauge, render_signal_card, render_sector_card,
    render_ticker_input, render_date_selector, render_personal_info_form,
    render_market_selector, render_sector_selector, render_component_scores,
    render_planetary_positions
)
from ui.visualizations import (
    create_sentiment_timeline, create_sector_heatmap, create_signal_calendar,
    create_planetary_chart, create_price_prediction_chart, create_confidence_gauge,
    create_sector_comparison, create_transit_timeline,
    create_calendar_heatmap, create_monthly_heatmap, create_day_of_week_analysis,
    create_score_distribution, create_feature_importance_chart, create_best_worst_days_chart
)
from models.pattern_recognition import QuickPatternAnalyzer, AstroPatternRecognizer

# Page configuration
st.set_page_config(
    page_title="AstroQuant - Market Intelligence",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1976d2, #7b1fa2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = PlanetaryEngine()
    st.session_state.vedic = VedicCalculator(st.session_state.engine)
    st.session_state.scorer = MarketSentimentScorer()
    st.session_state.sector_mapper = SectorMapper()
    st.session_state.signal_gen = SignalGenerator()
    st.session_state.market_signal_gen = MarketIndexSignalGenerator()

    try:
        st.session_state.data_fetcher = MarketDataFetcher()
    except ImportError:
        st.session_state.data_fetcher = None


def main():
    """Main application entry point."""

    # Header
    st.markdown('<h1 class="main-header">AstroQuant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Vedic Astrology Meets Machine Learning for Market Intelligence</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=AQ", width=100)
        st.markdown("### Navigation")

        page = st.radio(
            "Select Module",
            ["Dashboard", "Pattern Recognition", "Signal Finder", "Sector Intelligence",
             "Market Forecast", "Personal Relevance"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Current cosmic snapshot
        st.markdown("### Current Cosmic Weather")
        now = datetime.now()
        current_sentiment = st.session_state.scorer.calculate_sentiment(now)

        st.metric("Market Sentiment", f"{current_sentiment.overall_score:.1f}",
                 delta=f"{current_sentiment.signal.replace('_', ' ').title()}")

        st.caption(f"Volatility: {current_sentiment.volatility_forecast.upper()}")

        # Lunar phase
        lunar = st.session_state.engine.get_lunar_phase(now)
        st.markdown(f"**Lunar Phase:** {lunar.phase_name}")
        st.caption(f"Tithi: {lunar.tithi_name}")

        st.markdown("---")
        st.caption("Built with Vedic Wisdom & Modern AI")

    # Main content based on selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Pattern Recognition":
        render_pattern_recognition()
    elif page == "Signal Finder":
        render_signal_finder()
    elif page == "Sector Intelligence":
        render_sector_intelligence()
    elif page == "Market Forecast":
        render_market_forecast()
    elif page == "Personal Relevance":
        render_personal_relevance()


def render_dashboard():
    """Render the main dashboard."""

    # Current date sentiment
    st.markdown("## Today's Cosmic Market Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        now = datetime.now()
        sentiment = st.session_state.scorer.calculate_sentiment(now)
        render_sentiment_gauge(sentiment)

        st.markdown("### Key Factors")
        for factor in sentiment.key_factors[:5]:
            st.markdown(f"- {factor}")

    with col2:
        # Sentiment timeline for next 30 days
        start_dt = datetime.now()
        sentiments = st.session_state.scorer.get_sentiment_range(
            start_dt, start_dt + timedelta(days=30)
        )
        fig = create_sentiment_timeline(sentiments, "30-Day Sentiment Forecast")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sector overview
    st.markdown("## Sector Cosmic Heatmap")

    analyses = st.session_state.sector_mapper.analyze_all_sectors(datetime.now())
    fig = create_sector_heatmap(analyses)
    st.plotly_chart(fig, use_container_width=True)

    # Sector cards
    col1, col2, col3 = st.columns(3)

    # Sort sectors by score
    sorted_sectors = sorted(analyses.items(), key=lambda x: x[1].score, reverse=True)

    # Top performing sectors
    with col1:
        st.markdown("### Top Sectors (Golden Phase)")
        for name, analysis in sorted_sectors[:3]:
            if analysis.phase in [SectorPhase.GOLDEN, SectorPhase.FAVORABLE]:
                render_sector_card(analysis)

    with col2:
        st.markdown("### Neutral Sectors")
        neutral = [s for s in sorted_sectors if s[1].phase == SectorPhase.NEUTRAL]
        for name, analysis in neutral[:3]:
            render_sector_card(analysis)

    with col3:
        st.markdown("### Stressed Sectors")
        for name, analysis in sorted_sectors[-3:]:
            if analysis.phase in [SectorPhase.STRESS, SectorPhase.CRITICAL]:
                render_sector_card(analysis)

    st.markdown("---")

    # Planetary positions
    st.markdown("## Current Planetary Positions")
    snapshot = st.session_state.engine.get_all_positions(datetime.now())

    col1, col2 = st.columns(2)

    with col1:
        render_planetary_positions(snapshot.positions)

    with col2:
        fig = create_planetary_chart(snapshot.positions)
        st.plotly_chart(fig, use_container_width=True)


def render_pattern_recognition():
    """Render the Pattern Recognition page with calendar heatmaps."""

    st.markdown("## Pattern Recognition Heatmap")
    st.markdown("""
    Discover favorable and unfavorable trading days using our AI model trained on
    **15 years of historical financial and Vedic astrological data**.
    """)

    # Analysis scope selection
    st.markdown("### Select Analysis Scope")

    col1, col2 = st.columns(2)

    with col1:
        scope = st.selectbox(
            "Analysis Type",
            ["Market (Overall)", "Sector Specific", "Stock Specific"],
            help="Choose whether to analyze the overall market, a specific sector, or individual stock"
        )

    with col2:
        if scope == "Sector Specific":
            sector = st.selectbox(
                "Select Sector",
                list(SECTOR_PLANETARY_MAP.keys())
            )
            ticker = None
        elif scope == "Stock Specific":
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
            sector = None
        else:
            sector = None
            ticker = None

    # Year selection for heatmap
    current_year = datetime.now().year
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_year = st.selectbox(
            "Heatmap Year",
            [current_year, current_year + 1],
            help="Select year to view forecast"
        )

    with col2:
        forecast_days = st.slider(
            "Forecast Days",
            min_value=90,
            max_value=730,
            value=365,
            step=30,
            help="Number of days to forecast from today"
        )

    with col3:
        use_quick_mode = st.checkbox(
            "Quick Mode",
            value=True,
            help="Quick mode uses sentiment scores directly. Uncheck for full ML model (slower but more accurate)"
        )

    # Generate button
    if st.button("Generate Heatmap", type="primary"):
        scope_value = 'market' if scope == "Market (Overall)" else ('sector' if scope == "Sector Specific" else 'stock')
        scope_name = "Market" if scope == "Market (Overall)" else (sector if sector else ticker)

        with st.spinner(f"Generating calendar heatmap for {scope_name}..."):
            try:
                if use_quick_mode:
                    # Use quick analyzer
                    analyzer = QuickPatternAnalyzer()
                    heatmap_data = analyzer.generate_quick_heatmap(
                        start_date=datetime.now(),
                        days=forecast_days,
                        scope=scope_value
                    )
                    feature_importance = {}
                    accuracy = None
                else:
                    # Use full ML model (this will take longer)
                    st.info("Training pattern recognition model on 15 years of data... This may take a few minutes.")
                    analyzer = AstroPatternRecognizer()
                    analysis = analyzer.analyze(
                        scope=scope_value,
                        ticker=ticker,
                        sector=sector,
                        forecast_days=forecast_days
                    )
                    heatmap_data = analysis.heatmap_data
                    feature_importance = analysis.feature_importance
                    accuracy = analysis.accuracy

                # Store in session state for persistence
                st.session_state.heatmap_data = heatmap_data
                st.session_state.feature_importance = feature_importance
                st.session_state.heatmap_accuracy = accuracy
                st.session_state.heatmap_scope = scope_name

            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")
                return

    # Display results if available
    if 'heatmap_data' in st.session_state and st.session_state.heatmap_data:
        heatmap_data = st.session_state.heatmap_data
        scope_name = st.session_state.get('heatmap_scope', 'Market')

        # Summary metrics
        st.markdown("---")
        st.markdown(f"### {scope_name} Calendar Heatmap")

        if st.session_state.get('heatmap_accuracy'):
            st.caption(f"Model Accuracy: {st.session_state.heatmap_accuracy*100:.1f}%")

        # Calculate summary stats
        scores = heatmap_data.scores
        avg_score = sum(scores) / len(scores) if scores else 50
        bullish_days = len([s for s in scores if s >= 60])
        bearish_days = len([s for s in scores if s <= 40])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Score", f"{avg_score:.1f}")

        with col2:
            st.metric("Bullish Days", bullish_days)

        with col3:
            st.metric("Bearish Days", bearish_days)

        with col4:
            bullish_pct = bullish_days / len(scores) * 100 if scores else 0
            st.metric("Bullish %", f"{bullish_pct:.0f}%")

        # Main calendar heatmap
        st.markdown("### Annual Calendar Heatmap")
        fig = create_calendar_heatmap(
            heatmap_data,
            year=selected_year,
            title=f"{scope_name} Astro-Financial Outlook"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Additional visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Monthly Breakdown", "Day Analysis", "Best/Worst Days", "Score Distribution"
        ])

        with tab1:
            fig = create_monthly_heatmap(heatmap_data, f"{scope_name} Monthly Outlook")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = create_day_of_week_analysis(heatmap_data, "Average Score by Day of Week")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Find best and worst days
            sorted_data = sorted(
                zip(heatmap_data.dates, heatmap_data.scores, heatmap_data.regimes, heatmap_data.confidence),
                key=lambda x: x[1],
                reverse=True
            )

            best_days = [{'date': d, 'score': s, 'regime': r, 'confidence': c}
                        for d, s, r, c in sorted_data[:10]]
            worst_days = [{'date': d, 'score': s, 'regime': r, 'confidence': c}
                         for d, s, r, c in sorted_data[-10:]]

            fig = create_best_worst_days_chart(best_days, worst_days)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed lists
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Top 10 Best Days")
                for day in best_days:
                    st.markdown(f"**{day['date'].strftime('%b %d, %Y')}** - Score: {day['score']:.0f} ({day['regime']})")

            with col2:
                st.markdown("#### Top 10 Worst Days")
                for day in worst_days:
                    st.markdown(f"**{day['date'].strftime('%b %d, %Y')}** - Score: {day['score']:.0f} ({day['regime']})")

        with tab4:
            fig = create_score_distribution(heatmap_data, "Score Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance (if available from full model)
        if st.session_state.get('feature_importance'):
            st.markdown("### Pattern Feature Importance")
            st.markdown("These astrological features have the strongest correlation with market movements:")
            fig = create_feature_importance_chart(
                st.session_state.feature_importance,
                "Top Predictive Features"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interpretation guide
        with st.expander("How to Read the Heatmap"):
            st.markdown("""
            ### Color Guide
            - **Dark Green (75-100)**: Strong Bullish - Favorable cosmic alignment for gains
            - **Light Green (60-74)**: Bullish - Generally positive conditions
            - **Yellow (40-59)**: Neutral - Mixed signals, exercise caution
            - **Orange (25-39)**: Bearish - Challenging conditions
            - **Red (0-24)**: Strong Bearish - Unfavorable cosmic weather

            ### Using the Heatmap
            1. **Best Days**: Look for dark green patches for optimal entry points
            2. **Avoid**: Red patches indicate higher risk periods
            3. **Weekly Patterns**: Check the day-of-week analysis for recurring patterns
            4. **Monthly Outlook**: See which months have stronger cosmic support

            ### Disclaimer
            This analysis combines Vedic astrology with machine learning for educational purposes.
            Always conduct your own research and consult financial advisors before making investment decisions.
            """)


def render_signal_finder():
    """Render the Signal Finder page."""

    st.markdown("## Signal Finder")
    st.markdown("Enter a stock ticker to discover high-probability pivot points and trading signals.")

    ticker, analyze = render_ticker_input()

    if analyze and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Get current price
                if st.session_state.data_fetcher:
                    try:
                        info = st.session_state.data_fetcher.get_ticker_info(ticker)
                        current_price = info.current_price
                        st.success(f"**{info.name}** ({ticker}) - Current Price: ${current_price:.2f}")

                        if info.sector:
                            st.caption(f"Sector: {info.sector} | Industry: {info.industry}")
                    except Exception:
                        current_price = 100.0
                        st.warning("Could not fetch current price. Using placeholder.")
                else:
                    current_price = 100.0
                    st.info("Market data not available. Using placeholder price.")

                # Date selection
                start_date, forecast_days = render_date_selector(90)

                # Generate signals
                calendar = st.session_state.signal_gen.generate_signal_calendar(
                    ticker, start_date, forecast_days, current_price
                )

                # Summary metrics
                st.markdown("### Signal Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Signals", calendar.summary['total_signals'])

                with col2:
                    st.metric("Bullish Bias", f"{calendar.summary['bullish_bias']*100:.0f}%")

                with col3:
                    st.metric("Avg Confidence", f"{calendar.summary['average_confidence']*100:.0f}%")

                with col4:
                    st.metric("Avg Sentiment", f"{calendar.summary['average_sentiment']:.1f}")

                # Signal calendar visualization
                st.markdown("### Signal Calendar")
                fig = create_signal_calendar(calendar.signals)
                st.plotly_chart(fig, use_container_width=True)

                # Best dates
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Best Buy Dates")
                    if calendar.best_buy_dates:
                        for dt in calendar.best_buy_dates[:5]:
                            signal = next(s for s in calendar.signals if s.date == dt)
                            st.markdown(f"- **{dt.strftime('%b %d, %Y')}** - Conf: {signal.confidence*100:.0f}%")
                    else:
                        st.info("No strong buy signals in this period.")

                with col2:
                    st.markdown("### Best Sell Dates")
                    if calendar.best_sell_dates:
                        for dt in calendar.best_sell_dates[:5]:
                            signal = next(s for s in calendar.signals if s.date == dt)
                            st.markdown(f"- **{dt.strftime('%b %d, %Y')}** - Conf: {signal.confidence*100:.0f}%")
                    else:
                        st.info("No strong sell signals in this period.")

                # Detailed signals
                st.markdown("### Signal Details")

                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    signal_filter = st.multiselect(
                        "Filter by Signal Type",
                        ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                        default=["Strong Buy", "Buy", "Sell", "Strong Sell"]
                    )

                with col2:
                    min_confidence = st.slider("Minimum Confidence", 0, 100, 50) / 100

                # Display filtered signals
                filtered = [s for s in calendar.signals
                           if s.signal_type.value in signal_filter
                           and s.confidence >= min_confidence]

                for signal in filtered[:10]:
                    render_signal_card(signal, show_targets=True)

                # Pivot points
                st.markdown("### Key Pivot Points")
                pivots = st.session_state.signal_gen.find_pivot_points(
                    ticker, start_date, forecast_days
                )

                if pivots:
                    for pivot in pivots[:8]:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.markdown(f"**{pivot['date'].strftime('%b %d, %Y')}**")
                        with col2:
                            st.markdown(f"{pivot['type'].replace('_', ' ').title()}")
                        with col3:
                            st.markdown(f"Conf: {pivot['confidence']*100:.0f}%")
                        st.caption(pivot['action'])
                        st.markdown("---")
                else:
                    st.info("No significant pivot points detected.")

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")


def render_sector_intelligence():
    """Render the Sector Intelligence page."""

    st.markdown("## Sector Intelligence")
    st.markdown("Discover which sectors are in Golden Phase (Buy) or Stress Phase (Sell).")

    # Sector selection
    sectors = render_sector_selector(allow_multiple=True)

    if not sectors:
        st.warning("Please select at least one sector.")
        return

    # Date range
    start_date, forecast_days = render_date_selector(90)

    # Get forecasts
    with st.spinner("Analyzing sector cosmic influences..."):
        forecasts = {}
        for sector in sectors:
            forecasts[sector] = st.session_state.sector_mapper.forecast_sector(
                sector, start_date, forecast_days
            )

    # Comparison chart
    st.markdown("### Sector Comparison")
    fig = create_sector_comparison(forecasts)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed sector analysis
    for sector in sectors:
        forecast = forecasts[sector]
        analysis = st.session_state.sector_mapper.analyze_sector(sector, start_date)

        st.markdown(f"### {sector}")

        col1, col2, col3 = st.columns(3)

        with col1:
            phase_colors = {
                SectorPhase.GOLDEN: "green",
                SectorPhase.FAVORABLE: "lightgreen",
                SectorPhase.NEUTRAL: "yellow",
                SectorPhase.STRESS: "orange",
                SectorPhase.CRITICAL: "red",
            }
            st.markdown(f"**Current Phase:** :{phase_colors[forecast.current_phase]}[{forecast.current_phase.value}]")
            st.metric("Score", f"{forecast.current_score:.1f}")

        with col2:
            st.markdown("**Golden Periods:**")
            if forecast.golden_periods:
                for start, end in forecast.golden_periods[:3]:
                    st.caption(f"{start.strftime('%b %d')} - {end.strftime('%b %d')}")
            else:
                st.caption("None in forecast period")

        with col3:
            st.markdown("**Stress Periods:**")
            if forecast.stress_periods:
                for start, end in forecast.stress_periods[:3]:
                    st.caption(f"{start.strftime('%b %d')} - {end.strftime('%b %d')}")
            else:
                st.caption("None in forecast period")

        # Phase shifts
        if forecast.upcoming_shifts:
            st.markdown("**Upcoming Phase Shifts:**")
            for shift_date, new_phase, reason in forecast.upcoming_shifts[:5]:
                st.markdown(f"- **{shift_date.strftime('%b %d')}**: {new_phase.value} ({reason})")

        # Sector tickers
        tickers = st.session_state.sector_mapper.get_sector_tickers(sector)
        if tickers:
            st.caption(f"Related stocks: {', '.join(tickers[:5])}")

        st.markdown("---")


def render_market_forecast():
    """Render the Market Forecast page."""

    st.markdown("## Market Index Forecast")
    st.markdown("Get cosmic forecasts for major market indices.")

    # Market selection
    selected_market = render_market_selector()
    start_date, forecast_days = render_date_selector(30)

    with st.spinner(f"Generating forecast for {selected_market}..."):
        # Get market signals
        calendar = st.session_state.market_signal_gen.generate_market_signals(
            selected_market, start_date, forecast_days
        )

        # Get historical data if available
        historical = None
        if st.session_state.data_fetcher:
            try:
                historical = st.session_state.data_fetcher.get_index_data(selected_market, "6mo")
            except Exception:
                pass

        # Summary
        st.markdown(f"### {selected_market} Forecast Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average Sentiment",
                     f"{calendar.summary['average_sentiment']:.1f}")

        with col2:
            st.metric("Bullish Days",
                     f"{calendar.summary['bullish_bias']*100:.0f}%")

        with col3:
            st.metric("Bearish Days",
                     f"{calendar.summary['bearish_bias']*100:.0f}%")

        with col4:
            if calendar.summary['best_buy_date']:
                st.metric("Best Entry",
                         calendar.summary['best_buy_date'].strftime('%b %d'))
            else:
                st.metric("Best Entry", "N/A")

        # Visualization
        st.markdown("### Sentiment Timeline")
        sentiments = st.session_state.scorer.get_sentiment_range(
            start_date, start_date + timedelta(days=forecast_days)
        )
        fig = create_sentiment_timeline(sentiments, f"{selected_market} Cosmic Forecast")
        st.plotly_chart(fig, use_container_width=True)

        # Signal calendar
        st.markdown("### Signal Calendar")
        fig = create_signal_calendar(calendar.signals)
        st.plotly_chart(fig, use_container_width=True)

        # Transit events
        st.markdown("### Major Cosmic Events")
        events = st.session_state.vedic.calculate_transit_events(
            start_date, start_date + timedelta(days=forecast_days)
        )

        significant_events = [e for e in events if abs(e.market_impact) > 0.1]

        if significant_events:
            fig = create_transit_timeline(significant_events[:15])
            st.plotly_chart(fig, use_container_width=True)

            for event in significant_events[:10]:
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.markdown(f"**{event.date.strftime('%b %d, %Y')}**")
                with col2:
                    st.markdown(event.description)
                with col3:
                    impact_color = "green" if event.market_impact > 0 else "red"
                    st.markdown(f":{impact_color}[{event.market_impact*100:+.0f}%]")
        else:
            st.info("No major cosmic events in this period.")


def render_personal_relevance():
    """Render the Personal Relevance page."""

    st.markdown("## Personal Cosmic Profile")
    st.markdown("See how market trends might specifically impact your personal financial timing.")

    birth_details = render_personal_info_form()

    if birth_details:
        with st.spinner("Calculating your personal cosmic profile..."):
            birth_date = birth_details['birth_date']
            birth_dt = datetime.combine(birth_date, datetime.min.time())

            # Get birth chart positions
            birth_snapshot = st.session_state.engine.get_all_positions(birth_dt)

            st.markdown("### Your Birth Chart Overview")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Planetary Positions at Birth")
                render_planetary_positions(birth_snapshot.positions)

            with col2:
                fig = create_planetary_chart(birth_snapshot.positions, "Your Birth Chart")
                st.plotly_chart(fig, use_container_width=True)

            # Current transits to birth chart
            st.markdown("### Current Transits to Your Chart")

            current_snapshot = st.session_state.engine.get_all_positions(datetime.now())

            # Find significant transits
            st.markdown("#### Active Personal Transits")

            personal_transits = []
            for planet in [Planet.JUPITER, Planet.SATURN, Planet.RAHU, Planet.KETU]:
                current_pos = current_snapshot.positions[planet]
                birth_pos = birth_snapshot.positions[Planet.MOON]  # Transit to natal Moon

                # Check if transit planet is in same sign as natal Moon
                if current_pos.sign == birth_pos.sign:
                    transit_type = "conjunction"
                    impact = "significant"
                elif (current_pos.sign.value - birth_pos.sign.value) % 12 == 6:
                    transit_type = "opposition"
                    impact = "challenging"
                elif (current_pos.sign.value - birth_pos.sign.value) % 12 in [4, 8]:
                    transit_type = "trine"
                    impact = "favorable"
                else:
                    continue

                personal_transits.append({
                    'planet': planet.value,
                    'transit_type': transit_type,
                    'impact': impact,
                    'to': 'natal Moon'
                })

            if personal_transits:
                for transit in personal_transits:
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        st.markdown(f"**{transit['planet']}**")
                    with col2:
                        st.markdown(f"{transit['transit_type']} {transit['to']}")
                    with col3:
                        impact_color = "green" if transit['impact'] == 'favorable' else "orange"
                        st.markdown(f":{impact_color}[{transit['impact'].title()}]")
            else:
                st.info("No major personal transits currently active.")

            # Personal timing recommendations
            st.markdown("### Personal Timing Recommendations")

            # Generate personalized sentiment
            personal_sentiment = st.session_state.scorer.calculate_sentiment(datetime.now())

            # Adjust based on birth chart (simplified)
            moon_nakshatra = birth_snapshot.positions[Planet.MOON].nakshatra_index
            current_moon_nakshatra = current_snapshot.positions[Planet.MOON].nakshatra_index

            # Same nakshatra = heightened sensitivity
            if moon_nakshatra == current_moon_nakshatra:
                st.success("Moon is transiting your birth nakshatra - heightened financial intuition today!")

            st.markdown(f"""
            Based on your birth chart and current cosmic conditions:

            - **Overall Personal Alignment:** {personal_sentiment.overall_score:.0f}/100
            - **Best Action:** {personal_sentiment.signal.replace('_', ' ').title()}
            - **Personal Risk Tolerance Today:** {personal_sentiment.volatility_forecast.title()}

            **Personalized Advice:**
            {personal_sentiment.interpretation}
            """)

            # Upcoming personal windows
            st.markdown("### Upcoming Personal Opportunity Windows")

            # Find days when transiting Moon conjuncts natal Jupiter (wealth indicator)
            natal_jupiter_sign = birth_snapshot.positions[Planet.JUPITER].sign

            windows = []
            for i in range(30):
                check_date = datetime.now() + timedelta(days=i)
                check_snapshot = st.session_state.engine.get_all_positions(check_date)

                if check_snapshot.positions[Planet.MOON].sign == natal_jupiter_sign:
                    windows.append(check_date)

            if windows:
                st.markdown("**Moon-Jupiter Activation Days (favorable for wealth):**")
                for window in windows[:5]:
                    st.markdown(f"- {window.strftime('%B %d, %Y')}")
            else:
                st.info("Next Jupiter activation window is beyond 30 days.")


if __name__ == "__main__":
    main()
