"""
Visualizations - Charts and Graphs
==================================
Creates interactive visualizations using Plotly for the AstroQuant dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config.settings import Planet, SECTOR_PLANETARY_MAP
from core.sentiment_scorer import MarketSentiment
from core.sector_mapper import SectorAnalysis, SectorPhase
from models.signal_generator import TradingSignal, SignalType

try:
    from models.pattern_recognition import HeatmapData
    PATTERN_AVAILABLE = True
except ImportError:
    PATTERN_AVAILABLE = False
    HeatmapData = None


def create_sentiment_timeline(sentiments: List[MarketSentiment],
                             title: str = "Market Sentiment Timeline") -> go.Figure:
    """
    Create a timeline chart of sentiment scores.

    Args:
        sentiments: List of MarketSentiment objects.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    dates = [s.timestamp for s in sentiments]
    scores = [s.overall_score for s in sentiments]

    # Create color array based on score
    colors = []
    for score in scores:
        if score >= 70:
            colors.append('#00c853')
        elif score >= 55:
            colors.append('#64dd17')
        elif score >= 45:
            colors.append('#ffc107')
        elif score >= 30:
            colors.append('#ff9800')
        else:
            colors.append('#f44336')

    fig = go.Figure()

    # Main sentiment line
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#1976d2', width=2),
        marker=dict(size=6, color=colors),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Score: %{y:.1f}<extra></extra>'
    ))

    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="Strong Buy Zone", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
                  annotation_text="Neutral", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="Strong Sell Zone", annotation_position="right")

    # Shade regions
    fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_sector_heatmap(analyses: Dict[str, SectorAnalysis],
                         title: str = "Sector Cosmic Heatmap") -> go.Figure:
    """
    Create a heatmap of sector scores.

    Args:
        analyses: Dictionary of sector analyses.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    sectors = list(analyses.keys())
    scores = [analyses[s].score for s in sectors]

    # Create heatmap data
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=sectors,
        y=['Score'],
        colorscale=[
            [0, '#f44336'],
            [0.25, '#ff9800'],
            [0.5, '#ffc107'],
            [0.75, '#64dd17'],
            [1, '#00c853']
        ],
        zmin=0,
        zmax=100,
        hovertemplate='<b>%{x}</b><br>Score: %{z:.1f}<extra></extra>',
        showscale=True,
        colorbar=dict(title="Score")
    ))

    # Add phase annotations
    annotations = []
    for i, sector in enumerate(sectors):
        phase = analyses[sector].phase.value
        annotations.append(dict(
            x=sector,
            y='Score',
            text=f"{scores[i]:.0f}",
            showarrow=False,
            font=dict(color='white', size=12, weight='bold')
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45),
        annotations=annotations,
        height=200,
        template='plotly_white'
    )

    return fig


def create_signal_calendar(signals: List[TradingSignal],
                          title: str = "Signal Calendar") -> go.Figure:
    """
    Create a calendar view of trading signals.

    Args:
        signals: List of TradingSignal objects.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    dates = [s.date for s in signals]
    confidences = [s.confidence * 100 for s in signals]

    # Color by signal type
    colors = []
    symbols = []
    texts = []

    for signal in signals:
        if signal.signal_type == SignalType.STRONG_BUY:
            colors.append('#00c853')
            symbols.append('triangle-up')
            texts.append('SB')
        elif signal.signal_type == SignalType.BUY:
            colors.append('#64dd17')
            symbols.append('triangle-up-open')
            texts.append('B')
        elif signal.signal_type == SignalType.HOLD:
            colors.append('#ffc107')
            symbols.append('circle')
            texts.append('H')
        elif signal.signal_type == SignalType.SELL:
            colors.append('#ff9800')
            symbols.append('triangle-down-open')
            texts.append('S')
        else:
            colors.append('#f44336')
            symbols.append('triangle-down')
            texts.append('SS')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=confidences,
        mode='markers+text',
        marker=dict(
            size=15,
            color=colors,
            symbol=symbols,
            line=dict(width=2, color='white')
        ),
        text=texts,
        textposition='middle center',
        textfont=dict(size=8, color='white'),
        hovertemplate='<b>%{x|%b %d}</b><br>' +
                      'Confidence: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        template='plotly_white',
        height=350,
        showlegend=False
    )

    # Add legend manually
    legend_items = [
        ('Strong Buy', '#00c853', 'triangle-up'),
        ('Buy', '#64dd17', 'triangle-up-open'),
        ('Hold', '#ffc107', 'circle'),
        ('Sell', '#ff9800', 'triangle-down-open'),
        ('Strong Sell', '#f44336', 'triangle-down'),
    ]

    for i, (name, color, symbol) in enumerate(legend_items):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol=symbol),
            name=name,
            showlegend=True
        ))

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_planetary_chart(positions: Dict, title: str = "Planetary Wheel") -> go.Figure:
    """
    Create a circular chart showing planetary positions.

    Args:
        positions: Dictionary of planetary positions.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    # Create radial positions
    planet_data = []
    planet_symbols = {
        Planet.SUN: ("Sun", "#ffd700"),
        Planet.MOON: ("Moon", "#c0c0c0"),
        Planet.MARS: ("Mars", "#ff4500"),
        Planet.MERCURY: ("Mercury", "#9370db"),
        Planet.JUPITER: ("Jupiter", "#ffa500"),
        Planet.VENUS: ("Venus", "#00ff7f"),
        Planet.SATURN: ("Saturn", "#4682b4"),
        Planet.RAHU: ("Rahu", "#696969"),
        Planet.KETU: ("Ketu", "#8b4513"),
    }

    for planet, pos in positions.items():
        if planet in planet_symbols:
            name, color = planet_symbols[planet]
            planet_data.append({
                'name': name,
                'longitude': pos.longitude,
                'color': color,
                'sign': pos.sign.name,
                'retrograde': pos.is_retrograde
            })

    # Create polar chart
    fig = go.Figure()

    # Add zodiac wheel background
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

    for i, sign in enumerate(signs):
        start_angle = i * 30
        fig.add_trace(go.Barpolar(
            r=[1],
            theta=[start_angle + 15],
            width=[30],
            marker_color=['rgba(200,200,200,0.3)'] if i % 2 == 0 else ['rgba(150,150,150,0.3)'],
            name=sign,
            showlegend=False,
            hovertemplate=f'{sign}<extra></extra>'
        ))

    # Add planets
    for p in planet_data:
        retro_marker = " (R)" if p['retrograde'] else ""
        fig.add_trace(go.Scatterpolar(
            r=[0.7],
            theta=[p['longitude']],
            mode='markers+text',
            marker=dict(size=20, color=p['color']),
            text=[p['name'][0]],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            name=f"{p['name']}{retro_marker}",
            hovertemplate=f"<b>{p['name']}</b><br>{p['sign']} {p['longitude']:.1f}°{retro_marker}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode='array',
                tickvals=list(range(0, 360, 30)),
                ticktext=signs
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500,
        template='plotly_white'
    )

    return fig


def create_price_prediction_chart(historical: pd.DataFrame,
                                 predictions: Dict,
                                 ticker: str) -> go.Figure:
    """
    Create a chart showing historical prices and predictions.

    Args:
        historical: DataFrame with historical prices.
        predictions: Dictionary with predicted values.
        ticker: Stock ticker symbol.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Forecast', 'Confidence Level')
    )

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='#1976d2', width=2)
        ),
        row=1, col=1
    )

    # Predictions
    if 'dates' in predictions and 'predicted_values' in predictions:
        pred_dates = predictions['dates']
        pred_values = predictions['predicted_values']

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='#00c853', width=2, dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Confidence intervals if available
        if 'confidence_lower' in predictions and 'confidence_upper' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=pred_dates + pred_dates[::-1],
                    y=predictions['confidence_upper'] + predictions['confidence_lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(0,200,83,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ),
                row=1, col=1
            )

        # Confidence scores
        if 'confidence_scores' in predictions:
            fig.add_trace(
                go.Bar(
                    x=pred_dates,
                    y=[c * 100 for c in predictions['confidence_scores']],
                    name='Confidence',
                    marker_color='#64dd17'
                ),
                row=2, col=1
            )

    fig.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Confidence %", range=[0, 100], row=2, col=1)

    return fig


def create_confidence_gauge(confidence: float, title: str = "Signal Confidence") -> go.Figure:
    """
    Create a gauge chart for confidence level.

    Args:
        confidence: Confidence value (0-1).
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcdd2'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))

    fig.update_layout(height=250, template='plotly_white')

    return fig


def create_sector_comparison(forecasts: Dict, metric: str = 'score') -> go.Figure:
    """
    Create a comparison chart of multiple sectors.

    Args:
        forecasts: Dictionary of sector forecasts.
        metric: Metric to compare ('score', 'golden_periods', etc.)

    Returns:
        Plotly Figure object.
    """
    sectors = list(forecasts.keys())
    scores = [forecasts[s].current_score for s in sectors]

    # Sort by score
    sorted_data = sorted(zip(sectors, scores), key=lambda x: x[1], reverse=True)
    sectors, scores = zip(*sorted_data)

    colors = []
    for score in scores:
        if score >= 70:
            colors.append('#00c853')
        elif score >= 55:
            colors.append('#64dd17')
        elif score >= 45:
            colors.append('#ffc107')
        elif score >= 30:
            colors.append('#ff9800')
        else:
            colors.append('#f44336')

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(sectors),
        orientation='h',
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition='inside'
    ))

    fig.update_layout(
        title="Sector Ranking by Cosmic Score",
        xaxis_title="Score",
        yaxis_title="Sector",
        xaxis=dict(range=[0, 100]),
        height=400,
        template='plotly_white'
    )

    return fig


def create_transit_timeline(events: List, title: str = "Upcoming Transits") -> go.Figure:
    """
    Create a timeline of upcoming transit events.

    Args:
        events: List of transit events.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text="No significant transits in this period",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    dates = [e.date for e in events]
    impacts = [e.market_impact * 100 for e in events]
    descriptions = [e.description for e in events]

    colors = ['green' if i > 0 else 'red' for i in impacts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=impacts,
        mode='markers+text',
        marker=dict(
            size=15,
            color=colors,
            symbol='diamond'
        ),
        text=[d[:20] + '...' if len(d) > 20 else d for d in descriptions],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{x|%b %d}</b><br>%{text}<br>Impact: %{y:.1f}%<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Market Impact (%)",
        template='plotly_white',
        height=350
    )

    return fig


def create_calendar_heatmap(heatmap_data, year: int = None,
                           title: str = "Astro-Financial Calendar Heatmap") -> go.Figure:
    """
    Create a calendar heatmap visualization for pattern recognition results.

    Args:
        heatmap_data: HeatmapData object or dict with dates, scores, regimes
        year: Optional year to filter (defaults to current year)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Handle both HeatmapData object and dict
    if hasattr(heatmap_data, 'dates'):
        dates = heatmap_data.dates
        scores = heatmap_data.scores
        regimes = heatmap_data.regimes
        confidence = heatmap_data.confidence
    else:
        dates = heatmap_data.get('dates', [])
        scores = heatmap_data.get('scores', [])
        regimes = heatmap_data.get('regimes', [])
        confidence = heatmap_data.get('confidence', [0.5] * len(dates))

    if not dates:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Filter by year if specified
    if year is None:
        year = datetime.now().year

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'date': dates,
        'score': scores,
        'regime': regimes,
        'confidence': confidence
    })

    # Filter to selected year
    df['year'] = df['date'].apply(lambda x: x.year if hasattr(x, 'year') else x.year)
    df = df[df['year'] == year].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {year}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    # Extract week and day of week
    df['week'] = df['date'].apply(lambda x: x.isocalendar()[1])
    df['dayofweek'] = df['date'].apply(lambda x: x.weekday())
    df['month'] = df['date'].apply(lambda x: x.month)

    # Create pivot table for heatmap
    pivot = df.pivot_table(
        values='score',
        index='dayofweek',
        columns='week',
        aggfunc='mean'
    )

    # Day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=list(range(1, 54)),
        y=day_labels[:len(pivot.index)],
        colorscale=[
            [0, '#d32f2f'],      # Strong Bearish - Red
            [0.25, '#ff9800'],   # Bearish - Orange
            [0.5, '#ffc107'],    # Neutral - Yellow
            [0.75, '#8bc34a'],   # Bullish - Light Green
            [1, '#2e7d32']       # Strong Bullish - Green
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(
            title="Score",
            tickvals=[10, 30, 50, 70, 90],
            ticktext=['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish']
        ),
        hovertemplate='Week %{x}<br>%{y}<br>Score: %{z:.1f}<extra></extra>'
    ))

    # Add month dividers and labels
    month_starts = df.groupby('month')['week'].min().to_dict()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    annotations = []
    for month, week in month_starts.items():
        annotations.append(dict(
            x=week,
            y=-0.8,
            text=month_names[month-1],
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='left'
        ))

    fig.update_layout(
        title=dict(text=f"{title} - {year}", font=dict(size=16)),
        xaxis=dict(
            title="Week of Year",
            showgrid=False,
            tickmode='array',
            tickvals=list(range(1, 54, 4)),
            ticktext=[str(i) for i in range(1, 54, 4)]
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            autorange='reversed'
        ),
        annotations=annotations,
        height=300,
        template='plotly_white'
    )

    return fig


def create_monthly_heatmap(heatmap_data, title: str = "Monthly Outlook") -> go.Figure:
    """
    Create a monthly breakdown heatmap showing average scores per month.

    Args:
        heatmap_data: HeatmapData object or dict
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if hasattr(heatmap_data, 'dates'):
        dates = heatmap_data.dates
        scores = heatmap_data.scores
    else:
        dates = heatmap_data.get('dates', [])
        scores = heatmap_data.get('scores', [])

    if not dates:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'score': scores})
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)

    # Group by year and month
    monthly = df.groupby(['year', 'month'])['score'].mean().reset_index()

    # Get unique years
    years = sorted(monthly['year'].unique())
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create matrix
    z_values = []
    for year in years:
        row = []
        for month in months:
            val = monthly[(monthly['year'] == year) & (monthly['month'] == month)]['score']
            row.append(val.values[0] if len(val) > 0 else None)
        z_values.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=month_names,
        y=[str(y) for y in years],
        colorscale=[
            [0, '#d32f2f'],
            [0.25, '#ff9800'],
            [0.5, '#ffc107'],
            [0.75, '#8bc34a'],
            [1, '#2e7d32']
        ],
        zmin=0,
        zmax=100,
        hovertemplate='%{y} %{x}<br>Score: %{z:.1f}<extra></extra>',
        colorbar=dict(title="Score")
    ))

    # Add text annotations
    annotations = []
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            if z_values[i][j] is not None:
                annotations.append(dict(
                    x=month_names[j],
                    y=str(year),
                    text=f"{z_values[i][j]:.0f}",
                    showarrow=False,
                    font=dict(
                        color='white' if z_values[i][j] < 40 or z_values[i][j] > 60 else 'black',
                        size=10
                    )
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        annotations=annotations,
        height=250,
        template='plotly_white'
    )

    return fig


def create_day_of_week_analysis(heatmap_data, title: str = "Day of Week Analysis") -> go.Figure:
    """
    Create bar chart showing average score by day of week.

    Args:
        heatmap_data: HeatmapData object or dict
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if hasattr(heatmap_data, 'dates'):
        dates = heatmap_data.dates
        scores = heatmap_data.scores
    else:
        dates = heatmap_data.get('dates', [])
        scores = heatmap_data.get('scores', [])

    if not dates:
        return go.Figure()

    df = pd.DataFrame({'date': dates, 'score': scores})
    df['dayofweek'] = df['date'].apply(lambda x: x.weekday())

    # Group by day of week
    daily = df.groupby('dayofweek')['score'].agg(['mean', 'std']).reset_index()
    daily['day_name'] = daily['dayofweek'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })

    # Filter to only weekdays if mostly weekday data
    if df['dayofweek'].max() <= 4:
        daily = daily[daily['dayofweek'] <= 4]

    colors = []
    for score in daily['mean']:
        if score >= 60:
            colors.append('#2e7d32')
        elif score >= 50:
            colors.append('#8bc34a')
        elif score >= 40:
            colors.append('#ffc107')
        else:
            colors.append('#d32f2f')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily['day_name'],
        y=daily['mean'],
        marker_color=colors,
        text=[f"{v:.1f}" for v in daily['mean']],
        textposition='outside',
        error_y=dict(
            type='data',
            array=daily['std'].fillna(0),
            visible=True
        )
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Neutral", annotation_position="right")

    fig.update_layout(
        title=title,
        xaxis_title="Day of Week",
        yaxis_title="Average Score",
        yaxis=dict(range=[0, 100]),
        height=350,
        template='plotly_white'
    )

    return fig


def create_score_distribution(heatmap_data, title: str = "Score Distribution") -> go.Figure:
    """
    Create histogram of score distribution.

    Args:
        heatmap_data: HeatmapData object or dict
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if hasattr(heatmap_data, 'scores'):
        scores = heatmap_data.scores
    else:
        scores = heatmap_data.get('scores', [])

    if not scores:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#1976d2',
        opacity=0.7
    ))

    # Add regime markers
    fig.add_vline(x=25, line_dash="dash", line_color="red",
                  annotation_text="Strong Sell", annotation_position="top")
    fig.add_vline(x=40, line_dash="dot", line_color="orange",
                  annotation_text="Sell")
    fig.add_vline(x=60, line_dash="dot", line_color="lightgreen",
                  annotation_text="Buy")
    fig.add_vline(x=75, line_dash="dash", line_color="green",
                  annotation_text="Strong Buy", annotation_position="top")

    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, 100]),
        height=300,
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_feature_importance_chart(importance: Dict[str, float],
                                   title: str = "Top Pattern Features") -> go.Figure:
    """
    Create horizontal bar chart of feature importance.

    Args:
        importance: Dictionary of feature name to importance score
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if not importance:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Get top 15 features
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features, importances = zip(*sorted_features)

    # Clean feature names for display
    clean_names = [f.replace('_', ' ').title() for f in features]

    fig = go.Figure(go.Bar(
        x=list(importances),
        y=clean_names,
        orientation='h',
        marker_color='#1976d2',
        text=[f"{v:.3f}" for v in importances],
        textposition='outside'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis=dict(autorange='reversed'),
        height=450,
        template='plotly_white',
        margin=dict(l=150)
    )

    return fig


def create_best_worst_days_chart(best_days: List[Dict], worst_days: List[Dict],
                                title: str = "Best & Worst Forecast Days") -> go.Figure:
    """
    Create comparison chart of best and worst predicted days.

    Args:
        best_days: List of best day dictionaries
        worst_days: List of worst day dictionaries
        title: Chart title

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 10 Best Days", "Top 10 Worst Days"))

    # Best days
    if best_days:
        best_dates = [d['date'].strftime('%b %d') for d in best_days[:10]]
        best_scores = [d['score'] for d in best_days[:10]]

        fig.add_trace(
            go.Bar(
                y=best_dates,
                x=best_scores,
                orientation='h',
                marker_color='#2e7d32',
                text=[f"{s:.0f}" for s in best_scores],
                textposition='outside',
                name='Best Days'
            ),
            row=1, col=1
        )

    # Worst days
    if worst_days:
        worst_dates = [d['date'].strftime('%b %d') for d in worst_days[:10]]
        worst_scores = [d['score'] for d in worst_days[:10]]

        fig.add_trace(
            go.Bar(
                y=worst_dates,
                x=worst_scores,
                orientation='h',
                marker_color='#d32f2f',
                text=[f"{s:.0f}" for s in worst_scores],
                textposition='outside',
                name='Worst Days'
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(range=[0, 100], row=1, col=1)
    fig.update_xaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)

    return fig
