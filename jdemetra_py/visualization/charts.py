"""High-level charting functions for time series visualization."""

from typing import Optional, List, Union, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..toolkit.timeseries import TsData
from ..sa.base.results import SeriesDecomposition
from ..sa.base import SaResults
from ..sa.diagnostics.quality import QualityMeasures, SlidingSpansStability
from .plots import TimeSeriesPlotter, DecompositionPlotter, DiagnosticPlotter
from .styles import PlotStyle


def plot_series(series: Union[TsData, List[TsData]], 
                labels: Optional[List[str]] = None,
                title: Optional[str] = None,
                style: Optional[PlotStyle] = None,
                **kwargs) -> Figure:
    """Plot one or more time series.
    
    Args:
        series: Time series data
        labels: Series labels
        title: Plot title
        style: Plot style
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure
    """
    plotter = TimeSeriesPlotter(style)
    fig, _ = plotter.plot(series, labels=labels, title=title, **kwargs)
    return fig


def plot_decomposition(decomposition: Union[SeriesDecomposition, SaResults],
                      title: Optional[str] = None,
                      style: Optional[PlotStyle] = None,
                      **kwargs) -> Figure:
    """Plot seasonal decomposition.
    
    Args:
        decomposition: Decomposition results or SA results
        title: Plot title
        style: Plot style
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    plotter = DecompositionPlotter(style)
    
    if isinstance(decomposition, SaResults):
        decomposition = decomposition.decomposition
    
    return plotter.plot(decomposition, title=title, **kwargs)


def plot_sa_comparison(original: TsData, 
                      sa_results: Union[TsData, SaResults, List[SaResults]],
                      labels: Optional[List[str]] = None,
                      title: Optional[str] = None,
                      style: Optional[PlotStyle] = None,
                      **kwargs) -> Figure:
    """Compare original series with SA results.
    
    Args:
        original: Original series
        sa_results: SA series or results
        labels: Labels for series
        title: Plot title
        style: Plot style
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    plotter = TimeSeriesPlotter(style)
    
    # Prepare series list
    series_list = [original]
    
    if isinstance(sa_results, TsData):
        series_list.append(sa_results)
        if labels is None:
            labels = ["Original", "Seasonally Adjusted"]
    elif isinstance(sa_results, SaResults):
        series_list.append(sa_results.decomposition.seasonally_adjusted)
        if labels is None:
            labels = ["Original", "Seasonally Adjusted"]
    elif isinstance(sa_results, list):
        for result in sa_results:
            if isinstance(result, SaResults):
                series_list.append(result.decomposition.seasonally_adjusted)
            else:
                series_list.append(result)
    
    fig, _ = plotter.plot(series_list, labels=labels, 
                         title=title or "Original vs Seasonally Adjusted", **kwargs)
    return fig


def plot_spectrum(series: TsData,
                 method: str = "periodogram",
                 title: Optional[str] = None,
                 style: Optional[PlotStyle] = None,
                 **kwargs) -> Figure:
    """Plot spectrum of time series.
    
    Args:
        series: Time series
        method: Spectrum method ("periodogram" or "ar")
        title: Plot title
        style: Plot style
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    plotter = DiagnosticPlotter(style)
    
    if method == "periodogram":
        return plotter.plot_spectrum(series, title=title, **kwargs)
    elif method == "ar":
        from .plots import SpectrumPlotter
        spec_plotter = SpectrumPlotter(style)
        return spec_plotter.plot_ar_spectrum(series, title=title, **kwargs)
    else:
        raise ValueError(f"Unknown spectrum method: {method}")


def plot_acf(series: TsData,
            max_lag: int = 40,
            title: Optional[str] = None,
            style: Optional[PlotStyle] = None,
            **kwargs) -> Figure:
    """Plot ACF and PACF.
    
    Args:
        series: Time series
        max_lag: Maximum lag
        title: Plot title
        style: Plot style
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    plotter = DiagnosticPlotter(style)
    return plotter.plot_acf(series, max_lag=max_lag, title=title, **kwargs)


def plot_diagnostics(sa_results: SaResults,
                    include: Optional[List[str]] = None,
                    style: Optional[PlotStyle] = None) -> Figure:
    """Plot comprehensive diagnostics.
    
    Args:
        sa_results: SA results
        include: Which diagnostics to include
        style: Plot style
        
    Returns:
        Figure
    """
    if include is None:
        include = ["residuals", "spectrum", "seasonal"]
    
    n_plots = len(include)
    fig = plt.figure(figsize=(12, 4 * n_plots))
    
    plot_idx = 1
    
    # Residual diagnostics
    if "residuals" in include and hasattr(sa_results, 'residuals'):
        ax = plt.subplot(n_plots, 1, plot_idx)
        plotter = DiagnosticPlotter(style)
        plotter.plot_residuals(sa_results.residuals)
        plot_idx += 1
    
    # Spectrum
    if "spectrum" in include:
        ax = plt.subplot(n_plots, 1, plot_idx)
        plotter = DiagnosticPlotter(style)
        if sa_results.decomposition.irregular:
            plotter.plot_spectrum(sa_results.decomposition.irregular,
                                mark_seasonal=True)
        plot_idx += 1
    
    # Seasonal factors
    if "seasonal" in include and sa_results.decomposition.seasonal:
        ax = plt.subplot(n_plots, 1, plot_idx)
        plotter = DecompositionPlotter(style)
        plotter.plot_seasonal_factors(sa_results.decomposition)
        plot_idx += 1
    
    plt.tight_layout()
    return fig


def plot_sliding_spans(series: TsData,
                      stability_results: Dict[str, float],
                      title: Optional[str] = None,
                      style: Optional[PlotStyle] = None) -> Figure:
    """Plot sliding spans stability analysis.
    
    Args:
        series: Original series
        stability_results: Stability percentages by component
        title: Plot title
        style: Plot style
        
    Returns:
        Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot series
    plotter = TimeSeriesPlotter(style)
    plotter.plot(series, labels=["Original"], ax=ax1)
    ax1.set_title("Original Series")
    
    # Plot stability bars
    components = list(stability_results.keys())
    values = list(stability_results.values())
    
    bars = ax2.bar(components, values)
    
    # Color bars based on stability
    for bar, val in zip(bars, values):
        if val >= 90:
            bar.set_color('green')
        elif val >= 70:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax2.set_ylabel("Stability %")
    ax2.set_title("Sliding Spans Stability")
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_revision_history(current: TsData,
                         previous: List[TsData],
                         labels: Optional[List[str]] = None,
                         title: Optional[str] = None,
                         style: Optional[PlotStyle] = None) -> Figure:
    """Plot revision history.
    
    Args:
        current: Current estimate
        previous: Previous estimates
        labels: Labels for estimates
        title: Plot title
        style: Plot style
        
    Returns:
        Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    plotter = TimeSeriesPlotter(style)
    
    # Plot all estimates
    all_series = [current] + previous
    if labels is None:
        labels = ["Current"] + [f"Previous {i+1}" for i in range(len(previous))]
    
    plotter.plot(all_series, labels=labels, ax=ax1)
    ax1.set_title("Estimates")
    ax1.set_xlabel("")
    
    # Plot revisions
    dates = plotter._get_dates(current)
    
    for i, prev in enumerate(previous):
        # Compute revision
        revision = current.values - prev.values
        ax2.plot(dates, revision, label=f"Revision from {labels[i+1]}")
    
    ax2.set_title("Revisions")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Revision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_quality_report(quality: QualityMeasures,
                       title: Optional[str] = None,
                       style: Optional[PlotStyle] = None) -> Figure:
    """Plot quality measures report.
    
    Args:
        quality: Quality measures
        title: Plot title
        style: Plot style
        
    Returns:
        Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # M-statistics
    if quality.m_statistics:
        ax = axes[0, 0]
        stats = list(quality.m_statistics.keys())
        values = list(quality.m_statistics.values())
        
        bars = ax.bar(stats, values)
        
        # Color based on thresholds
        for bar, stat, val in zip(bars, stats, values):
            if val < 1:
                bar.set_color('green')
            elif val < 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_title("M-Statistics")
        ax.set_ylabel("Value")
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=3, color='red', linestyle='--', alpha=0.5)
    
    # Overall quality gauge
    ax = axes[0, 1]
    if quality.q_statistic is not None:
        # Simple gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Color zones
        ax.fill_between(x[:33], 0, y[:33], color='green', alpha=0.3)
        ax.fill_between(x[33:66], 0, y[33:66], color='orange', alpha=0.3)
        ax.fill_between(x[66:], 0, y[66:], color='red', alpha=0.3)
        
        # Needle
        q_angle = np.pi * (1 - quality.q_statistic / 3)
        ax.plot([0, np.cos(q_angle)], [0, np.sin(q_angle)], 'k-', linewidth=3)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Overall Quality (Q={quality.q_statistic:.2f})")
    
    # Text summary
    ax = axes[1, 0]
    ax.axis('off')
    
    summary_text = f"Overall Quality: {quality.overall_quality or 'Unknown'}\n\n"
    
    if quality.q_statistic is not None:
        summary_text += f"Q-Statistic: {quality.q_statistic:.3f}\n"
    if quality.sliding_spans_percentage is not None:
        summary_text += f"Sliding Spans Stability: {quality.sliding_spans_percentage:.1f}%\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top')
    
    # Spectral diagnostics
    if quality.spectral_diagnostics:
        ax = axes[1, 1]
        spec = quality.spectral_diagnostics
        
        # Simple bar chart
        checks = ["Residual\nSeasonality", "Residual\nTrading Days"]
        values = [1 if spec.residual_seasonality else 0,
                 1 if spec.residual_td else 0]
        colors = ['red' if v else 'green' for v in values]
        
        ax.bar(checks, values, color=colors, alpha=0.7)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Present")
        ax.set_title("Spectral Checks")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig