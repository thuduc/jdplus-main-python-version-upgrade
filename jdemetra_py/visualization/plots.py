"""Plotting classes for time series visualization."""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from ..toolkit.timeseries import TsData
from ..sa.base.results import SeriesDecomposition
from .styles import PlotStyle, set_plot_style


class TimeSeriesPlotter:
    """Plotter for time series data."""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize plotter.
        
        Args:
            style: Plot style to use
        """
        self.style = style or PlotStyle()
        set_plot_style(self.style)
    
    def plot(self, series: Union[TsData, List[TsData]], 
             labels: Optional[List[str]] = None,
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (12, 6),
             ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
        """Plot time series.
        
        Args:
            series: Single series or list of series
            labels: Series labels
            title: Plot title
            figsize: Figure size
            ax: Existing axes to plot on
            
        Returns:
            Figure and axes
        """
        # Convert single series to list
        if isinstance(series, TsData):
            series = [series]
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot each series
        for i, ts in enumerate(series):
            dates = self._get_dates(ts)
            label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
            
            ax.plot(dates, ts.values, label=label, linewidth=1.5)
        
        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Format x-axis
        self._format_date_axis(ax, series[0].frequency)
        
        # Legend
        if len(series) > 1:
            ax.legend(loc='best')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_with_bands(self, series: TsData,
                       lower: Optional[TsData] = None,
                       upper: Optional[TsData] = None,
                       title: Optional[str] = None,
                       label: str = "Series",
                       band_label: str = "Confidence Band",
                       figsize: Tuple[float, float] = (12, 6),
                       ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
        """Plot series with confidence bands.
        
        Args:
            series: Main series
            lower: Lower band
            upper: Upper band
            title: Plot title
            label: Series label
            band_label: Band label
            figsize: Figure size
            ax: Existing axes
            
        Returns:
            Figure and axes
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        dates = self._get_dates(series)
        
        # Plot main series
        ax.plot(dates, series.values, label=label, linewidth=2, zorder=3)
        
        # Plot bands if provided
        if lower is not None and upper is not None:
            ax.fill_between(dates, lower.values, upper.values,
                          alpha=0.2, label=band_label, zorder=1)
        
        # Formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Format x-axis
        self._format_date_axis(ax, series.frequency)
        
        # Legend
        ax.legend(loc='best')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def _get_dates(self, series: TsData) -> pd.DatetimeIndex:
        """Get date index for series."""
        from ..toolkit.timeseries import TsFrequency
        
        start = series.start
        freq_map = {
            TsFrequency.MONTHLY: 'MS',
            TsFrequency.QUARTERLY: 'QS',
            TsFrequency.YEARLY: 'YS',
            TsFrequency.DAILY: 'D'
        }
        
        if series.frequency in freq_map:
            # Convert start period to timestamp
            if series.frequency == TsFrequency.MONTHLY:
                start_date = pd.Timestamp(year=start.year, month=start.position + 1, day=1)
            elif series.frequency == TsFrequency.QUARTERLY:
                month = start.position * 3 + 1
                start_date = pd.Timestamp(year=start.year, month=month, day=1)
            elif series.frequency == TsFrequency.YEARLY:
                start_date = pd.Timestamp(year=start.year, month=1, day=1)
            else:
                # Daily - use epoch
                start_date = pd.Timestamp('1970-01-01') + pd.Timedelta(days=start.epoch_period)
            
            # Generate date range
            return pd.date_range(
                start=start_date,
                periods=series.length,
                freq=freq_map[series.frequency]
            )
        else:
            # Unknown frequency - use integer index
            return pd.RangeIndex(start=0, stop=series.length)
    
    def _format_date_axis(self, ax: Axes, frequency):
        """Format date axis based on frequency."""
        from ..toolkit.timeseries import TsFrequency
        
        if frequency == TsFrequency.MONTHLY:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        elif frequency == TsFrequency.QUARTERLY:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif frequency == TsFrequency.YEARLY:
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif frequency == TsFrequency.DAILY:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # Rotate labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


class DecompositionPlotter:
    """Plotter for seasonal decomposition results."""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize plotter.
        
        Args:
            style: Plot style to use
        """
        self.style = style or PlotStyle()
        self.ts_plotter = TimeSeriesPlotter(style)
        set_plot_style(self.style)
    
    def plot(self, decomposition: SeriesDecomposition,
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (12, 10)) -> Figure:
        """Plot decomposition components.
        
        Args:
            decomposition: Decomposition results
            title: Overall title
            figsize: Figure size
            
        Returns:
            Figure
        """
        # Count components
        components = []
        if decomposition.series is not None:
            components.append(("Original", decomposition.series))
        if decomposition.seasonally_adjusted is not None:
            components.append(("Seasonally Adjusted", decomposition.seasonally_adjusted))
        if decomposition.trend is not None:
            components.append(("Trend", decomposition.trend))
        if decomposition.seasonal is not None:
            components.append(("Seasonal", decomposition.seasonal))
        if decomposition.irregular is not None:
            components.append(("Irregular", decomposition.irregular))
        
        n_plots = len(components)
        
        # Create subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        # Plot each component
        for i, (name, component) in enumerate(components):
            self.ts_plotter.plot(component, labels=[name], ax=axes[i])
            axes[i].set_title(name, fontsize=12)
            axes[i].set_xlabel("")  # Remove x-label except for bottom
        
        # Add x-label to bottom plot
        axes[-1].set_xlabel("Date")
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def plot_comparison(self, original: TsData, 
                       seasonally_adjusted: TsData,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (12, 6)) -> Figure:
        """Plot original vs seasonally adjusted.
        
        Args:
            original: Original series
            seasonally_adjusted: SA series
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure
        """
        fig, ax = self.ts_plotter.plot(
            [original, seasonally_adjusted],
            labels=["Original", "Seasonally Adjusted"],
            title=title or "Original vs Seasonally Adjusted",
            figsize=figsize
        )
        
        return fig
    
    def plot_seasonal_factors(self, decomposition: SeriesDecomposition,
                            title: Optional[str] = None,
                            figsize: Tuple[float, float] = (12, 6)) -> Figure:
        """Plot seasonal factors by period.
        
        Args:
            decomposition: Decomposition results
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure
        """
        if decomposition.seasonal is None:
            raise ValueError("No seasonal component in decomposition")
        
        # Extract seasonal factors
        seasonal = decomposition.seasonal
        period = seasonal.frequency.periods_per_year
        
        # Reshape into matrix (years x periods)
        n_years = seasonal.length // period
        seasonal_matrix = seasonal.values[:n_years * period].reshape(n_years, period)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot by period
        ax1.boxplot(seasonal_matrix.T)
        ax1.set_xlabel("Period")
        ax1.set_ylabel("Seasonal Factor")
        ax1.set_title("Seasonal Factors by Period")
        ax1.grid(True, alpha=0.3)
        
        # Line plot by year
        for i in range(min(5, n_years)):  # Show first 5 years
            ax2.plot(range(period), seasonal_matrix[i, :], 
                    label=f"Year {i+1}", alpha=0.7)
        
        ax2.set_xlabel("Period")
        ax2.set_ylabel("Seasonal Factor")
        ax2.set_title("Seasonal Pattern Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig


class DiagnosticPlotter:
    """Plotter for diagnostic results."""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize plotter.
        
        Args:
            style: Plot style to use
        """
        self.style = style or PlotStyle()
        set_plot_style(self.style)
    
    def plot_acf(self, series: TsData, 
                 max_lag: int = 40,
                 title: Optional[str] = None,
                 figsize: Tuple[float, float] = (12, 6)) -> Figure:
        """Plot autocorrelation function.
        
        Args:
            series: Time series
            max_lag: Maximum lag
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ACF
        plot_acf(series.values[~np.isnan(series.values)], 
                lags=max_lag, ax=ax1, alpha=0.05)
        ax1.set_title("Autocorrelation Function (ACF)")
        
        # PACF
        plot_pacf(series.values[~np.isnan(series.values)], 
                 lags=max_lag, ax=ax2, alpha=0.05)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def plot_residuals(self, residuals: TsData,
                      title: Optional[str] = None,
                      figsize: Tuple[float, float] = (12, 8)) -> Figure:
        """Plot residual diagnostics.
        
        Args:
            residuals: Residual series
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        values = residuals.values[~np.isnan(residuals.values)]
        
        # Time plot
        axes[0, 0].plot(values)
        axes[0, 0].set_title("Residuals")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[0, 1].hist(values, bins=30, density=True, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Histogram")
        axes[0, 1].set_xlabel("Value")
        axes[0, 1].set_ylabel("Density")
        
        # Add normal curve
        from scipy import stats
        x = np.linspace(values.min(), values.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, values.mean(), values.std()), 
                       'r-', label='Normal')
        axes[0, 1].legend()
        
        # Q-Q plot
        stats.probplot(values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        
        # ACF
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(values, lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title("ACF of Residuals")
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    def plot_spectrum(self, series: TsData,
                     title: Optional[str] = None,
                     figsize: Tuple[float, float] = (10, 6),
                     mark_seasonal: bool = True) -> Figure:
        """Plot periodogram/spectrum.
        
        Args:
            series: Time series
            title: Plot title
            figsize: Figure size
            mark_seasonal: Mark seasonal frequencies
            
        Returns:
            Figure
        """
        from scipy.signal import periodogram
        
        # Remove missing values
        values = series.values[~np.isnan(series.values)]
        
        # Compute periodogram
        freqs, psd = periodogram(values, fs=series.frequency.periods_per_year)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectrum
        ax.semilogy(freqs, psd)
        ax.set_xlabel("Frequency (cycles per year)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title(title or "Periodogram")
        ax.grid(True, alpha=0.3)
        
        # Mark seasonal frequencies
        if mark_seasonal and series.frequency.periods_per_year > 1:
            period = series.frequency.periods_per_year
            for k in range(1, period // 2 + 1):
                freq = k
                ax.axvline(freq, color='red', linestyle='--', alpha=0.5,
                          label=f"Seasonal ({k}/{period})" if k == 1 else None)
            
            # Trading day frequencies for monthly
            if period == 12:
                ax.axvline(0.348, color='green', linestyle='--', alpha=0.5,
                          label="Trading Day (0.348)")
                ax.axvline(0.432, color='green', linestyle='--', alpha=0.5)
            
            ax.legend()
        
        plt.tight_layout()
        
        return fig


class SpectrumPlotter:
    """Specialized spectrum plotting."""
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize plotter.
        
        Args:
            style: Plot style to use
        """
        self.style = style or PlotStyle()
        self.diagnostic_plotter = DiagnosticPlotter(style)
    
    def plot_ar_spectrum(self, series: TsData,
                        ar_order: Optional[int] = None,
                        title: Optional[str] = None,
                        figsize: Tuple[float, float] = (10, 6)) -> Figure:
        """Plot AR spectrum estimate.
        
        Args:
            series: Time series
            ar_order: AR order (auto if None)
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure
        """
        from statsmodels.tsa.ar_model import AutoReg
        
        # Fit AR model
        values = series.values[~np.isnan(series.values)]
        
        if ar_order is None:
            # Use AIC to select order
            ar_order = AutoReg(values, lags=range(1, 21), old_names=False).select_order()
        
        model = AutoReg(values, lags=ar_order, old_names=False)
        results = model.fit()
        
        # Compute spectrum
        from statsmodels.tsa.stattools import arma_periodogram
        ar_params = np.r_[1, -results.params[1:]]  # Include unit root
        freqs, spectrum = arma_periodogram(ar_params, worN=512, whole=False)
        
        # Scale frequencies
        freqs = freqs * series.frequency.periods_per_year / (2 * np.pi)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.semilogy(freqs, spectrum)
        ax.set_xlabel("Frequency (cycles per year)")
        ax.set_ylabel("Spectrum")
        ax.set_title(title or f"AR({ar_order}) Spectrum")
        ax.grid(True, alpha=0.3)
        
        # Mark seasonal frequencies
        if series.frequency.periods_per_year > 1:
            period = series.frequency.periods_per_year
            for k in range(1, period // 2 + 1):
                freq = k
                ax.axvline(freq, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        return fig