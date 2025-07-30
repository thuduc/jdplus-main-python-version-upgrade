"""Visualization tools for time series and seasonal adjustment."""

from .plots import (
    TimeSeriesPlotter,
    DecompositionPlotter,
    DiagnosticPlotter,
    SpectrumPlotter
)
from .charts import (
    plot_series,
    plot_decomposition,
    plot_sa_comparison,
    plot_spectrum,
    plot_acf,
    plot_diagnostics,
    plot_sliding_spans,
    plot_revision_history
)
from .styles import (
    PlotStyle,
    set_plot_style,
    get_color_palette
)

__all__ = [
    # Plotters
    "TimeSeriesPlotter",
    "DecompositionPlotter",
    "DiagnosticPlotter",
    "SpectrumPlotter",
    # Chart functions
    "plot_series",
    "plot_decomposition",
    "plot_sa_comparison",
    "plot_spectrum",
    "plot_acf",
    "plot_diagnostics",
    "plot_sliding_spans",
    "plot_revision_history",
    # Styles
    "PlotStyle",
    "set_plot_style",
    "get_color_palette",
]