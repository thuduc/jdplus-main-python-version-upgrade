"""Plot styling for visualization."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotStyle:
    """Plot style configuration."""
    
    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    accent_color: str = "#2ca02c"
    
    # Color palette
    palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Figure style
    style: str = "whitegrid"
    context: str = "notebook"
    
    # Font sizes
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    
    # Line properties
    line_width: float = 1.5
    marker_size: float = 6
    
    # Grid properties
    grid_alpha: float = 0.3
    grid_style: str = "-"
    
    # Figure properties
    fig_dpi: int = 100
    fig_facecolor: str = "white"
    
    def apply(self):
        """Apply style settings."""
        # Set seaborn style
        sns.set_style(self.style)
        sns.set_context(self.context)
        
        # Set matplotlib parameters
        plt.rcParams.update({
            # Figure
            'figure.dpi': self.fig_dpi,
            'figure.facecolor': self.fig_facecolor,
            
            # Font sizes
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            
            # Lines
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            
            # Grid
            'axes.grid': True,
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_style,
            
            # Colors
            'axes.prop_cycle': plt.cycler(color=self.palette),
        })
    
    @classmethod
    def default(cls) -> 'PlotStyle':
        """Default JDemetra+ style."""
        return cls()
    
    @classmethod
    def minimal(cls) -> 'PlotStyle':
        """Minimal style."""
        return cls(
            style="white",
            grid_alpha=0.2,
            palette=["#333333", "#666666", "#999999"]
        )
    
    @classmethod
    def colorful(cls) -> 'PlotStyle':
        """Colorful style."""
        return cls(
            palette=sns.color_palette("husl", 10).as_hex()
        )
    
    @classmethod
    def publication(cls) -> 'PlotStyle':
        """Publication-ready style."""
        return cls(
            style="white",
            fig_dpi=300,
            line_width=2.0,
            title_size=16,
            label_size=14,
            tick_size=12,
            legend_size=12
        )


def set_plot_style(style: Optional[PlotStyle] = None):
    """Set global plot style.
    
    Args:
        style: Plot style to apply (uses default if None)
    """
    if style is None:
        style = PlotStyle.default()
    
    style.apply()


def get_color_palette(n_colors: int = 10, 
                     palette_name: Optional[str] = None) -> List[str]:
    """Get color palette.
    
    Args:
        n_colors: Number of colors
        palette_name: Palette name (uses default if None)
        
    Returns:
        List of color codes
    """
    if palette_name:
        return sns.color_palette(palette_name, n_colors).as_hex()
    else:
        style = PlotStyle.default()
        if n_colors <= len(style.palette):
            return style.palette[:n_colors]
        else:
            # Generate more colors
            return sns.color_palette("husl", n_colors).as_hex()


def create_figure(n_subplots: int = 1,
                 layout: Optional[Tuple[int, int]] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 style: Optional[PlotStyle] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create figure with style applied.
    
    Args:
        n_subplots: Number of subplots
        layout: Subplot layout (rows, cols)
        figsize: Figure size
        style: Plot style
        
    Returns:
        Figure and axes
    """
    # Apply style
    set_plot_style(style)
    
    # Determine layout
    if layout is None:
        if n_subplots <= 1:
            layout = (1, 1)
        elif n_subplots <= 2:
            layout = (2, 1)
        elif n_subplots <= 4:
            layout = (2, 2)
        elif n_subplots <= 6:
            layout = (3, 2)
        else:
            layout = (4, 2)
    
    # Default figure size
    if figsize is None:
        figsize = (12, 4 * layout[0])
    
    # Create figure
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
    
    # Flatten axes if needed
    if n_subplots == 1:
        axes = axes if isinstance(axes, plt.Axes) else axes.flat[0]
    
    return fig, axes


# Color schemes for specific purposes

class ColorSchemes:
    """Predefined color schemes."""
    
    # Decomposition components
    DECOMPOSITION = {
        "original": "#1f77b4",
        "seasonally_adjusted": "#ff7f0e",
        "trend": "#2ca02c",
        "seasonal": "#d62728",
        "irregular": "#9467bd",
        "outliers": "#e377c2",
        "trading_days": "#7f7f7f",
        "easter": "#bcbd22"
    }
    
    # Quality indicators
    QUALITY = {
        "good": "#2ca02c",
        "acceptable": "#ff7f0e",
        "poor": "#d62728",
        "uncertain": "#7f7f7f"
    }
    
    # Residual diagnostics
    RESIDUALS = {
        "residuals": "#1f77b4",
        "fitted": "#ff7f0e",
        "normal": "#d62728",
        "confidence": "#7f7f7f"
    }
    
    # Spectrum
    SPECTRUM = {
        "spectrum": "#1f77b4",
        "seasonal_freq": "#d62728",
        "trading_day_freq": "#2ca02c",
        "noise": "#7f7f7f"
    }
    
    @classmethod
    def get_decomposition_colors(cls) -> Dict[str, str]:
        """Get decomposition color scheme."""
        return cls.DECOMPOSITION.copy()
    
    @classmethod
    def get_quality_colors(cls) -> Dict[str, str]:
        """Get quality color scheme."""
        return cls.QUALITY.copy()
    
    @classmethod
    def get_residual_colors(cls) -> Dict[str, str]:
        """Get residual diagnostic color scheme."""
        return cls.RESIDUALS.copy()
    
    @classmethod
    def get_spectrum_colors(cls) -> Dict[str, str]:
        """Get spectrum color scheme."""
        return cls.SPECTRUM.copy()