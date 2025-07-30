"""Seasonal adjustment diagnostics."""

from .seasonality import (
    SeasonalityTest,
    FriedmanTest,
    KruskalWallisTest,
    CombinedSeasonalityTest,
    ResidualSeasonalityTest,
    SeasonalityTests
)
from .quality import (
    QualityMeasures,
    MStatistics,
    SlidingSpansStability,
    RevisionHistory
)
from .residuals import (
    ResidualsDiagnostics,
    NormalityTests,
    IndependenceTests,
    StationarityTests
)

def compute_comprehensive_quality(results):
    """Compute comprehensive quality measures."""
    # Placeholder implementation
    return QualityMeasures()

__all__ = [
    # Seasonality
    "SeasonalityTest",
    "FriedmanTest",
    "KruskalWallisTest", 
    "CombinedSeasonalityTest",
    "ResidualSeasonalityTest",
    "SeasonalityTests",
    # Quality
    "QualityMeasures",
    "MStatistics",
    "SlidingSpansStability",
    "RevisionHistory",
    # Residuals
    "ResidualsDiagnostics",
    "NormalityTests",
    "IndependenceTests",
    "StationarityTests",
    # Functions
    "compute_comprehensive_quality",
]