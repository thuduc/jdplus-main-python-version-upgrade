"""Quality measures for seasonal adjustment."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

from ...toolkit.timeseries import TsData
from ...toolkit.stats.tests import TestResult


@dataclass
class QualityMeasures:
    """Container for SA quality measures."""
    
    m_statistics: Optional[Dict[str, float]] = None
    q_statistic: Optional[float] = None
    q_statistic_sa: Optional[float] = None
    
    # Sliding spans
    sliding_spans_stable: Optional[bool] = None
    sliding_spans_percentage: Optional[float] = None
    
    # Revisions
    revision_mean: Optional[float] = None
    revision_std: Optional[float] = None
    
    # Overall assessment
    overall_quality: Optional[str] = None  # "Good", "Uncertain", "Bad"
    
    # Additional diagnostics
    spectral_diagnostics: Optional['SpectralDiagnostics'] = None
    residual_diagnostics: Optional[Dict[str, Any]] = None
    
    def summary(self) -> str:
        """Get quality summary."""
        lines = ["Quality Measures:"]
        
        if self.m_statistics:
            lines.append("\nM-Statistics:")
            for stat, value in sorted(self.m_statistics.items()):
                lines.append(f"  {stat}: {value:.3f}")
        
        if self.q_statistic is not None:
            lines.append(f"\nQ-Statistic: {self.q_statistic:.3f}")
        
        if self.sliding_spans_percentage is not None:
            lines.append(f"\nSliding Spans Stability: {self.sliding_spans_percentage:.1f}%")
        
        if self.overall_quality:
            lines.append(f"\nOverall Quality: {self.overall_quality}")
        
        return "\n".join(lines)


class MStatistics:
    """X-11 M-statistics for quality assessment."""
    
    # Thresholds for M-statistics
    THRESHOLDS = {
        "M1": 1.0,   # Relative contribution of irregular
        "M2": 1.0,   # Relative contribution of irregular (3x3 MA)
        "M3": 1.0,   # Irregular vs Trend-Cycle ratio
        "M4": 1.0,   # Randomness of irregular
        "M5": 1.0,   # Statistical significance of seasonality
        "M6": 1.0,   # Seasonality evolution
        "M7": 0.5,   # Combined test for identifiable seasonality
        "M8": 1.0,   # Fluctuations in seasonal component
        "M9": 1.0,   # Linear movement in seasonal component
        "M10": 1.0,  # Same month/quarter stability
        "M11": 1.0,  # Same month/quarter evolution
        "Q": 1.0,    # Overall Q statistic
    }
    
    @classmethod
    def compute(cls, decomposition: 'SeriesDecomposition') -> Dict[str, float]:
        """Compute M-statistics from decomposition.
        
        Args:
            decomposition: Series decomposition
            
        Returns:
            Dictionary of M-statistics
        """
        m_stats = {}
        
        # Extract components
        irregular = decomposition.irregular
        seasonal = decomposition.seasonal
        sa = decomposition.seasonally_adjusted
        
        if irregular is None or seasonal is None or sa is None:
            return m_stats
        
        # M1: Relative contribution of irregular
        m1 = cls._compute_m1(irregular, sa)
        if m1 is not None:
            m_stats["M1"] = m1
        
        # M3: Irregular vs Trend-Cycle ratio
        if decomposition.trend is not None:
            m3 = cls._compute_m3(irregular, decomposition.trend)
            if m3 is not None:
                m_stats["M3"] = m3
        
        # M7: Combined test for identifiable seasonality
        m7 = cls._compute_m7(seasonal, irregular)
        if m7 is not None:
            m_stats["M7"] = m7
        
        # Q: Overall quality statistic (weighted average)
        if len(m_stats) >= 3:
            q = cls._compute_q_statistic(m_stats)
            m_stats["Q"] = q
        
        return m_stats
    
    @staticmethod
    def _compute_m1(irregular: TsData, sa: TsData) -> Optional[float]:
        """M1: Relative contribution of irregular to SA."""
        try:
            # Compute variances of changes
            irr_diff = np.diff(irregular.values)
            sa_diff = np.diff(sa.values)
            
            # Remove NaN
            irr_diff = irr_diff[~np.isnan(irr_diff)]
            sa_diff = sa_diff[~np.isnan(sa_diff)]
            
            if len(irr_diff) == 0 or len(sa_diff) == 0:
                return None
            
            var_irr = np.var(irr_diff)
            var_sa = np.var(sa_diff)
            
            if var_sa == 0:
                return None
            
            return 10 * var_irr / var_sa
        except:
            return None
    
    @staticmethod
    def _compute_m3(irregular: TsData, trend: TsData) -> Optional[float]:
        """M3: Ratio of irregular to trend changes."""
        try:
            # Month-to-month changes
            irr_abs_change = np.abs(np.diff(irregular.values))
            trend_abs_change = np.abs(np.diff(trend.values))
            
            # Remove NaN
            mask = ~(np.isnan(irr_abs_change) | np.isnan(trend_abs_change))
            irr_abs_change = irr_abs_change[mask]
            trend_abs_change = trend_abs_change[mask]
            
            if len(irr_abs_change) == 0:
                return None
            
            # Average absolute changes
            avg_irr = np.mean(irr_abs_change)
            avg_trend = np.mean(trend_abs_change)
            
            if avg_trend == 0:
                return None
            
            return avg_irr / avg_trend
        except:
            return None
    
    @staticmethod
    def _compute_m7(seasonal: TsData, irregular: TsData) -> Optional[float]:
        """M7: Combined test for identifiable seasonality."""
        try:
            # Simplified version - ratio of seasonal to irregular variance
            seasonal_var = np.var(seasonal.values[~np.isnan(seasonal.values)])
            irregular_var = np.var(irregular.values[~np.isnan(irregular.values)])
            
            if irregular_var == 0:
                return None
            
            # F-ratio
            f_ratio = seasonal_var / irregular_var
            
            # Convert to M7 scale (0-3)
            if f_ratio > 3.0:
                return 0.0  # Very good
            elif f_ratio > 1.0:
                return 1.0 - (f_ratio - 1.0) / 2.0
            else:
                return 3.0  # Bad
        except:
            return None
    
    @staticmethod
    def _compute_q_statistic(m_stats: Dict[str, float]) -> float:
        """Compute overall Q statistic."""
        # Weights for different M statistics
        weights = {
            "M1": 0.10,
            "M2": 0.10,
            "M3": 0.10,
            "M4": 0.05,
            "M5": 0.05,
            "M6": 0.10,
            "M7": 0.20,
            "M8": 0.10,
            "M9": 0.05,
            "M10": 0.05,
            "M11": 0.10,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for stat, value in m_stats.items():
            if stat in weights and stat != "Q":
                weight = weights[stat]
                weighted_sum += weight * min(value, 3.0)  # Cap at 3
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.nan


class SlidingSpansStability:
    """Sliding spans analysis for stability assessment."""
    
    def __init__(self, n_spans: int = 4, span_length: Optional[int] = None):
        """Initialize sliding spans.
        
        Args:
            n_spans: Number of spans
            span_length: Length of each span (auto if None)
        """
        self.n_spans = n_spans
        self.span_length = span_length
    
    def analyze(self, data: TsData, processor: 'SaProcessor') -> Tuple[float, Dict[str, float]]:
        """Perform sliding spans analysis.
        
        Args:
            data: Original series
            processor: SA processor
            
        Returns:
            Percentage of stable periods and detailed results
        """
        n = data.length
        
        # Determine span length
        if self.span_length is None:
            # Use 8-10 years if possible
            target_years = 8
            target_length = target_years * data.domain.frequency.periods_per_year
            self.span_length = min(target_length, n - self.n_spans + 1)
        
        # Generate spans
        span_results = []
        for i in range(self.n_spans):
            start_idx = i
            end_idx = start_idx + self.span_length
            
            # Extract span
            span_data = TsData.of(
                data.domain.get(start_idx),
                data.values[start_idx:end_idx]
            )
            
            # Process span
            try:
                result = processor.process(span_data)
                span_results.append(result.decomposition.seasonally_adjusted)
            except:
                span_results.append(None)
        
        # Compare results
        stable_periods = self._compute_stability(span_results)
        
        return stable_periods, {"n_spans": self.n_spans, "span_length": self.span_length}
    
    def _compute_stability(self, span_results: List[Optional[TsData]]) -> float:
        """Compute percentage of stable periods.
        
        Args:
            span_results: SA results from different spans
            
        Returns:
            Percentage of stable periods
        """
        # Filter valid results
        valid_results = [r for r in span_results if r is not None]
        
        if len(valid_results) < 2:
            return 0.0
        
        # Find common period
        common_start = max(r.start.epoch_period for r in valid_results)
        common_end = min(r.domain.end.epoch_period for r in valid_results)
        common_length = common_end - common_start
        
        if common_length <= 0:
            return 0.0
        
        # Extract common values
        common_values = []
        for result in valid_results:
            start_idx = common_start - result.start.epoch_period
            end_idx = start_idx + common_length
            common_values.append(result.values[start_idx:end_idx])
        
        common_values = np.array(common_values)
        
        # Compute stability (max relative difference < 3%)
        stable_count = 0
        for t in range(common_length):
            values_t = common_values[:, t]
            if not np.any(np.isnan(values_t)):
                max_val = np.max(values_t)
                min_val = np.min(values_t)
                if max_val != 0:
                    rel_diff = (max_val - min_val) / np.abs(max_val)
                    if rel_diff < 0.03:  # 3% threshold
                        stable_count += 1
        
        return 100.0 * stable_count / common_length


class RevisionHistory:
    """Analysis of revision history."""
    
    @staticmethod
    def compute_revisions(current: TsData, previous: TsData) -> TsData:
        """Compute revisions between two vintages.
        
        Args:
            current: Current estimates
            previous: Previous estimates
            
        Returns:
            Revision series
        """
        # Find common period
        common_domain = current.domain.intersection(previous.domain)
        
        if common_domain is None:
            raise ValueError("No common period for revision analysis")
        
        # Extract common values
        current_common = current.select(common_domain)
        previous_common = previous.select(common_domain)
        
        # Compute revisions
        revisions = current_common.values - previous_common.values
        
        return TsData.of(common_domain.start, revisions)
    
    @staticmethod
    def revision_statistics(revisions: TsData) -> Dict[str, float]:
        """Compute revision statistics.
        
        Args:
            revisions: Revision series
            
        Returns:
            Dictionary of statistics
        """
        values = revisions.values[~np.isnan(revisions.values)]
        
        if len(values) == 0:
            return {}
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "mae": np.mean(np.abs(values)),
            "rmse": np.sqrt(np.mean(values**2)),
            "max": np.max(np.abs(values)),
            "relative_mae": np.mean(np.abs(values)) / np.mean(np.abs(values))
        }


@dataclass
class SpectralDiagnostics:
    """Spectral diagnostics for seasonal adjustment quality."""
    
    # Seasonal frequencies
    seasonal_frequencies: List[float]
    seasonal_peaks: List[float]
    seasonal_significance: List[bool]
    
    # Trading day frequencies
    td_frequencies: List[float]
    td_peaks: List[float]
    td_significance: List[bool]
    
    # Residual seasonality/TD
    residual_seasonality: bool
    residual_td: bool
    
    # Overall spectral quality
    spectral_quality: str  # "Good", "Acceptable", "Poor"


class SpectralAnalyzer:
    """Spectral analysis for SA diagnostics."""
    
    @staticmethod
    def analyze_component(series: TsData, component_type: str = "sa") -> SpectralDiagnostics:
        """Analyze spectral properties of a component.
        
        Args:
            series: Time series component
            component_type: Type of component ("sa", "irregular", etc.)
            
        Returns:
            Spectral diagnostics
        """
        from scipy.signal import periodogram
        
        # Remove missing values
        values = series.values[~np.isnan(series.values)]
        if len(values) < 24:
            return None
        
        # Compute periodogram
        freqs, psd = periodogram(values, fs=series.domain.frequency.periods_per_year)
        
        # Identify seasonal frequencies
        period = series.domain.frequency.periods_per_year
        seasonal_freqs = [k / period for k in range(1, period // 2 + 1)]
        
        # Trading day frequencies (if monthly)
        td_freqs = []
        if period == 12:
            td_freqs = [0.348, 0.432]  # Common TD frequencies
        
        # Extract peaks at seasonal frequencies
        seasonal_peaks = []
        seasonal_sig = []
        
        for sf in seasonal_freqs:
            idx = np.argmin(np.abs(freqs - sf))
            peak = psd[idx]
            seasonal_peaks.append(peak)
            
            # Simple significance test (compare to neighboring frequencies)
            window = 5
            start = max(0, idx - window)
            end = min(len(psd), idx + window + 1)
            local_mean = np.mean(psd[start:end])
            seasonal_sig.append(peak > 2 * local_mean)
        
        # Extract TD peaks
        td_peaks = []
        td_sig = []
        
        for tf in td_freqs:
            idx = np.argmin(np.abs(freqs - tf))
            peak = psd[idx]
            td_peaks.append(peak)
            
            # Significance test
            window = 5
            start = max(0, idx - window)
            end = min(len(psd), idx + window + 1)
            local_mean = np.mean(psd[start:end])
            td_sig.append(peak > 2 * local_mean)
        
        # Overall assessment
        residual_seas = any(seasonal_sig) if component_type in ["sa", "irregular"] else False
        residual_td = any(td_sig) if component_type in ["sa", "irregular"] else False
        
        # Quality rating
        if residual_seas or residual_td:
            quality = "Poor"
        elif max(seasonal_peaks) > 1.5 * np.median(psd):
            quality = "Acceptable"
        else:
            quality = "Good"
        
        return SpectralDiagnostics(
            seasonal_frequencies=seasonal_freqs,
            seasonal_peaks=seasonal_peaks,
            seasonal_significance=seasonal_sig,
            td_frequencies=td_freqs,
            td_peaks=td_peaks,
            td_significance=td_sig,
            residual_seasonality=residual_seas,
            residual_td=residual_td,
            spectral_quality=quality
        )


def compute_comprehensive_quality(decomposition: 'SeriesDecomposition',
                                sa_results=None) -> QualityMeasures:
    """Compute comprehensive quality measures.
    
    Args:
        decomposition: Series decomposition
        sa_results: Full SA results (optional)
        
    Returns:
        Quality measures
    """
    measures = QualityMeasures()
    
    # M-statistics
    measures.m_statistics = MStatistics.compute(decomposition)
    if "Q" in measures.m_statistics:
        measures.q_statistic = measures.m_statistics["Q"]
    
    # Spectral diagnostics
    if decomposition.seasonally_adjusted is not None:
        try:
            measures.spectral_diagnostics = SpectralAnalyzer.analyze_component(
                decomposition.seasonally_adjusted, "sa"
            )
        except:
            pass
    
    # Overall quality assessment
    if measures.q_statistic is not None:
        if measures.q_statistic < 1.0:
            measures.overall_quality = "Good"
        elif measures.q_statistic < 2.0:
            measures.overall_quality = "Uncertain"
        else:
            measures.overall_quality = "Bad"
    
    # Check spectral quality
    if measures.spectral_diagnostics:
        if measures.spectral_diagnostics.residual_seasonality:
            measures.overall_quality = "Bad"
    
    return measures