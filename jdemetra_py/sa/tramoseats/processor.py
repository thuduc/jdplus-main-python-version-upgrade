"""TRAMO-SEATS processor."""

from typing import Optional
import logging

from ...toolkit.timeseries import TsData
from ..base import SaProcessor, SaResults
from .specification import TramoSeatsSpecification
from .tramo import TramoProcessor, TramoResults
from .seats import SeatsDecomposer, SeatsResults


class TramoSeatsProcessor(SaProcessor):
    """TRAMO-SEATS seasonal adjustment processor."""
    
    def __init__(self, specification: Optional[TramoSeatsSpecification] = None):
        """Initialize TRAMO-SEATS processor.
        
        Args:
            specification: TRAMO-SEATS specification (uses default if None)
        """
        if specification is None:
            specification = TramoSeatsSpecification.rsa1()  # Default automatic
        
        self.spec = specification
        self.tramo_processor = TramoProcessor(specification.tramo)
        self.seats_decomposer = SeatsDecomposer(specification.seats)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, series: TsData) -> 'TramoSeatsResults':
        """Process series with TRAMO-SEATS.
        
        Args:
            series: Input time series
            
        Returns:
            TRAMO-SEATS results
        """
        self.logger.info(f"Processing series with TRAMO-SEATS (length={series.length})")
        
        # Step 1: TRAMO pre-adjustment
        self.logger.info("Running TRAMO pre-adjustment...")
        tramo_results = self.tramo_processor.process(series)
        
        # Step 2: SEATS decomposition
        self.logger.info("Running SEATS decomposition...")
        seats_results = self.seats_decomposer.decompose(tramo_results)
        
        # Step 3: Create final decomposition
        decomposition = self.seats_decomposer.create_decomposition(
            seats_results, tramo_results
        )
        
        # Create results
        results = TramoSeatsResults(
            specification=self.spec,
            decomposition=decomposition,
            tramo_results=tramo_results,
            seats_results=seats_results
        )
        
        # Compute diagnostics
        results.compute_diagnostics()
        
        self.logger.info("TRAMO-SEATS processing complete")
        
        return results
    
    def get_name(self) -> str:
        """Get processor name."""
        return "TRAMO-SEATS"
    
    def get_version(self) -> str:
        """Get processor version."""
        return "1.0.0"


class TramoSeatsResults(SaResults):
    """Results from TRAMO-SEATS seasonal adjustment."""
    
    def __init__(self, specification: TramoSeatsSpecification,
                 decomposition, tramo_results: TramoResults,
                 seats_results: SeatsResults):
        """Initialize TRAMO-SEATS results.
        
        Args:
            specification: Specification used
            decomposition: Series decomposition
            tramo_results: TRAMO pre-adjustment results
            seats_results: SEATS decomposition results
        """
        super().__init__(specification, decomposition)
        self.tramo_results = tramo_results
        self.seats_results = seats_results
    
    def compute_diagnostics(self):
        """Compute diagnostics for TRAMO-SEATS results."""
        # Basic decomposition diagnostics
        super().compute_diagnostics()
        
        # Add TRAMO-specific diagnostics
        if self.tramo_results.diagnostics:
            self.diagnostics['tramo'] = self.tramo_results.diagnostics
        
        # Add SEATS-specific diagnostics
        if self.seats_results.diagnostics:
            self.diagnostics['seats'] = self.seats_results.diagnostics
        
        # Model information
        if self.tramo_results.arima_model is not None:
            self.diagnostics['model'] = {
                'order': str(self.tramo_results.arima_model.order),
                'n_params': self.tramo_results.arima_model.n_params()
            }
        
        # Outliers information
        if self.tramo_results.outliers:
            self.diagnostics['outliers'] = {
                'count': len(self.tramo_results.outliers),
                'types': [o.type.name for o in self.tramo_results.outliers]
            }
        
        # Component variances
        if self.seats_results.trend_variance is not None:
            self.diagnostics['component_variances'] = {
                'trend': self.seats_results.trend_variance,
                'seasonal': self.seats_results.seasonal_variance,
                'irregular': self.seats_results.irregular_variance
            }
    
    def summary(self) -> str:
        """Get summary of results."""
        lines = ["TRAMO-SEATS Results Summary"]
        lines.append("=" * 40)
        
        # Series info
        lines.append(f"\nSeries length: {self.decomposition.series.length}")
        lines.append(f"Period: {self.decomposition.series.domain.frequency.periods_per_year}")
        
        # Transformation
        if self.tramo_results.log_transformed:
            lines.append("\nTransformation: Logarithm")
            if self.tramo_results.transformation_adjustment:
                lines.append(f"  Adjustment: {self.tramo_results.transformation_adjustment:.2f}")
        else:
            lines.append("\nTransformation: None")
        
        # Model
        if self.tramo_results.arima_model:
            lines.append(f"\nARIMA Model: {self.tramo_results.arima_model.order}")
        
        # Outliers
        if self.tramo_results.outliers:
            lines.append(f"\nOutliers detected: {len(self.tramo_results.outliers)}")
            for outlier in self.tramo_results.outliers[:5]:  # Show first 5
                lines.append(f"  {outlier.name()}")
            if len(self.tramo_results.outliers) > 5:
                lines.append(f"  ... and {len(self.tramo_results.outliers) - 5} more")
        
        # Decomposition
        lines.append("\nDecomposition:")
        lines.append(f"  Mode: {self.decomposition.mode.name}")
        
        # Component variances
        if 'component_variances' in self.diagnostics:
            variances = self.diagnostics['component_variances']
            total_var = sum(v for v in variances.values() if v is not None)
            lines.append("\nComponent Contributions:")
            for comp, var in variances.items():
                if var is not None and total_var > 0:
                    pct = 100 * var / total_var
                    lines.append(f"  {comp.capitalize()}: {pct:.1f}%")
        
        # Key diagnostics
        lines.append("\nKey Diagnostics:")
        if 'residuals' in self.diagnostics.get('tramo', {}):
            residual_diag = self.diagnostics['tramo']['residuals']
            if 'independence_tests' in residual_diag:
                ljung_box = residual_diag['independence_tests'].get('ljung_box')
                if ljung_box:
                    lines.append(f"  Ljung-Box test p-value: {ljung_box.pvalue:.4f}")
        
        return "\n".join(lines)