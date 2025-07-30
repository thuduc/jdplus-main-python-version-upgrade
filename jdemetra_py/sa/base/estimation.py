"""Estimation policies for seasonal adjustment."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class EstimationPolicyType(Enum):
    """Type of estimation policy."""
    
    NONE = "None"
    CURRENT = "Current"
    FIXED = "Fixed"
    FIXED_PARAMETERS = "FixedParameters"
    FIXED_AUTO_REGRESSORS = "FixedAutoRegressors"
    FREE_PARAMETERS = "FreeParameters"
    OUTLIERS = "Outliers"
    OUTLIERS_STOCHASTIC = "OutliersStochastic"
    LAST_OUTLIERS = "LastOutliers"
    COMPLETE = "Complete"


@dataclass
class EstimationPolicy:
    """Policy for model estimation in seasonal adjustment."""
    
    policy_type: EstimationPolicyType = EstimationPolicyType.COMPLETE
    
    # Span for estimation
    estimation_span_start: Optional['TsPeriod'] = None
    estimation_span_end: Optional['TsPeriod'] = None
    
    # Outlier span
    outlier_span_start: Optional['TsPeriod'] = None
    outlier_span_end: Optional['TsPeriod'] = None
    
    # Tolerance
    tolerance: float = 1e-7
    
    def is_fixed(self) -> bool:
        """Check if policy fixes some parameters."""
        return self.policy_type in [
            EstimationPolicyType.FIXED,
            EstimationPolicyType.FIXED_PARAMETERS,
            EstimationPolicyType.FIXED_AUTO_REGRESSORS
        ]
    
    def allows_outlier_detection(self) -> bool:
        """Check if policy allows outlier detection."""
        return self.policy_type in [
            EstimationPolicyType.OUTLIERS,
            EstimationPolicyType.OUTLIERS_STOCHASTIC,
            EstimationPolicyType.LAST_OUTLIERS,
            EstimationPolicyType.COMPLETE
        ]
    
    def allows_parameter_estimation(self) -> bool:
        """Check if policy allows parameter estimation."""
        return self.policy_type not in [
            EstimationPolicyType.NONE,
            EstimationPolicyType.FIXED,
            EstimationPolicyType.FIXED_PARAMETERS
        ]