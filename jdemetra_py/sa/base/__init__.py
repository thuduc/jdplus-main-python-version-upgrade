"""Base classes for seasonal adjustment."""

from .definition import SaDefinition, ComponentType, DecompositionMode
from .specification import SaSpecification
from .processor import SaProcessor
from .results import SaResults, SeriesDecomposition
from .estimation import EstimationPolicy, EstimationPolicyType

__all__ = [
    "SaDefinition",
    "ComponentType",
    "DecompositionMode",
    "SaSpecification",
    "SaProcessor",
    "SaResults",
    "SeriesDecomposition",
    "EstimationPolicy",
    "EstimationPolicyType",
]