"""Standard state space model components."""

from typing import Optional, List
import numpy as np

from .statespace import StateSpaceModel, TimeInvariantSSM
from ..math.matrices import FastMatrix
from ..arima.models import ArimaModel


class LocalLevel(TimeInvariantSSM):
    """Local level (random walk) model.
    
    State: α[t+1] = α[t] + η[t], η[t] ~ N(0, σ²_η)
    Observation: y[t] = α[t] + ε[t], ε[t] ~ N(0, σ²_ε)
    """
    
    def __init__(self, level_variance: float = 1.0, obs_variance: float = 1.0):
        """Initialize local level model.
        
        Args:
            level_variance: Variance of level disturbance
            obs_variance: Variance of observation error
        """
        # Transition matrix (1x1 identity)
        T = FastMatrix.identity(1)
        
        # Measurement matrix (1x1 identity)
        Z = FastMatrix.identity(1)
        
        # State covariance
        Q = FastMatrix([[level_variance]])
        
        # Measurement variance
        H = obs_variance
        
        super().__init__(T, Z, Q, H)


class LocalLinearTrend(TimeInvariantSSM):
    """Local linear trend model.
    
    State: [μ[t+1], β[t+1]] = [[1, 1], [0, 1]] * [μ[t], β[t]] + η[t]
    Observation: y[t] = [1, 0] * [μ[t], β[t]] + ε[t]
    
    Where:
        μ[t] = level
        β[t] = slope
    """
    
    def __init__(self, 
                 level_variance: float = 1.0,
                 slope_variance: float = 1.0, 
                 obs_variance: float = 1.0):
        """Initialize local linear trend model.
        
        Args:
            level_variance: Variance of level disturbance
            slope_variance: Variance of slope disturbance
            obs_variance: Variance of observation error
        """
        # Transition matrix
        T = FastMatrix([[1, 1], [0, 1]])
        
        # Measurement matrix
        Z = FastMatrix([[1, 0]])
        
        # State covariance
        Q = FastMatrix([[level_variance, 0], [0, slope_variance]])
        
        # Measurement variance
        H = obs_variance
        
        super().__init__(T, Z, Q, H)


class SeasonalComponent(TimeInvariantSSM):
    """Seasonal component using dummy variable approach.
    
    The state vector contains s-1 seasonal components where s is the period.
    Sum constraint: Σ γ[t-j] = 0 for j=0 to s-1
    """
    
    def __init__(self, period: int, variance: float = 1.0, fixed: bool = False):
        """Initialize seasonal component.
        
        Args:
            period: Seasonal period
            variance: Variance of seasonal disturbance (0 if fixed)
            fixed: Whether seasonal pattern is fixed
        """
        self.period = period
        self.variance = 0.0 if fixed else variance
        
        # State dimension is period - 1
        dim = period - 1
        
        # Transition matrix: shifts seasonal components
        T = FastMatrix.make(dim, dim)
        # First row: -1, -1, ..., -1 (sum constraint)
        for j in range(dim):
            T.set(0, j, -1)
        # Remaining rows: shift operator
        for i in range(1, dim):
            T.set(i, i - 1, 1)
        
        # Measurement matrix: picks first component
        Z = FastMatrix.make(1, dim)
        Z.set(0, 0, 1)
        
        # State covariance: only first component has variance
        Q = FastMatrix.make(dim, dim)
        Q.set(0, 0, self.variance)
        
        # No measurement error (handled by other components)
        H = 0.0
        
        super().__init__(T, Z, Q, H)


class ArmaComponent(StateSpaceModel):
    """ARMA component in state space form."""
    
    def __init__(self, arima_model: ArimaModel):
        """Initialize ARMA component from ARIMA model.
        
        Args:
            arima_model: ARIMA model (must have d=0)
        """
        if arima_model.order.d != 0:
            raise ValueError("ARMA component requires stationary model (d=0)")
        
        self.arima_model = arima_model
        self.p = arima_model.order.p
        self.q = arima_model.order.q
        self.r = max(self.p, self.q + 1)
        
        # Build state space representation
        self._build_matrices()
    
    def _build_matrices(self):
        """Build state space matrices for ARMA model."""
        # State dimension
        self._state_dim = self.r
        
        # Transition matrix
        self.T = FastMatrix.make(self.r, self.r)
        
        # First row contains AR coefficients
        for j in range(self.p):
            self.T.set(0, j, self.arima_model.ar_params[j])
        
        # Shift operator for remaining rows
        for i in range(1, self.r):
            self.T.set(i, i - 1, 1)
        
        # Measurement matrix
        self.Z = FastMatrix.make(1, self.r)
        self.Z.set(0, 0, 1)
        
        # Selection matrix for state disturbance
        self.R = FastMatrix.make(self.r, 1)
        self.R.set(0, 0, 1)
        for j in range(self.q):
            self.R.set(j + 1, 0, self.arima_model.ma_params[j])
        
        # State covariance (scalar variance)
        self.Q = FastMatrix([[self.arima_model.variance]])
        
        # No additional measurement error
        self.H = 0.0
    
    def state_dim(self) -> int:
        return self._state_dim
    
    def obs_dim(self) -> int:
        return 1
    
    def transition_matrix(self, t: int) -> FastMatrix:
        return self.T
    
    def measurement_matrix(self, t: int) -> FastMatrix:
        return self.Z
    
    def state_covariance(self, t: int) -> FastMatrix:
        return self.Q
    
    def measurement_variance(self, t: int) -> float:
        return self.H
    
    def selection_matrix(self, t: int) -> Optional[FastMatrix]:
        return self.R


class RegressionComponent(StateSpaceModel):
    """Regression component with time-varying regressors."""
    
    def __init__(self, X: np.ndarray, coefficients: Optional[np.ndarray] = None):
        """Initialize regression component.
        
        Args:
            X: Matrix of regressors (n_obs x n_regressors)
            coefficients: Regression coefficients (if known)
        """
        self.X = X
        self.n_obs, self.n_regressors = X.shape
        
        if coefficients is not None:
            self.coefficients = coefficients
            self.fixed = True
        else:
            # Unknown coefficients - use diffuse initialization
            self.coefficients = np.zeros(self.n_regressors)
            self.fixed = False
    
    def state_dim(self) -> int:
        return self.n_regressors
    
    def obs_dim(self) -> int:
        return 1
    
    def transition_matrix(self, t: int) -> FastMatrix:
        # Regression coefficients are constant
        return FastMatrix.identity(self.n_regressors)
    
    def measurement_matrix(self, t: int) -> FastMatrix:
        # Time-varying based on regressors
        if t < self.n_obs:
            Z = FastMatrix.make(1, self.n_regressors)
            for j in range(self.n_regressors):
                Z.set(0, j, self.X[t, j])
            return Z
        else:
            # Beyond data - return zeros
            return FastMatrix.make(1, self.n_regressors)
    
    def state_covariance(self, t: int) -> FastMatrix:
        # No state evolution noise
        return FastMatrix.make(self.n_regressors, self.n_regressors)
    
    def measurement_variance(self, t: int) -> float:
        # No additional measurement error
        return 0.0
    
    def initial_state(self) -> 'StateVector':
        """Initial state for regression coefficients."""
        from .statespace import StateVector
        
        if self.fixed:
            # Known coefficients
            return StateVector(
                values=self.coefficients,
                covariance=FastMatrix.make(self.n_regressors, self.n_regressors)
            )
        else:
            # Diffuse initialization for unknown coefficients
            return StateVector(
                values=np.zeros(self.n_regressors),
                covariance=FastMatrix.identity(self.n_regressors).mul(1e7)
            )