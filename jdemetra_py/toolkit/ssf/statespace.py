"""State space model representation matching JDemetra+."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from ..math.matrices import FastMatrix


@dataclass
class StateVector:
    """State vector at time t."""
    values: np.ndarray
    covariance: Optional[FastMatrix] = None
    
    @property
    def dim(self) -> int:
        """State dimension."""
        return len(self.values)
    
    def copy(self) -> 'StateVector':
        """Create copy."""
        cov_copy = self.covariance.copy() if self.covariance else None
        return StateVector(self.values.copy(), cov_copy)


@dataclass  
class Measurement:
    """Measurement/observation at time t."""
    value: float
    variance: float = 1.0
    is_missing: bool = False
    
    @classmethod
    def missing(cls) -> 'Measurement':
        """Create missing observation."""
        return cls(np.nan, 1.0, True)


class StateSpaceModel(ABC):
    """Abstract base class for state space models.
    
    State space representation:
        State equation: α[t+1] = T[t] * α[t] + R[t] * η[t]
        Measurement equation: y[t] = Z[t] * α[t] + ε[t]
        
    Where:
        α[t] = state vector
        y[t] = observation
        η[t] ~ N(0, Q[t]) = state disturbances
        ε[t] ~ N(0, H[t]) = measurement error
    """
    
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state vector."""
        pass
    
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observations."""
        pass
    
    @abstractmethod
    def transition_matrix(self, t: int) -> FastMatrix:
        """Get transition matrix T[t]."""
        pass
    
    @abstractmethod
    def measurement_matrix(self, t: int) -> FastMatrix:
        """Get measurement matrix Z[t]."""
        pass
    
    @abstractmethod
    def state_covariance(self, t: int) -> FastMatrix:
        """Get state disturbance covariance Q[t]."""
        pass
    
    @abstractmethod
    def measurement_variance(self, t: int) -> float:
        """Get measurement error variance H[t]."""
        pass
    
    def selection_matrix(self, t: int) -> Optional[FastMatrix]:
        """Get selection matrix R[t] (optional).
        
        If None, R is assumed to be identity.
        """
        return None
    
    def initial_state(self) -> StateVector:
        """Get initial state distribution."""
        # Default: diffuse initialization
        dim = self.state_dim()
        return StateVector(
            values=np.zeros(dim),
            covariance=FastMatrix.identity(dim).mul(1e6)
        )
    
    def is_time_invariant(self) -> bool:
        """Check if model is time-invariant."""
        return False
    
    def predict_state(self, state: StateVector, t: int) -> StateVector:
        """Predict next state.
        
        Args:
            state: Current state
            t: Time index
            
        Returns:
            Predicted state at t+1
        """
        T = self.transition_matrix(t)
        
        # Predicted state mean
        predicted_mean = T.times(FastMatrix(state.values.reshape(-1, 1))).to_array().ravel()
        
        # Predicted state covariance
        if state.covariance is not None:
            Q = self.state_covariance(t)
            R = self.selection_matrix(t)
            
            if R is not None:
                # P[t+1|t] = T * P[t|t] * T' + R * Q * R'
                RQRt = R.times(Q).times(R.transpose())
                predicted_cov = T.times(state.covariance).times(T.transpose()).add(RQRt)
            else:
                # P[t+1|t] = T * P[t|t] * T' + Q
                predicted_cov = T.times(state.covariance).times(T.transpose()).add(Q)
        else:
            predicted_cov = None
        
        return StateVector(predicted_mean, predicted_cov)
    
    def predict_observation(self, state: StateVector, t: int) -> Tuple[float, float]:
        """Predict observation from state.
        
        Args:
            state: Current state
            t: Time index
            
        Returns:
            Predicted observation and its variance
        """
        Z = self.measurement_matrix(t)
        H = self.measurement_variance(t)
        
        # Predicted observation
        y_pred = (Z.times(FastMatrix(state.values.reshape(-1, 1)))).to_array()[0, 0]
        
        # Prediction variance
        if state.covariance is not None:
            # F[t] = Z * P[t|t-1] * Z' + H
            f = (Z.times(state.covariance).times(Z.transpose())).to_array()[0, 0] + H
        else:
            f = H
        
        return y_pred, f


class TimeInvariantSSM(StateSpaceModel):
    """Time-invariant state space model."""
    
    def __init__(self,
                 T: FastMatrix,
                 Z: FastMatrix,
                 Q: FastMatrix,
                 H: float,
                 R: Optional[FastMatrix] = None):
        """Initialize time-invariant model.
        
        Args:
            T: Transition matrix
            Z: Measurement matrix
            Q: State covariance matrix
            H: Measurement variance
            R: Selection matrix (optional)
        """
        self.T = T
        self.Z = Z
        self.Q = Q
        self.H = H
        self.R = R
        
        self._state_dim = T.nrows
        self._obs_dim = Z.nrows
    
    def state_dim(self) -> int:
        return self._state_dim
    
    def obs_dim(self) -> int:
        return self._obs_dim
    
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
    
    def is_time_invariant(self) -> bool:
        return True


class CompositeSSM(StateSpaceModel):
    """Composite state space model from multiple components."""
    
    def __init__(self, components: List[StateSpaceModel]):
        """Initialize composite model.
        
        Args:
            components: List of component models
        """
        self.components = components
        self._state_dims = [c.state_dim() for c in components]
        self._state_dim_total = sum(self._state_dims)
        self._state_offsets = np.cumsum([0] + self._state_dims[:-1])
    
    def state_dim(self) -> int:
        return self._state_dim_total
    
    def obs_dim(self) -> int:
        # Assume all components contribute to same observation
        return 1
    
    def transition_matrix(self, t: int) -> FastMatrix:
        """Build block-diagonal transition matrix."""
        T = FastMatrix.make(self._state_dim_total, self._state_dim_total)
        
        for i, comp in enumerate(self.components):
            offset = self._state_offsets[i]
            comp_T = comp.transition_matrix(t)
            
            # Copy component transition matrix to appropriate block
            for r in range(comp_T.nrows):
                for c in range(comp_T.ncols):
                    T.set(offset + r, offset + c, comp_T.get(r, c))
        
        return T
    
    def measurement_matrix(self, t: int) -> FastMatrix:
        """Build measurement matrix by concatenating components."""
        Z = FastMatrix.make(1, self._state_dim_total)
        
        for i, comp in enumerate(self.components):
            offset = self._state_offsets[i]
            comp_Z = comp.measurement_matrix(t)
            
            # Copy component measurement matrix
            for c in range(comp_Z.ncols):
                Z.set(0, offset + c, comp_Z.get(0, c))
        
        return Z
    
    def state_covariance(self, t: int) -> FastMatrix:
        """Build block-diagonal state covariance."""
        Q = FastMatrix.make(self._state_dim_total, self._state_dim_total)
        
        for i, comp in enumerate(self.components):
            offset = self._state_offsets[i]
            comp_Q = comp.state_covariance(t)
            
            # Copy component covariance to appropriate block
            for r in range(comp_Q.nrows):
                for c in range(comp_Q.ncols):
                    Q.set(offset + r, offset + c, comp_Q.get(r, c))
        
        return Q
    
    def measurement_variance(self, t: int) -> float:
        """Combined measurement variance."""
        # Sum variances from all components
        return sum(comp.measurement_variance(t) for comp in self.components)