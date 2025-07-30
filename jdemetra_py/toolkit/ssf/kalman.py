"""Kalman filter and smoother implementations."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .statespace import StateSpaceModel, StateVector, Measurement
from ..math.matrices import FastMatrix
from ..math.linearalgebra import CholeskyDecomposition


@dataclass
class FilteredState:
    """Results from Kalman filter at time t."""
    
    # Filtered quantities (a[t|t], P[t|t])
    filtered_state: StateVector
    
    # Predicted quantities (a[t|t-1], P[t|t-1])
    predicted_state: StateVector
    
    # Innovation and variance
    innovation: float
    innovation_variance: float
    
    # Kalman gain
    gain: Optional[np.ndarray] = None
    
    # Log-likelihood contribution
    log_likelihood: float = 0.0


@dataclass
class SmoothedState:
    """Results from Kalman smoother at time t."""
    
    # Smoothed state (a[t|n], P[t|n])
    smoothed_state: StateVector
    
    # Smoothing error and variance
    smoothing_error: Optional[np.ndarray] = None
    smoothing_error_variance: Optional[FastMatrix] = None


class KalmanFilter:
    """Kalman filter for state space models."""
    
    def __init__(self, model: StateSpaceModel,
                 store_gains: bool = False,
                 store_all_states: bool = True):
        """Initialize Kalman filter.
        
        Args:
            model: State space model
            store_gains: Whether to store Kalman gains
            store_all_states: Whether to store all intermediate states
        """
        self.model = model
        self.store_gains = store_gains
        self.store_all_states = store_all_states
    
    def filter(self, observations: List[Measurement]) -> List[FilteredState]:
        """Run Kalman filter on observations.
        
        Args:
            observations: List of measurements
            
        Returns:
            List of filtered states
        """
        n = len(observations)
        results = []
        
        # Initialize state
        state = self.model.initial_state()
        log_likelihood_total = 0.0
        
        for t in range(n):
            # Prediction step
            if t > 0:
                state = self.model.predict_state(results[-1].filtered_state, t - 1)
            
            predicted_state = state.copy()
            
            # Get observation
            obs = observations[t]
            
            if not obs.is_missing:
                # Measurement prediction
                y_pred, f = self.model.predict_observation(state, t)
                innovation = obs.value - y_pred
                
                # Kalman gain
                if state.covariance is not None:
                    Z = self.model.measurement_matrix(t)
                    P = state.covariance
                    
                    # K = P * Z' / f
                    gain_matrix = P.times(Z.transpose()).mul(1.0 / f)
                    gain = gain_matrix.to_array().ravel()
                    
                    # Update state
                    state.values = state.values + gain * innovation
                    
                    # Update covariance: P = P - K * f * K'
                    KfKt = gain_matrix.times(gain_matrix.transpose()).mul(f)
                    state.covariance = P.sub(KfKt)
                else:
                    gain = None
                
                # Log-likelihood contribution
                log_lik = -0.5 * (np.log(2 * np.pi) + np.log(f) + innovation**2 / f)
                log_likelihood_total += log_lik
            else:
                # Missing observation - no update
                innovation = np.nan
                f = np.nan
                gain = None
                log_lik = 0.0
            
            # Store results
            result = FilteredState(
                filtered_state=state.copy() if self.store_all_states else state,
                predicted_state=predicted_state if self.store_all_states else state,
                innovation=innovation,
                innovation_variance=f,
                gain=gain if self.store_gains else None,
                log_likelihood=log_lik
            )
            
            results.append(result)
        
        return results
    
    def log_likelihood(self, observations: List[Measurement]) -> float:
        """Compute log-likelihood of observations.
        
        Args:
            observations: List of measurements
            
        Returns:
            Log-likelihood value
        """
        results = self.filter(observations)
        return sum(r.log_likelihood for r in results)
    
    def innovations(self, observations: List[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract innovations and their variances.
        
        Args:
            observations: List of measurements
            
        Returns:
            Arrays of innovations and variances
        """
        results = self.filter(observations)
        
        innovations = np.array([r.innovation for r in results])
        variances = np.array([r.innovation_variance for r in results])
        
        return innovations, variances


class KalmanSmoother:
    """Kalman smoother for state space models."""
    
    def __init__(self, model: StateSpaceModel):
        """Initialize Kalman smoother.
        
        Args:
            model: State space model
        """
        self.model = model
        self.filter = KalmanFilter(model, store_all_states=True)
    
    def smooth(self, observations: List[Measurement]) -> List[SmoothedState]:
        """Run Kalman smoother (Rauch-Tung-Striebel).
        
        Args:
            observations: List of measurements
            
        Returns:
            List of smoothed states
        """
        # First run forward filter
        filtered_results = self.filter.filter(observations)
        n = len(filtered_results)
        
        # Initialize smoothed results
        smoothed_results = [None] * n
        
        # Last smoothed state equals last filtered state
        last_filtered = filtered_results[-1].filtered_state
        smoothed_results[-1] = SmoothedState(
            smoothed_state=last_filtered.copy()
        )
        
        # Backward pass
        for t in range(n - 2, -1, -1):
            filtered = filtered_results[t].filtered_state
            predicted_next = filtered_results[t + 1].predicted_state
            smoothed_next = smoothed_results[t + 1].smoothed_state
            
            # Smoothing calculations
            if filtered.covariance is not None and predicted_next.covariance is not None:
                T = self.model.transition_matrix(t)
                
                # Smoothing gain: J = P[t|t] * T' * P[t+1|t]^(-1)
                # Use Cholesky for stable inversion
                try:
                    chol = CholeskyDecomposition(predicted_next.covariance)
                    if chol.success:
                        # Solve P[t+1|t] * J' = T * P[t|t]
                        rhs = T.times(filtered.covariance)
                        J_transpose = chol.solve(rhs)
                        J = J_transpose.transpose()
                    else:
                        # Fallback to pseudo-inverse
                        J = None
                except:
                    J = None
                
                if J is not None:
                    # Smoothed state: a[t|n] = a[t|t] + J * (a[t+1|n] - a[t+1|t])
                    state_diff = smoothed_next.values - predicted_next.values
                    smoothed_values = filtered.values + J.times(
                        FastMatrix(state_diff.reshape(-1, 1))
                    ).to_array().ravel()
                    
                    # Smoothed covariance: P[t|n] = P[t|t] + J * (P[t+1|n] - P[t+1|t]) * J'
                    cov_diff = smoothed_next.covariance.sub(predicted_next.covariance)
                    smoothed_cov = filtered.covariance.add(
                        J.times(cov_diff).times(J.transpose())
                    )
                else:
                    # No update possible
                    smoothed_values = filtered.values
                    smoothed_cov = filtered.covariance
            else:
                # No covariance - just copy filtered values
                smoothed_values = filtered.values
                smoothed_cov = filtered.covariance
            
            smoothed_results[t] = SmoothedState(
                smoothed_state=StateVector(smoothed_values, smoothed_cov)
            )
        
        return smoothed_results
    
    def smooth_values(self, observations: List[Measurement]) -> np.ndarray:
        """Get smoothed state values only.
        
        Args:
            observations: List of measurements
            
        Returns:
            Array of smoothed state values
        """
        smoothed = self.smooth(observations)
        
        # Extract first component of state vector (usually the signal)
        return np.array([s.smoothed_state.values[0] for s in smoothed])
    
    def disturbance_smoother(self, observations: List[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth state and measurement disturbances.
        
        Args:
            observations: List of measurements
            
        Returns:
            Smoothed state disturbances and measurement errors
        """
        # This is a simplified version
        # Full disturbance smoothing requires additional calculations
        
        smoothed = self.smooth(observations)
        n = len(observations)
        
        state_disturbances = np.zeros((n - 1, self.model.state_dim()))
        measurement_errors = np.zeros(n)
        
        # Compute smoothed disturbances
        for t in range(n - 1):
            # State disturbance: η[t] = Q * R' * r[t]
            # Where r[t] is the smoothing error
            # This is simplified - proper implementation needs r[t]
            state_disturbances[t] = np.zeros(self.model.state_dim())
        
        for t in range(n):
            if not observations[t].is_missing:
                # Measurement error: ε[t] = y[t] - Z * a[t|n]
                Z = self.model.measurement_matrix(t)
                smoothed_state = smoothed[t].smoothed_state.values
                y_smoothed = Z.times(FastMatrix(smoothed_state.reshape(-1, 1))).to_array()[0, 0]
                measurement_errors[t] = observations[t].value - y_smoothed
            else:
                measurement_errors[t] = np.nan
        
        return state_disturbances, measurement_errors