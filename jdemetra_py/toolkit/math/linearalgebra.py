"""Linear algebra operations matching JDemetra+ implementation."""

from typing import Tuple, Optional, Union
import numpy as np
from scipy import linalg

from .matrices import FastMatrix, DataBlock


class QRDecomposition:
    """QR decomposition using Householder reflections."""
    
    def __init__(self, matrix: FastMatrix, pivot: bool = False):
        """Compute QR decomposition of matrix.
        
        Args:
            matrix: Input matrix
            pivot: Whether to use column pivoting
        """
        self._m = matrix.nrows
        self._n = matrix.ncols
        self._pivot = pivot
        
        # Perform decomposition
        if pivot:
            self._q, self._r, self._p = linalg.qr(matrix.to_array(), 
                                                   mode='economic', 
                                                   pivoting=True)
        else:
            self._q, self._r = linalg.qr(matrix.to_array(), mode='economic')
            self._p = None
    
    @property
    def q(self) -> FastMatrix:
        """Get Q matrix."""
        return FastMatrix(self._q)
    
    @property
    def r(self) -> FastMatrix:
        """Get R matrix."""
        return FastMatrix(self._r)
    
    @property 
    def pivots(self) -> Optional[np.ndarray]:
        """Get pivot indices if pivoting was used."""
        return self._p
    
    def solve(self, b: Union[DataBlock, FastMatrix]) -> Union[DataBlock, FastMatrix]:
        """Solve Ax = b using QR decomposition."""
        if isinstance(b, DataBlock):
            # Solve for vector
            x = linalg.solve_triangular(self._r, self._q.T @ b.to_array())
            return DataBlock(x)
        else:
            # Solve for matrix
            x = linalg.solve_triangular(self._r, self._q.T @ b.to_array())
            return FastMatrix(x)
    
    def is_full_rank(self) -> bool:
        """Check if matrix has full rank."""
        return np.linalg.matrix_rank(self._r) == min(self._m, self._n)


class SVD:
    """Singular Value Decomposition."""
    
    def __init__(self, matrix: FastMatrix, compute_uv: bool = True):
        """Compute SVD of matrix.
        
        Args:
            matrix: Input matrix
            compute_uv: Whether to compute U and V matrices
        """
        if compute_uv:
            self._u, self._s, self._vt = linalg.svd(matrix.to_array(), 
                                                     full_matrices=False)
        else:
            self._s = linalg.svdvals(matrix.to_array())
            self._u = None
            self._vt = None
    
    @property
    def u(self) -> Optional[FastMatrix]:
        """Get U matrix."""
        return FastMatrix(self._u) if self._u is not None else None
    
    @property
    def s(self) -> np.ndarray:
        """Get singular values."""
        return self._s
    
    @property
    def v(self) -> Optional[FastMatrix]:
        """Get V matrix (not transposed)."""
        return FastMatrix(self._vt.T) if self._vt is not None else None
    
    @property
    def vt(self) -> Optional[FastMatrix]:
        """Get V transposed."""
        return FastMatrix(self._vt) if self._vt is not None else None
    
    def rank(self, tol: Optional[float] = None) -> int:
        """Compute matrix rank."""
        if tol is None:
            tol = self._s.max() * max(self._u.shape) * np.finfo(float).eps
        return np.sum(self._s > tol)
    
    def condition_number(self) -> float:
        """Compute condition number."""
        return self._s[0] / self._s[-1] if self._s[-1] > 0 else np.inf
    
    def pseudo_inverse(self) -> FastMatrix:
        """Compute Moore-Penrose pseudo-inverse."""
        if self._u is None or self._vt is None:
            raise ValueError("U and V matrices not computed")
        
        # Compute pseudo-inverse using SVD
        s_inv = np.zeros_like(self._s)
        tol = self._s.max() * max(self._u.shape) * np.finfo(float).eps
        s_inv[self._s > tol] = 1.0 / self._s[self._s > tol]
        
        return FastMatrix(self._vt.T @ np.diag(s_inv) @ self._u.T)


class Householder:
    """Householder reflection operations."""
    
    @staticmethod
    def make_householder(x: DataBlock) -> Tuple[DataBlock, float]:
        """Create Householder vector and beta.
        
        Args:
            x: Input vector
            
        Returns:
            Householder vector v and scalar beta
        """
        x_array = x.to_array()
        norm_x = np.linalg.norm(x_array)
        
        if norm_x == 0:
            return DataBlock(x_array), 0.0
        
        # Create Householder vector
        v = x_array.copy()
        v[0] += np.sign(x_array[0]) * norm_x
        v = v / np.linalg.norm(v)
        
        beta = 2.0
        
        return DataBlock(v), beta
    
    @staticmethod
    def apply_householder(v: DataBlock, beta: float, 
                         a: FastMatrix, from_left: bool = True) -> FastMatrix:
        """Apply Householder reflection to matrix.
        
        Args:
            v: Householder vector
            beta: Scalar beta
            a: Matrix to transform
            from_left: Apply from left (True) or right (False)
            
        Returns:
            Transformed matrix
        """
        v_array = v.to_array()
        a_array = a.to_array()
        
        if from_left:
            # H*A = A - beta*v*(v'*A)
            result = a_array - beta * np.outer(v_array, v_array @ a_array)
        else:
            # A*H = A - beta*(A*v)*v'
            result = a_array - beta * np.outer(a_array @ v_array, v_array)
        
        return FastMatrix(result)


class CholeskyDecomposition:
    """Cholesky decomposition for positive definite matrices."""
    
    def __init__(self, matrix: FastMatrix):
        """Compute Cholesky decomposition.
        
        Args:
            matrix: Positive definite matrix
        """
        try:
            self._l = linalg.cholesky(matrix.to_array(), lower=True)
            self._success = True
        except linalg.LinAlgError:
            self._l = None
            self._success = False
    
    @property
    def l(self) -> Optional[FastMatrix]:
        """Get lower triangular matrix L."""
        return FastMatrix(self._l) if self._l is not None else None
    
    @property
    def success(self) -> bool:
        """Check if decomposition succeeded."""
        return self._success
    
    def solve(self, b: Union[DataBlock, FastMatrix]) -> Union[DataBlock, FastMatrix]:
        """Solve Ax = b using Cholesky decomposition."""
        if not self._success:
            raise ValueError("Cholesky decomposition failed")
        
        if isinstance(b, DataBlock):
            # Forward substitution: Ly = b
            y = linalg.solve_triangular(self._l, b.to_array(), lower=True)
            # Back substitution: L'x = y
            x = linalg.solve_triangular(self._l.T, y, lower=False)
            return DataBlock(x)
        else:
            # Solve for matrix
            y = linalg.solve_triangular(self._l, b.to_array(), lower=True)
            x = linalg.solve_triangular(self._l.T, y, lower=False)
            return FastMatrix(x)
    
    def determinant(self) -> float:
        """Compute determinant using Cholesky factor."""
        if not self._success:
            raise ValueError("Cholesky decomposition failed")
        return np.prod(np.diag(self._l)) ** 2
    
    def log_determinant(self) -> float:
        """Compute log determinant."""
        if not self._success:
            raise ValueError("Cholesky decomposition failed")
        return 2 * np.sum(np.log(np.diag(self._l)))


class EigenDecomposition:
    """Eigenvalue decomposition."""
    
    def __init__(self, matrix: FastMatrix, symmetric: bool = False):
        """Compute eigenvalue decomposition.
        
        Args:
            matrix: Input matrix
            symmetric: Whether matrix is symmetric
        """
        if symmetric:
            self._values, self._vectors = linalg.eigh(matrix.to_array())
        else:
            self._values, self._vectors = linalg.eig(matrix.to_array())
    
    @property
    def values(self) -> np.ndarray:
        """Get eigenvalues."""
        return self._values
    
    @property
    def vectors(self) -> FastMatrix:
        """Get eigenvectors (as columns)."""
        return FastMatrix(self._vectors)
    
    def get_real_eigenvalues(self) -> np.ndarray:
        """Get real parts of eigenvalues."""
        return np.real(self._values)
    
    def get_imaginary_eigenvalues(self) -> np.ndarray:
        """Get imaginary parts of eigenvalues."""
        return np.imag(self._values)