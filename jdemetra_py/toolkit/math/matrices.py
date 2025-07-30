"""Matrix operations and utilities matching JDemetra+ FastMatrix."""

from typing import Union, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass


class FastMatrix:
    """Fast matrix implementation wrapping numpy arrays.
    
    Provides JDemetra+ compatible matrix operations.
    """
    
    def __init__(self, data: Optional[np.ndarray] = None, 
                 nrows: Optional[int] = None, 
                 ncols: Optional[int] = None,
                 copy: bool = True):
        """Initialize matrix.
        
        Args:
            data: Optional numpy array
            nrows: Number of rows (required if data is None)
            ncols: Number of columns (required if data is None)
            copy: Whether to copy the data
        """
        if data is not None:
            self._data = np.array(data, dtype=np.float64, copy=copy)
            if self._data.ndim == 1:
                # Convert 1D to column vector
                self._data = self._data.reshape(-1, 1)
            elif self._data.ndim > 2:
                raise ValueError("Data must be 1D or 2D")
        else:
            if nrows is None or ncols is None:
                raise ValueError("Must specify nrows and ncols if data is None")
            self._data = np.zeros((nrows, ncols), dtype=np.float64)
    
    @classmethod
    def make(cls, nrows: int, ncols: int) -> 'FastMatrix':
        """Create zero matrix of specified size."""
        return cls(nrows=nrows, ncols=ncols)
    
    @classmethod
    def identity(cls, n: int) -> 'FastMatrix':
        """Create identity matrix."""
        return cls(np.eye(n))
    
    @classmethod
    def diagonal(cls, diag: Union[np.ndarray, list]) -> 'FastMatrix':
        """Create diagonal matrix."""
        return cls(np.diag(diag))
    
    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self._data.shape[0]
    
    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self._data.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape."""
        return self._data.shape
    
    def is_empty(self) -> bool:
        """Check if matrix is empty."""
        return self.nrows == 0 or self.ncols == 0
    
    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.nrows == self.ncols
    
    def column(self, col: int) -> 'DataBlock':
        """Get column as DataBlock."""
        if col < 0 or col >= self.ncols:
            raise IndexError(f"Column {col} out of range")
        return DataBlock(self._data[:, col])
    
    def row(self, row: int) -> 'DataBlock':
        """Get row as DataBlock."""
        if row < 0 or row >= self.nrows:
            raise IndexError(f"Row {row} out of range")
        return DataBlock(self._data[row, :])
    
    def get(self, row: int, col: int) -> float:
        """Get element at position."""
        return self._data[row, col]
    
    def set(self, row: int, col: int, value: float):
        """Set element at position."""
        self._data[row, col] = value
    
    def extract(self, row_start: int, row_count: int, 
                col_start: int, col_count: int) -> 'FastMatrix':
        """Extract submatrix."""
        row_end = row_start + row_count
        col_end = col_start + col_count
        return FastMatrix(self._data[row_start:row_end, col_start:col_end])
    
    def transpose(self) -> 'FastMatrix':
        """Get transpose."""
        return FastMatrix(self._data.T)
    
    def copy(self) -> 'FastMatrix':
        """Create copy."""
        return FastMatrix(self._data, copy=True)
    
    def add(self, other: Union['FastMatrix', float]) -> 'FastMatrix':
        """Add matrix or scalar."""
        if isinstance(other, FastMatrix):
            return FastMatrix(self._data + other._data)
        else:
            return FastMatrix(self._data + other)
    
    def sub(self, other: Union['FastMatrix', float]) -> 'FastMatrix':
        """Subtract matrix or scalar."""
        if isinstance(other, FastMatrix):
            return FastMatrix(self._data - other._data)
        else:
            return FastMatrix(self._data - other)
    
    def mul(self, other: Union['FastMatrix', float]) -> 'FastMatrix':
        """Element-wise multiply."""
        if isinstance(other, FastMatrix):
            return FastMatrix(self._data * other._data)
        else:
            return FastMatrix(self._data * other)
    
    def times(self, other: 'FastMatrix') -> 'FastMatrix':
        """Matrix multiplication."""
        return FastMatrix(self._data @ other._data)
    
    def apply(self, fn: Callable[[float], float]) -> 'FastMatrix':
        """Apply function to all elements."""
        vectorized = np.vectorize(fn)
        return FastMatrix(vectorized(self._data))
    
    def diagonal_matrix(self) -> 'FastMatrix':
        """Extract diagonal as diagonal matrix."""
        if not self.is_square():
            raise ValueError("Matrix must be square")
        return FastMatrix.diagonal(np.diag(self._data))
    
    def trace(self) -> float:
        """Compute trace."""
        if not self.is_square():
            raise ValueError("Matrix must be square")
        return np.trace(self._data)
    
    def determinant(self) -> float:
        """Compute determinant."""
        if not self.is_square():
            raise ValueError("Matrix must be square")
        return np.linalg.det(self._data)
    
    def norm1(self) -> float:
        """Compute 1-norm (max column sum)."""
        return np.linalg.norm(self._data, ord=1)
    
    def norm2(self) -> float:
        """Compute 2-norm (spectral norm)."""
        return np.linalg.norm(self._data, ord=2)
    
    def norm_infinity(self) -> float:
        """Compute infinity norm (max row sum)."""
        return np.linalg.norm(self._data, ord=np.inf)
    
    def norm_frobenius(self) -> float:
        """Compute Frobenius norm."""
        return np.linalg.norm(self._data, ord='fro')
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._data.copy()
    
    def __getitem__(self, key):
        """Support indexing."""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Support indexed assignment."""
        self._data[key] = value
    
    def __repr__(self) -> str:
        return f"FastMatrix({self.nrows}x{self.ncols})"
    
    def __str__(self) -> str:
        return str(self._data)


class DataBlock:
    """Data block for vector operations (matching JDemetra+ DataBlock)."""
    
    def __init__(self, data: Union[np.ndarray, list], copy: bool = True):
        """Initialize data block."""
        if isinstance(data, list):
            self._data = np.array(data, dtype=np.float64)
        else:
            self._data = np.array(data, dtype=np.float64, copy=copy)
        
        # Ensure 1D
        self._data = self._data.ravel()
    
    @classmethod
    def make(cls, length: int) -> 'DataBlock':
        """Create zero data block."""
        return cls(np.zeros(length))
    
    @classmethod
    def of(cls, data: Union[np.ndarray, list]) -> 'DataBlock':
        """Create from data."""
        return cls(data)
    
    @property
    def length(self) -> int:
        """Length of data block."""
        return len(self._data)
    
    def get(self, index: int) -> float:
        """Get element at index."""
        return self._data[index]
    
    def set(self, index: int, value: float):
        """Set element at index."""
        self._data[index] = value
    
    def copy(self) -> 'DataBlock':
        """Create copy."""
        return DataBlock(self._data, copy=True)
    
    def extract(self, start: int, length: int) -> 'DataBlock':
        """Extract sub-block."""
        return DataBlock(self._data[start:start + length])
    
    def set_all(self, value: float):
        """Set all elements to value."""
        self._data.fill(value)
    
    def copy_from(self, other: Union['DataBlock', np.ndarray], start: int = 0):
        """Copy from another block."""
        if isinstance(other, DataBlock):
            data = other._data
        else:
            data = other
        
        self._data[start:start + len(data)] = data
    
    def add(self, value: float):
        """Add scalar to all elements."""
        self._data += value
    
    def mul(self, value: float):
        """Multiply all elements by scalar."""
        self._data *= value
    
    def dot(self, other: 'DataBlock') -> float:
        """Dot product."""
        return np.dot(self._data, other._data)
    
    def norm1(self) -> float:
        """L1 norm."""
        return np.linalg.norm(self._data, ord=1)
    
    def norm2(self) -> float:
        """L2 norm."""
        return np.linalg.norm(self._data, ord=2)
    
    def sum(self) -> float:
        """Sum of elements."""
        return np.sum(self._data)
    
    def ssq(self) -> float:
        """Sum of squares."""
        return np.sum(self._data ** 2)
    
    def apply(self, fn: Callable[[float], float]):
        """Apply function to all elements in-place."""
        vectorized = np.vectorize(fn)
        self._data = vectorized(self._data)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._data.copy()
    
    def __getitem__(self, key):
        """Support indexing."""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Support indexed assignment."""
        self._data[key] = value
    
    def __len__(self) -> int:
        """Length."""
        return self.length
    
    def __repr__(self) -> str:
        return f"DataBlock(length={self.length})"


@dataclass
class MatrixWindow:
    """Window into a matrix (for efficient submatrix operations)."""
    
    matrix: FastMatrix
    row_start: int
    row_count: int
    col_start: int
    col_count: int
    
    def get(self, row: int, col: int) -> float:
        """Get element in window coordinates."""
        return self.matrix.get(self.row_start + row, self.col_start + col)
    
    def set(self, row: int, col: int, value: float):
        """Set element in window coordinates."""
        self.matrix.set(self.row_start + row, self.col_start + col, value)
    
    def to_matrix(self) -> FastMatrix:
        """Extract window as new matrix."""
        return self.matrix.extract(self.row_start, self.row_count,
                                  self.col_start, self.col_count)