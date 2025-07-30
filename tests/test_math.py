"""Unit tests for mathematical utilities."""

import pytest
import numpy as np

from jdemetra_py.toolkit.math import (
    FastMatrix, DataBlock, Polynomial,
    QRDecomposition, SVD, CholeskyDecomposition
)


class TestFastMatrix:
    """Tests for FastMatrix class."""
    
    def test_creation(self):
        # From shape
        m1 = FastMatrix.make(3, 4)
        assert m1.nrows == 3
        assert m1.ncols == 4
        assert np.all(m1.to_array() == 0)
        
        # From array
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        m2 = FastMatrix(arr)
        assert m2.nrows == 3
        assert m2.ncols == 2
        
        # Identity
        m3 = FastMatrix.identity(3)
        assert m3.is_square()
        assert m3.trace() == 3.0
        
    def test_indexing(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        m = FastMatrix(arr)
        
        assert m.get(0, 0) == 1
        assert m.get(1, 2) == 6
        
        m.set(0, 1, 10)
        assert m.get(0, 1) == 10
        
        # Direct indexing
        assert m[1, 1] == 5
        m[1, 1] = 20
        assert m[1, 1] == 20
        
    def test_column_row_access(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        m = FastMatrix(arr)
        
        col1 = m.column(1)
        assert isinstance(col1, DataBlock)
        np.testing.assert_array_equal(col1.to_array(), [2, 5])
        
        row0 = m.row(0)
        np.testing.assert_array_equal(row0.to_array(), [1, 2, 3])
        
    def test_arithmetic(self):
        m1 = FastMatrix(np.array([[1, 2], [3, 4]]))
        m2 = FastMatrix(np.array([[5, 6], [7, 8]]))
        
        # Addition
        m3 = m1.add(m2)
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(m3.to_array(), expected)
        
        # Scalar multiplication
        m4 = m1.mul(2)
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(m4.to_array(), expected)
        
        # Matrix multiplication
        m5 = m1.times(m2)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(m5.to_array(), expected)
        
    def test_matrix_properties(self):
        # Square matrix
        m = FastMatrix(np.array([[1, 2], [3, 4]]))
        assert m.is_square()
        assert m.determinant() == pytest.approx(-2.0)
        assert m.trace() == 5.0
        
        # Transpose
        mt = m.transpose()
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(mt.to_array(), expected)
        
    def test_norms(self):
        m = FastMatrix(np.array([[1, -2], [3, 4]]))
        
        assert m.norm1() == 6.0  # Max column sum
        assert m.norm_infinity() == 7.0  # Max row sum
        assert m.norm_frobenius() == pytest.approx(np.sqrt(30))


class TestDataBlock:
    """Tests for DataBlock class."""
    
    def test_creation(self):
        # From list
        db1 = DataBlock([1, 2, 3, 4])
        assert db1.length == 4
        
        # From array
        db2 = DataBlock(np.array([5, 6, 7]))
        assert db2.length == 3
        
        # Make zeros
        db3 = DataBlock.make(5)
        assert db3.length == 5
        assert np.all(db3.to_array() == 0)
        
    def test_operations(self):
        db = DataBlock([1, 2, 3, 4])
        
        # Element access
        assert db.get(0) == 1
        assert db[2] == 3
        
        db.set(1, 10)
        assert db.get(1) == 10
        
        # Arithmetic
        db.add(5)
        np.testing.assert_array_equal(db.to_array(), [6, 15, 8, 9])
        
        db.mul(2)
        np.testing.assert_array_equal(db.to_array(), [12, 30, 16, 18])
        
    def test_vector_operations(self):
        db1 = DataBlock([1, 2, 3])
        db2 = DataBlock([4, 5, 6])
        
        # Dot product
        assert db1.dot(db2) == 32  # 1*4 + 2*5 + 3*6
        
        # Norms
        assert db1.norm1() == 6  # |1| + |2| + |3|
        assert db1.norm2() == pytest.approx(np.sqrt(14))
        
        # Sum and sum of squares
        assert db1.sum() == 6
        assert db1.ssq() == 14


class TestPolynomial:
    """Tests for Polynomial class."""
    
    def test_creation(self):
        # From coefficients
        p1 = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
        assert p1.degree == 2
        np.testing.assert_array_equal(p1.coefficients, [1, 2, 3])
        
        # Zero polynomial
        p2 = Polynomial.zero()
        assert p2.is_zero()
        assert p2.degree == 0
        
        # Monomial
        p3 = Polynomial.monomial(3, 2.0)  # 2x^3
        assert p3.degree == 3
        assert p3.get(3) == 2.0
        
    def test_evaluation(self):
        p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
        
        assert p.evaluate(0) == 1
        assert p.evaluate(1) == 6
        assert p.evaluate(2) == 17
        
        # Array evaluation
        x = np.array([0, 1, 2])
        np.testing.assert_array_equal(p.evaluate(x), [1, 6, 17])
        
    def test_arithmetic(self):
        p1 = Polynomial([1, 2])  # 1 + 2x
        p2 = Polynomial([3, 4])  # 3 + 4x
        
        # Addition
        p3 = p1.add(p2)
        np.testing.assert_array_equal(p3.coefficients, [4, 6])
        
        # Multiplication
        p4 = p1.mul(p2)  # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
        np.testing.assert_array_equal(p4.coefficients, [3, 10, 8])
        
        # Scalar multiplication
        p5 = p1.times(3)
        np.testing.assert_array_equal(p5.coefficients, [3, 6])
        
    def test_division(self):
        # (x^2 + 3x + 2) / (x + 1) = x + 2
        dividend = Polynomial([2, 3, 1])  # 2 + 3x + x^2
        divisor = Polynomial([1, 1])  # 1 + x
        
        quotient, remainder = dividend.div(divisor)
        np.testing.assert_array_almost_equal(quotient.coefficients, [2, 1])
        assert remainder.is_zero() or remainder.degree == 0
        
    def test_calculus(self):
        p = Polynomial([1, 2, 3, 4])  # 1 + 2x + 3x^2 + 4x^3
        
        # Derivative: 2 + 6x + 12x^2
        dp = p.differentiate()
        np.testing.assert_array_equal(dp.coefficients, [2, 6, 12])
        
        # Second derivative: 6 + 24x
        d2p = p.differentiate(2)
        np.testing.assert_array_equal(d2p.coefficients, [6, 24])
        
        # Integration (with constant 0)
        ip = p.integrate()
        # x + x^2 + x^3 + x^4
        np.testing.assert_array_almost_equal(ip.coefficients, [0, 1, 1, 1, 1])
        
    def test_roots(self):
        # (x - 1)(x - 2) = x^2 - 3x + 2
        p = Polynomial([2, -3, 1])
        roots = p.roots()
        
        assert len(roots) == 2
        np.testing.assert_array_almost_equal(sorted(roots), [1, 2])


class TestLinearAlgebra:
    """Tests for linear algebra operations."""
    
    def test_qr_decomposition(self):
        # Test matrix
        A = FastMatrix(np.array([[1, 2], [3, 4], [5, 6]]))
        
        qr = QRDecomposition(A)
        Q = qr.q
        R = qr.r
        
        # Check dimensions
        assert Q.nrows == 3
        assert Q.ncols == 2
        assert R.nrows == 2
        assert R.ncols == 2
        
        # Check orthogonality of Q
        QtQ = Q.transpose().times(Q)
        np.testing.assert_array_almost_equal(QtQ.to_array(), np.eye(2))
        
        # Check A = QR
        A_reconstructed = Q.times(R)
        np.testing.assert_array_almost_equal(A_reconstructed.to_array(), A.to_array())
        
    def test_svd(self):
        A = FastMatrix(np.array([[1, 2], [3, 4], [5, 6]]))
        
        svd = SVD(A)
        U = svd.u
        S = svd.s
        Vt = svd.vt
        
        # Check dimensions
        assert U.nrows == 3
        assert U.ncols == 2
        assert len(S) == 2
        assert Vt.nrows == 2
        assert Vt.ncols == 2
        
        # Check singular values are non-negative and sorted
        assert np.all(S >= 0)
        assert np.all(S[:-1] >= S[1:])
        
        # Reconstruct A
        S_mat = FastMatrix.make(2, 2)
        S_mat[0, 0] = S[0]
        S_mat[1, 1] = S[1]
        
        A_reconstructed = U.times(S_mat).times(Vt)
        np.testing.assert_array_almost_equal(A_reconstructed.to_array(), A.to_array())
        
    def test_cholesky(self):
        # Create positive definite matrix
        A = np.array([[4, 2], [2, 3]])
        matrix = FastMatrix(A)
        
        chol = CholeskyDecomposition(matrix)
        assert chol.success
        
        L = chol.l
        
        # Check A = L * L'
        LLt = L.times(L.transpose())
        np.testing.assert_array_almost_equal(LLt.to_array(), A)
        
        # Test solving
        b = DataBlock([1, 2])
        x = chol.solve(b)
        
        # Check Ax = b
        Ax = matrix.times(FastMatrix(x.to_array().reshape(-1, 1)))
        np.testing.assert_array_almost_equal(Ax.to_array().ravel(), b.to_array())