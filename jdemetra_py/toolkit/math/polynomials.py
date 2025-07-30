"""Polynomial operations matching JDemetra+ implementation."""

from typing import Union, List, Tuple, Optional
import numpy as np
from numpy.polynomial import Polynomial as NpPoly


class Polynomial:
    """Polynomial representation with JDemetra+ compatible operations.
    
    Coefficients are stored in ascending order: a0 + a1*x + a2*x^2 + ...
    """
    
    def __init__(self, coefficients: Union[List[float], np.ndarray]):
        """Initialize polynomial with coefficients.
        
        Args:
            coefficients: Polynomial coefficients in ascending order
        """
        # Remove trailing zeros
        coeffs = np.array(coefficients, dtype=np.float64)
        
        # Find last non-zero coefficient
        nonzero = np.nonzero(coeffs)[0]
        if len(nonzero) > 0:
            self._coeffs = coeffs[:nonzero[-1] + 1]
        else:
            # All zeros - keep just one
            self._coeffs = np.array([0.0])
    
    @classmethod
    def zero(cls) -> 'Polynomial':
        """Create zero polynomial."""
        return cls([0.0])
    
    @classmethod
    def one(cls) -> 'Polynomial':
        """Create unit polynomial."""
        return cls([1.0])
    
    @classmethod
    def monomial(cls, degree: int, coefficient: float = 1.0) -> 'Polynomial':
        """Create monomial c*x^degree."""
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = coefficient
        return cls(coeffs)
    
    @classmethod
    def from_roots(cls, roots: List[complex]) -> 'Polynomial':
        """Create polynomial from roots."""
        # Use numpy to create polynomial from roots
        np_poly = NpPoly.fromroots(roots)
        return cls(np_poly.coef)
    
    @property
    def degree(self) -> int:
        """Degree of polynomial."""
        return len(self._coeffs) - 1
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get coefficients array."""
        return self._coeffs.copy()
    
    def get(self, index: int) -> float:
        """Get coefficient at index."""
        if index < 0 or index >= len(self._coeffs):
            return 0.0
        return self._coeffs[index]
    
    def is_zero(self) -> bool:
        """Check if polynomial is zero."""
        return self.degree == 0 and self._coeffs[0] == 0.0
    
    def is_one(self) -> bool:
        """Check if polynomial is one."""
        return self.degree == 0 and self._coeffs[0] == 1.0
    
    def evaluate(self, x: Union[float, complex, np.ndarray]) -> Union[float, complex, np.ndarray]:
        """Evaluate polynomial at x."""
        # Use Horner's method for efficiency
        result = self._coeffs[-1]
        for i in range(len(self._coeffs) - 2, -1, -1):
            result = result * x + self._coeffs[i]
        return result
    
    def add(self, other: Union['Polynomial', float]) -> 'Polynomial':
        """Add polynomial or scalar."""
        if isinstance(other, (int, float)):
            # Add scalar to constant term
            new_coeffs = self._coeffs.copy()
            new_coeffs[0] += other
            return Polynomial(new_coeffs)
        
        # Add polynomials
        max_len = max(len(self._coeffs), len(other._coeffs))
        result = np.zeros(max_len)
        result[:len(self._coeffs)] = self._coeffs
        result[:len(other._coeffs)] += other._coeffs
        return Polynomial(result)
    
    def sub(self, other: Union['Polynomial', float]) -> 'Polynomial':
        """Subtract polynomial or scalar."""
        if isinstance(other, (int, float)):
            return self.add(-other)
        
        # Subtract polynomials
        max_len = max(len(self._coeffs), len(other._coeffs))
        result = np.zeros(max_len)
        result[:len(self._coeffs)] = self._coeffs
        result[:len(other._coeffs)] -= other._coeffs
        return Polynomial(result)
    
    def mul(self, other: Union['Polynomial', float]) -> 'Polynomial':
        """Multiply by polynomial or scalar."""
        if isinstance(other, (int, float)):
            return Polynomial(self._coeffs * other)
        
        # Polynomial multiplication using convolution
        return Polynomial(np.convolve(self._coeffs, other._coeffs))
    
    def div(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """Polynomial division returning quotient and remainder."""
        if divisor.is_zero():
            raise ValueError("Division by zero polynomial")
        
        # Use numpy polynomial division
        dividend_poly = NpPoly(self._coeffs)
        divisor_poly = NpPoly(divisor._coeffs)
        
        quotient_poly, remainder_poly = divmod(dividend_poly, divisor_poly)
        
        return Polynomial(quotient_poly.coef), Polynomial(remainder_poly.coef)
    
    def times(self, scalar: float) -> 'Polynomial':
        """Multiply by scalar."""
        return Polynomial(self._coeffs * scalar)
    
    def negate(self) -> 'Polynomial':
        """Negate polynomial."""
        return Polynomial(-self._coeffs)
    
    def differentiate(self, n: int = 1) -> 'Polynomial':
        """Compute nth derivative."""
        if n < 0:
            raise ValueError("Derivative order must be non-negative")
        if n == 0:
            return Polynomial(self._coeffs)
        
        # Use numpy polynomial derivative
        poly = NpPoly(self._coeffs)
        for _ in range(n):
            poly = poly.deriv()
        
        return Polynomial(poly.coef if len(poly.coef) > 0 else [0.0])
    
    def integrate(self, constant: float = 0.0) -> 'Polynomial':
        """Compute integral with given constant."""
        # Use numpy polynomial integration
        poly = NpPoly(self._coeffs)
        integrated = poly.integ(lbnd=constant)
        return Polynomial(integrated.coef)
    
    def roots(self) -> np.ndarray:
        """Find polynomial roots."""
        if self.degree == 0:
            return np.array([])
        
        # Use numpy to find roots
        poly = NpPoly(self._coeffs)
        return poly.roots()
    
    def reverse(self) -> 'Polynomial':
        """Reverse polynomial coefficients (x^n * P(1/x))."""
        return Polynomial(self._coeffs[::-1])
    
    def shift(self, shift: float) -> 'Polynomial':
        """Shift polynomial: P(x+shift)."""
        # Use Taylor expansion
        poly = NpPoly(self._coeffs)
        # Convert to different basis (shift origin)
        new_poly = poly(NpPoly([shift, 1]))
        return Polynomial(new_poly.coef)
    
    def gcd(self, other: 'Polynomial') -> 'Polynomial':
        """Greatest common divisor of two polynomials."""
        a, b = self, other
        
        while not b.is_zero():
            _, remainder = a.div(b)
            a, b = b, remainder
        
        # Normalize so leading coefficient is 1
        if a.degree >= 0 and a._coeffs[-1] != 0:
            return a.times(1.0 / a._coeffs[-1])
        return a
    
    def __add__(self, other):
        """Operator +"""
        return self.add(other)
    
    def __sub__(self, other):
        """Operator -"""
        return self.sub(other)
    
    def __mul__(self, other):
        """Operator *"""
        return self.mul(other)
    
    def __truediv__(self, other):
        """Operator / (returns quotient only)"""
        if isinstance(other, (int, float)):
            return self.times(1.0 / other)
        quotient, _ = self.div(other)
        return quotient
    
    def __neg__(self):
        """Operator -"""
        return self.negate()
    
    def __repr__(self) -> str:
        return f"Polynomial(degree={self.degree})"
    
    def __str__(self) -> str:
        """String representation of polynomial."""
        if self.is_zero():
            return "0"
        
        terms = []
        for i, coef in enumerate(self._coeffs):
            if coef != 0:
                if i == 0:
                    terms.append(f"{coef}")
                elif i == 1:
                    if coef == 1:
                        terms.append("x")
                    elif coef == -1:
                        terms.append("-x")
                    else:
                        terms.append(f"{coef}*x")
                else:
                    if coef == 1:
                        terms.append(f"x^{i}")
                    elif coef == -1:
                        terms.append(f"-x^{i}")
                    else:
                        terms.append(f"{coef}*x^{i}")
        
        return " + ".join(terms).replace(" + -", " - ")