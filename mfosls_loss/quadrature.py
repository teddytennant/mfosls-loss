"""Gauss-Legendre quadrature on a 1D interval. Not a finite-element mesh."""

import numpy as np
import jax.numpy as jnp


def gauss_legendre(n, a=0.0, b=1.0):
    """Nodes and weights for the integral from a to b, mapped from numpy's Legendre rule."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not b > a:
        raise ValueError("need a < b")
    xi, wi = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    nodes = mid + half * xi
    weights = half * wi
    return jnp.asarray(nodes), jnp.asarray(weights)


def integrate(values, weights):
    """Quadrature of a scalar integrand given by point values."""
    values = jnp.asarray(values)
    weights = jnp.asarray(weights)
    if values.shape != weights.shape:
        raise ValueError(
            f"values shape {values.shape} does not match weights shape {weights.shape}"
        )
    return jnp.sum(weights * values)
