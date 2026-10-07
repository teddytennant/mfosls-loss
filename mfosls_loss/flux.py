"""Monotone flux used in Section 6.1 of arXiv:2610.05767.

Equation (32) sets sigma_alpha(q) = -phi_alpha(|q|) q.
Equation (48a) sets phi_alpha = alpha^{-1} phi.
Example 4.2 and Section 6.1 use phi(t) = 2 + (1 + t)^{-1}, so beta_1 = 2 and beta_2 = 3.
In one dimension the Euclidean length |q| is the absolute value.
"""

import math

import jax.numpy as jnp

BETA_1 = 2.0
BETA_2 = 3.0


def phi(t):
    """Profile phi(t) = 2 + (1 + t)^{-1} from Example 4.2 and Section 6.1."""
    t = jnp.asarray(t)
    return 2.0 + 1.0 / (1.0 + t)


def phi_alpha(t, alpha):
    """phi_alpha = alpha^{-1} phi, equation (48a)."""
    alpha = jnp.asarray(alpha)
    return phi(t) / alpha


def sigma_alpha(p, alpha):
    """sigma_alpha(p) = -phi_alpha(|p|) p, equation (32) applied to the gradient variable p."""
    p = jnp.asarray(p)
    alpha = jnp.asarray(alpha)
    return -phi_alpha(jnp.abs(p), alpha) * p


def parameter_weights(alpha_1, alpha_2, beta_1=BETA_1, beta_2=BETA_2):
    """Return mu, kappa_1, kappa_2 from (56a) for a parameter interval [alpha_1, alpha_2].

    Section 6.1 takes mu to be the smallest integer strictly larger than
    (beta_2 alpha_2 / (beta_1 alpha_1))^2.
    """
    alpha_1 = float(alpha_1)
    alpha_2 = float(alpha_2)
    beta_1 = float(beta_1)
    beta_2 = float(beta_2)
    if alpha_1 <= 0.0 or alpha_2 < alpha_1:
        raise ValueError("need 0 < alpha_1 <= alpha_2")
    if beta_1 <= 0.0 or beta_2 < beta_1:
        raise ValueError("need 0 < beta_1 <= beta_2")
    ratio = (beta_2 * alpha_2 / (beta_1 * alpha_1)) ** 2
    mu = _smallest_integer_strictly_larger(ratio)
    kappa_1 = (beta_2 / alpha_1) ** 2
    kappa_2 = mu * beta_1 / alpha_2
    return {"mu": mu, "kappa_1": kappa_1, "kappa_2": kappa_2}


def _smallest_integer_strictly_larger(value):
    nearest = round(value)
    if abs(value - nearest) <= 1e-8 * max(1.0, abs(value)):
        value = float(nearest)
    return int(math.floor(value) + 1)
