"""Mesh-free first-order least-squares loss, equation (46).

L(alpha, z) = kappa * (kappa_1^{-2} ||f - div q||^2 + ||p - grad u||^2)
    + kappa_1^{-4} (kappa_1^2 + kappa_2^2) ||q - sigma_alpha(p)||^2
for z = (q, u, p). Every term is an L2 integral. No mesh, broken space, or
interface variable enters (43a)-(46).

The trial norm is (43c):
||z||^2 = ||u||^2 + ||grad u||^2 + ||p||^2 + kappa_1^{-2} (||q||^2 + ||div q||^2).
"""

import jax.numpy as jnp

from mfosls_loss.flux import sigma_alpha
from mfosls_loss.quadrature import integrate


def mfosls_loss(
    f,
    q,
    u,
    p,
    div_q,
    grad_u,
    alpha,
    weights,
    kappa,
    kappa_1,
    kappa_2,
):
    """Quadrature of the MFOSLS loss (46).

    Field arguments are point values on the quadrature nodes. ``u`` is accepted
    so the call site carries the full triple z = (q, u, p); the loss itself
    depends on u only through grad u, as in (44) and (46). ``alpha`` is a
    positive scalar or point values. ``kappa``, ``kappa_1``, and ``kappa_2``
    are the positive coefficients in (46) and (56a).
    """
    del u  # conformity data; residual uses grad_u, matching (46)
    f = jnp.asarray(f)
    q = jnp.asarray(q)
    p = jnp.asarray(p)
    div_q = jnp.asarray(div_q)
    grad_u = jnp.asarray(grad_u)
    kappa_1 = jnp.asarray(kappa_1)
    kappa_2 = jnp.asarray(kappa_2)
    kappa = jnp.asarray(kappa)
    closure = q - sigma_alpha(p, alpha)
    div_residual = f - div_q
    grad_residual = p - grad_u
    residual_sq = kappa_1 ** (-2) * integrate(div_residual**2, weights) + integrate(
        grad_residual**2, weights
    )
    closure_sq = integrate(closure**2, weights)
    closure_coefficient = kappa_1 ** (-4) * (kappa_1**2 + kappa_2**2)
    return kappa * residual_sq + closure_coefficient * closure_sq


def trial_norm_sq(q, u, p, div_q, grad_u, weights, kappa_1):
    """Squared trial norm (43c) of z = (q, u, p), by the same quadrature."""
    q = jnp.asarray(q)
    u = jnp.asarray(u)
    p = jnp.asarray(p)
    div_q = jnp.asarray(div_q)
    grad_u = jnp.asarray(grad_u)
    kappa_1 = jnp.asarray(kappa_1)
    flux_sq = integrate(q**2, weights) + integrate(div_q**2, weights)
    return (
        integrate(u**2, weights)
        + integrate(grad_u**2, weights)
        + integrate(p**2, weights)
        + kappa_1 ** (-2) * flux_sq
    )
