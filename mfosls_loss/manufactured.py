"""1D manufactured solution of div(sigma_alpha(grad u)) = f, equation (33).

u(x) = sin(pi x) on (0, 1) vanishes at the endpoints, so it lies in H0^1(0, 1).
The flux is q = sigma_alpha(grad u) from (32), and f = div q is the derivative
of that flux. The source is not a separately chosen profile.
"""

import jax
import jax.numpy as jnp

from mfosls_loss.flux import sigma_alpha


def manufactured_solution(x, alpha):
    """Point values of the exact triple and of f = div q on the nodes ``x``."""
    alpha = jnp.asarray(alpha)

    def u_fn(t):
        return jnp.sin(jnp.pi * t)

    def p_fn(t):
        return jax.grad(u_fn)(t)

    def q_fn(t):
        return sigma_alpha(p_fn(t), alpha)

    def div_q_fn(t):
        return jax.grad(q_fn)(t)

    u = jax.vmap(u_fn)(x)
    grad_u = jax.vmap(p_fn)(x)
    q = jax.vmap(q_fn)(x)
    div_q = jax.vmap(div_q_fn)(x)
    return {
        "u": u,
        "p": grad_u,
        "q": q,
        "grad_u": grad_u,
        "div_q": div_q,
        "f": div_q,
        "alpha": alpha,
    }


def with_u_perturbation(fields, x, eps):
    """Add eps * sin(2 pi x) to u and update grad u. p, q, and f stay fixed."""
    eps = jnp.asarray(eps)

    def bump(t):
        return eps * jnp.sin(2.0 * jnp.pi * t)

    out = dict(fields)
    out["u"] = fields["u"] + jax.vmap(bump)(x)
    out["grad_u"] = fields["grad_u"] + jax.vmap(jax.grad(bump))(x)
    return out


def with_q_perturbation(fields, x, eps):
    """Add eps * sin(pi x) to q and update div q. u, p, and f stay fixed."""
    eps = jnp.asarray(eps)

    def bump(t):
        return eps * jnp.sin(jnp.pi * t)

    out = dict(fields)
    out["q"] = fields["q"] + jax.vmap(bump)(x)
    out["div_q"] = fields["div_q"] + jax.vmap(jax.grad(bump))(x)
    return out


def with_p_perturbation(fields, x, eps):
    """Add eps * sin(pi x) to p. No derivative of p enters (43c) or (46)."""
    eps = jnp.asarray(eps)
    out = dict(fields)
    out["p"] = fields["p"] + eps * jnp.sin(jnp.pi * x)
    return out


def with_combined_perturbation(fields, x, eps):
    """Deterministic perturbation of u, q, and p together."""
    out = with_u_perturbation(fields, x, eps)
    out = with_q_perturbation(out, x, -0.5 * eps)
    out = with_p_perturbation(out, x, 0.25 * eps)
    return out
