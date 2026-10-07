"""MFOSLS loss for the monotone flux in arXiv:2610.05767.

The numerical profile and the mesh-free loss are equations (32), (48a), and (46).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from mfosls_loss.flux import (
    BETA_1,
    BETA_2,
    parameter_weights,
    phi,
    phi_alpha,
    sigma_alpha,
)
from mfosls_loss.loss import mfosls_loss, trial_norm_sq
from mfosls_loss.manufactured import (
    manufactured_solution,
    with_combined_perturbation,
    with_p_perturbation,
    with_q_perturbation,
    with_u_perturbation,
)
from mfosls_loss.quadrature import gauss_legendre, integrate

__all__ = [
    "BETA_1",
    "BETA_2",
    "gauss_legendre",
    "integrate",
    "manufactured_solution",
    "mfosls_loss",
    "parameter_weights",
    "phi",
    "phi_alpha",
    "sigma_alpha",
    "trial_norm_sq",
    "with_combined_perturbation",
    "with_p_perturbation",
    "with_q_perturbation",
    "with_u_perturbation",
]
