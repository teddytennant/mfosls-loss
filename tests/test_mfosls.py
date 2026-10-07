"""CPU checks of the Section 6.1 flux and the MFOSLS loss (46)."""

import math

import jax
import jax.numpy as jnp
import pytest

from mfosls_loss import (
    BETA_1,
    BETA_2,
    gauss_legendre,
    integrate,
    manufactured_solution,
    mfosls_loss,
    parameter_weights,
    phi,
    sigma_alpha,
    trial_norm_sq,
    with_combined_perturbation,
    with_p_perturbation,
    with_q_perturbation,
    with_u_perturbation,
)

ALPHA = 4.5
KAPPA = 1.0
N_QUAD = 64


def _weights_q1():
    weights = parameter_weights(3.0, 6.0)
    assert weights["mu"] == 10
    assert weights["kappa_1"] == pytest.approx(1.0)
    assert weights["kappa_2"] == pytest.approx(10.0 / 3.0)
    return weights


def _nodes():
    return gauss_legendre(N_QUAD, 0.0, 1.0)


def _loss(fields, nodes_weights, coeff):
    nodes, weights = nodes_weights
    del nodes
    return mfosls_loss(
        f=fields["f"],
        q=fields["q"],
        u=fields["u"],
        p=fields["p"],
        div_q=fields["div_q"],
        grad_u=fields["grad_u"],
        alpha=fields["alpha"],
        weights=weights,
        kappa=KAPPA,
        kappa_1=coeff["kappa_1"],
        kappa_2=coeff["kappa_2"],
    )


def _error_norm_sq(fields, exact, weights, kappa_1):
    return trial_norm_sq(
        q=fields["q"] - exact["q"],
        u=fields["u"] - exact["u"],
        p=fields["p"] - exact["p"],
        div_q=fields["div_q"] - exact["div_q"],
        grad_u=fields["grad_u"] - exact["grad_u"],
        weights=weights,
        kappa_1=kappa_1,
    )


def test_phi_matches_example_4_2():
    t = jnp.linspace(0.0, 20.0, 9)
    expected = 2.0 + 1.0 / (1.0 + t)
    assert jnp.allclose(phi(t), expected)
    assert float(phi(0.0)) == pytest.approx(3.0)
    # Example 4.2: 2 < phi(t) <= 3 and 2 < phi(t) + t phi'(t) <= 3 for t >= 0.
    assert float(jnp.min(phi(t))) > 2.0
    assert float(jnp.max(phi(t))) <= 3.0 + 1e-15


def test_sigma_matches_equation_32():
    alpha = 4.5
    p = jnp.array([-1.5, -0.2, 0.0, 0.7, 3.0])
    expected = -(2.0 + 1.0 / (1.0 + jnp.abs(p))) / alpha * p
    assert jnp.allclose(sigma_alpha(p, alpha), expected)
    assert float(sigma_alpha(0.0, alpha)) == 0.0


def test_flux_monotone_bounds_from_lemma_4_4():
    # (51): (beta_1 / alpha) |q - r|^2 <= -(sigma(q) - sigma(r)) · (q - r)
    #       <= (beta_2 / alpha) |q - r|^2.
    alpha = 4.0
    samples = jnp.linspace(-2.0, 2.0, 9)
    for q in samples:
        for r in samples:
            diff = float(q - r)
            if diff == 0.0:
                continue
            gap = float(-(sigma_alpha(q, alpha) - sigma_alpha(r, alpha)) * (q - r))
            lower = (BETA_1 / alpha) * diff**2
            upper = (BETA_2 / alpha) * diff**2
            assert gap >= lower - 1e-12
            assert gap <= upper + 1e-12


def test_section_6_1_parameter_boxes():
    for n in (1, 2, 3):
        weights = parameter_weights(3.0 / n, 6.0 / n)
        assert weights["mu"] == 10
        assert weights["kappa_1"] == pytest.approx(n**2)
        assert weights["kappa_2"] == pytest.approx(10.0 * n / 3.0)


def test_quadrature_of_known_integrals():
    nodes, weights = _nodes()
    assert float(integrate(jnp.ones_like(nodes), weights)) == pytest.approx(1.0)
    assert float(integrate(jnp.sin(jnp.pi * nodes) ** 2, weights)) == pytest.approx(0.5)


def test_loss_coefficients_on_constant_fields():
    nodes, weights = gauss_legendre(4, 0.0, 1.0)
    zeros = jnp.zeros_like(nodes)
    ones = jnp.ones_like(nodes)
    kappa_1 = 1.0
    kappa_2 = 10.0 / 3.0
    # Closure only. kappa must not multiply this term.
    closure_only = mfosls_loss(
        f=zeros,
        q=ones,
        u=zeros,
        p=zeros,
        div_q=zeros,
        grad_u=zeros,
        alpha=1.0,
        weights=weights,
        kappa=2.0,
        kappa_1=kappa_1,
        kappa_2=kappa_2,
    )
    assert float(closure_only) == pytest.approx(109.0 / 9.0)
    # Divergence residual only: L = kappa * kappa_1^{-2} ||f||^2.
    div_only = mfosls_loss(
        f=ones,
        q=zeros,
        u=zeros,
        p=zeros,
        div_q=zeros,
        grad_u=zeros,
        alpha=1.0,
        weights=weights,
        kappa=3.0,
        kappa_1=2.0,
        kappa_2=kappa_2,
    )
    assert float(div_only) == pytest.approx(0.75)
    # Gradient residual only: L = kappa ||p - grad u||^2.
    grad_only = mfosls_loss(
        f=zeros,
        q=sigma_alpha(ones, 1.0),
        u=zeros,
        p=ones,
        div_q=zeros,
        grad_u=zeros,
        alpha=1.0,
        weights=weights,
        kappa=3.0,
        kappa_1=2.0,
        kappa_2=kappa_2,
    )
    assert float(grad_only) == pytest.approx(3.0)


def test_trial_norm_groups_flux_terms():
    nodes, weights = gauss_legendre(4, 0.0, 1.0)
    zeros = jnp.zeros_like(nodes)
    ones = jnp.ones_like(nodes)
    grouped = trial_norm_sq(
        q=ones,
        u=zeros,
        p=zeros,
        div_q=ones,
        grad_u=zeros,
        weights=weights,
        kappa_1=2.0,
    )
    # kappa_1^{-2} multiplies both ||q||^2 and ||div q||^2, equation (43c).
    assert float(grouped) == pytest.approx(0.5)
    state = trial_norm_sq(
        q=zeros,
        u=ones,
        p=ones,
        div_q=zeros,
        grad_u=ones,
        weights=weights,
        kappa_1=2.0,
    )
    assert float(state) == pytest.approx(3.0)


def test_exact_triple_has_zero_loss():
    nodes_weights = _nodes()
    nodes, _weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    assert float(jnp.max(jnp.abs(exact["p"] - exact["grad_u"]))) == pytest.approx(0.0)
    assert float(jnp.max(jnp.abs(exact["q"] - sigma_alpha(exact["p"], ALPHA)))) < 1e-12
    loss = _loss(exact, nodes_weights, _weights_q1())
    assert abs(float(loss)) < 1e-18


def test_perturbation_has_positive_loss():
    nodes_weights = _nodes()
    nodes, _weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    perturbed = with_u_perturbation(exact, nodes, 0.1)
    loss = _loss(perturbed, nodes_weights, _weights_q1())
    assert float(loss) > 1e-6


def test_ratio_positive_and_finite_on_perturbations():
    nodes_weights = _nodes()
    nodes, weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    coeff = _weights_q1()
    builders = (
        with_u_perturbation,
        with_q_perturbation,
        with_p_perturbation,
        with_combined_perturbation,
    )
    amplitudes = (1e-3, 0.05, -0.2)
    for build in builders:
        for eps in amplitudes:
            fields = build(exact, nodes, eps)
            loss = float(_loss(fields, nodes_weights, coeff))
            denom = float(_error_norm_sq(fields, exact, weights, coeff["kappa_1"]))
            assert denom > 0.0
            ratio = loss / denom
            assert math.isfinite(ratio)
            assert ratio > 0.0


def test_u_perturbation_ratio_matches_direct_integration():
    """Implementation check, not a reliability constant from Theorem 4.12."""
    nodes_weights = _nodes()
    nodes, weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    coeff = _weights_q1()
    eps = 0.1
    fields = with_u_perturbation(exact, nodes, eps)
    loss = float(_loss(fields, nodes_weights, coeff))
    denom = float(_error_norm_sq(fields, exact, weights, coeff["kappa_1"]))
    # ||eps sin(2 pi x)||^2 = eps^2 / 2
    # ||eps * 2 pi cos(2 pi x)||^2 = 2 pi^2 eps^2
    # L = kappa * that gradient residual, and the other residuals vanish.
    expected = KAPPA * (2.0 * math.pi**2) / (0.5 + 2.0 * math.pi**2)
    assert loss / denom == pytest.approx(expected, rel=1e-12)


def test_q_perturbation_ratio_matches_direct_integration():
    """Checks that (43c) weights both ||q||^2 and ||div q||^2 by kappa_1^{-2}."""
    nodes_weights = _nodes()
    nodes, weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    coeff = _weights_q1()
    eps = 0.2
    fields = with_q_perturbation(exact, nodes, eps)
    loss = float(_loss(fields, nodes_weights, coeff))
    denom = float(_error_norm_sq(fields, exact, weights, coeff["kappa_1"]))
    kappa_1 = coeff["kappa_1"]
    kappa_2 = coeff["kappa_2"]
    closure_coefficient = kappa_1 ** (-4) * (kappa_1**2 + kappa_2**2)
    # ||sin(pi x)||^2 = 1/2, ||pi cos(pi x)||^2 = pi^2 / 2.
    numerator = KAPPA * kappa_1 ** (-2) * (math.pi**2 / 2.0) + closure_coefficient * 0.5
    denominator = kappa_1 ** (-2) * (0.5 + math.pi**2 / 2.0)
    assert loss / denom == pytest.approx(numerator / denominator, rel=1e-12)


def test_loss_scales_quadratically_for_a_linear_perturbation():
    nodes_weights = _nodes()
    nodes, _weights = nodes_weights
    exact = manufactured_solution(nodes, ALPHA)
    coeff = _weights_q1()
    small = _loss(with_u_perturbation(exact, nodes, 1e-3), nodes_weights, coeff)
    large = _loss(with_u_perturbation(exact, nodes, 2e-3), nodes_weights, coeff)
    assert float(large) / float(small) == pytest.approx(4.0, rel=1e-10)


def test_loss_is_jax_differentiable_in_amplitude():
    nodes, weights = _nodes()
    exact = manufactured_solution(nodes, ALPHA)
    coeff = _weights_q1()

    def loss_of(eps):
        fields = with_u_perturbation(exact, nodes, eps)
        return mfosls_loss(
            f=fields["f"],
            q=fields["q"],
            u=fields["u"],
            p=fields["p"],
            div_q=fields["div_q"],
            grad_u=fields["grad_u"],
            alpha=fields["alpha"],
            weights=weights,
            kappa=KAPPA,
            kappa_1=coeff["kappa_1"],
            kappa_2=coeff["kappa_2"],
        )

    derivative = jax.grad(loss_of)(0.1)
    assert math.isfinite(float(derivative))
    assert float(derivative) != 0.0
