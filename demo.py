"""Evaluate the MFOSLS loss on a 1D manufactured solution and one perturbation."""

from mfosls_loss import (
    gauss_legendre,
    manufactured_solution,
    mfosls_loss,
    parameter_weights,
    with_u_perturbation,
)

# Q_1 = [3, 6] from Section 6.1. alpha = 4.5 lies in that interval.
# kappa = 1 is a positive penalty in (46). It is not (1 + 2^{-20}) kappa_0:
# kappa_0 in (12) needs the Friedrichs constant, which the paper does not evaluate.
ALPHA = 4.5
KAPPA = 1.0
EPS = 0.1


def _evaluate(fields, weights, coeff):
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


def main():
    coeff = parameter_weights(3.0, 6.0)
    nodes, weights = gauss_legendre(64, 0.0, 1.0)
    exact = manufactured_solution(nodes, ALPHA)
    perturbed = with_u_perturbation(exact, nodes, EPS)
    exact_loss = float(_evaluate(exact, weights, coeff))
    perturbed_loss = float(_evaluate(perturbed, weights, coeff))
    print(f"alpha = {ALPHA}")
    print(f"kappa = {KAPPA}")
    print(f"kappa_1 = {coeff['kappa_1']}")
    print(f"kappa_2 = {coeff['kappa_2']}")
    print(f"exact L = {exact_loss:.16e}")
    print(f"perturbed L = {perturbed_loss:.16e}")


if __name__ == "__main__":
    main()
