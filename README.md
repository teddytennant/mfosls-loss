A small JAX evaluation of the mesh-free first-order least-squares (MFOSLS) loss from https://arxiv.org/abs/2610.05767.

Install:

    pip install -e ".[test]"

Run:

    JAX_PLATFORMS=cpu python -m pytest -q
    JAX_PLATFORMS=cpu python demo.py

What does not match the paper

The integrals are 1D Gauss quadrature on (0, 1), not the paper's Raviart-Thomas and continuous finite-element spaces on (0, 1)^2. MDPG, broken test spaces, Fortin operators, and the Section 6 parameter-to-solution network are omitted. The two-sided constants in Theorem 4.12 were not checked. They depend on the Friedrichs constant in (57a), which the paper does not evaluate numerically. The demo penalty kappa is 1, not the Section 6.1 choice (1 + 2^{-20}) kappa_0, because that threshold is not numerically available without the Friedrichs constant.
