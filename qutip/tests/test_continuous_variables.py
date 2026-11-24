import pytest
import numpy as np
import qutip


def _two_mode_squeezed_vacuum(r, N=40):
    """|psi> = sqrt(1 - x**2) * sum_n x**n |n, n> with x = tanh(r)."""
    x = np.tanh(r)
    psi = sum(
        x**n * qutip.tensor(qutip.basis(N, n), qutip.basis(N, n))
        for n in range(N)
    )
    return psi.unit()


class TestLogarithmicNegativity:
    # The logarithmic negativity of a two-mode squeezed vacuum with squeezing
    # r is 2 * r, whatever quadrature convention (g) the covariance matrix is
    # written in.

    @pytest.mark.parametrize("g", [np.sqrt(2), 1, 2, 0.5])
    @pytest.mark.parametrize("r", [0.25, 0.5, 1])
    def test_two_mode_squeezed_vacuum_analytic(self, r, g):
        c, s = np.cosh(2 * r), np.sinh(2 * r)
        Z = np.diag([1, -1])
        V = np.block([[c * np.eye(2), s * Z], [s * Z, c * np.eye(2)]]) / g**2
        logneg = qutip.logarithmic_negativity(V, g=g)
        np.testing.assert_allclose(logneg, 2 * r, atol=1e-12)

    # The vacuum sits exactly on the separability boundary, so only use
    # values of g for which the arithmetic is exact.
    @pytest.mark.parametrize("g", [1, 2, 0.5])
    def test_vacuum_is_separable(self, g):
        V = np.eye(4) / g**2
        assert qutip.logarithmic_negativity(V, g=g) == 0

    @pytest.mark.parametrize("g", [np.sqrt(2), 1, 2])
    @pytest.mark.parametrize("r", [0.25, 0.5])
    def test_two_mode_squeezed_vacuum_from_state(self, r, g):
        N = 40
        a1 = qutip.tensor(qutip.destroy(N), qutip.qeye(N))
        a2 = qutip.tensor(qutip.qeye(N), qutip.destroy(N))
        psi = _two_mode_squeezed_vacuum(r, N)
        V = qutip.wigner_covariance_matrix(a1, a2, rho=psi, g=g)
        logneg = qutip.logarithmic_negativity(V, g=g)
        np.testing.assert_allclose(logneg, 2 * r, atol=1e-8)
