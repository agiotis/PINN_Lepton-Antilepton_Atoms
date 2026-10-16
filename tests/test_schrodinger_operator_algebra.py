"""Deterministic algebra and diagonal triplet Hamiltonian coefficient checks."""

import itertools
import math
import unittest
from contextlib import ExitStack
from decimal import Decimal, localcontext
from functools import lru_cache
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy.polynomial import Polynomial

from quantum_solver.neural_network import NeuralNetwork
from quantum_solver.quantum_system import Particle


TRIAL_METHODS = (
    "arfunc", "firderarfunc", "secderarfunc", "thirderarfunc", "fourderarfunc",
)
NETWORK_METHODS = (
    "neuralfunc", "firderneuralfunc", "secderneuralfunc",
    "thirderneuralfunc", "fourderneuralfunc",
)


class ExpLaurent:
    """Independent analytic fixture exp(-beta*r) * sum(c_p*r**p)."""

    def __init__(self, terms, beta):
        self.terms = dict(terms)
        self.beta = beta

    def __call__(self, r):
        return math.exp(-self.beta * r) * sum(c * r ** p for p, c in self.terms.items())

    def derivative(self, order=1):
        result = self
        for _ in range(order):
            terms = {}
            for p, c in result.terms.items():
                if p:
                    terms[p - 1] = terms.get(p - 1, 0) + p * c
                terms[p] = terms.get(p, 0) - result.beta * c
            result = ExpLaurent(terms, self.beta)
        return result

    def shift(self, power, scale=1):
        return ExpLaurent({p + power: scale * c for p, c in self.terms.items()}, self.beta)

    def __add__(self, other):
        assert self.beta == other.beta
        terms = self.terms.copy()
        for p, c in other.terms.items():
            terms[p] = terms.get(p, 0) + c
        return ExpLaurent(terms, self.beta)


def radial_laplacian(radial, l):
    """Apply the unreduced R operator, then transform results independently."""
    return (radial.derivative(2) + radial.derivative().shift(-1, 2)
            + radial.shift(-2, -l * (l + 1)))


def reference_activation(name, order, s):
    """Differentiate via polynomial recurrences, not expanded production formulas."""
    if name == "smplus":
        if order == 0:
            return math.log1p(math.exp(s))
        return reference_activation("sigmoid", order - 1, s)
    if name == "relu":
        if order == 0:
            return max(0, s)
        return float(s > 0) if order == 1 else 0.0  # Only away from kinks.
    x = Polynomial([0, 1])
    polynomial = x
    if name == "sigmoid":
        argument = 1 / (1 + math.exp(-s))
        chain = x * (1 - x)
    else:
        argument = math.tanh(s)
        chain = 1 - x * x
    for _ in range(order):
        polynomial = polynomial.deriv() * chain
    return polynomial(argument)


def fixture_particle(l=1, equal_masses=False):
    # Bypass constructors and their file output; these tests never solve or optimize.
    particle = Particle.__new__(Particle)
    particle.mass1 = 2.0
    particle.mass2 = 2.0 if equal_masses else 3.0
    particle.mass = particle.mass1 * particle.mass2 / (particle.mass1 + particle.mass2)
    particle.alpha0 = 2.3
    particle.azimuthal_quant_num = l
    particle.total_angular_momentum = 1
    particle.total_spin_momentum = 1
    particle.angular_momentum1 = l - 0.5
    particle.angular_momentum2 = l + 0.5
    particle.k_optimization_const = 0.7
    particle.num_hidden_units = 2
    particle.activation = "tanh"
    particle.fundamental_values = SimpleNamespace(hbarc=1.7, e_charge=0.8)
    particle.energy = -0.2
    particle.verbose = False
    return particle


def trial_patches(stack, trial):
    for order, name in enumerate(TRIAL_METHODS):
        derivative = trial.derivative(order)
        stack.enter_context(mock.patch.object(
            NeuralNetwork, name, side_effect=lambda r, *args, f=derivative: f(r)))


def kinetic_fourth_coefficient(masses, hbarc):
    """Extract the k^4 coefficient from exact relativistic kinetic energies.

    High precision avoids subtraction loss; Richardson extrapolation removes
    the leading k^2 error after division by k^4. No production prefactor is used.
    """
    with localcontext() as context:
        context.prec = 60
        h = Decimal(str(hbarc))
        energies = [Decimal(str(mass)) for mass in masses]

        def quotient(k):
            q2 = (h * k) ** 2
            remainder = sum((mass * mass + q2).sqrt() - mass - q2 / (2 * mass)
                            for mass in energies)
            return remainder / k ** 4

        k = Decimal("0.0001")
        return float((4 * quotient(k / 2) - quotient(k)) / 3)


@lru_cache(maxsize=None)
def triplet_angular_reference(l):
    """Compute J=1,M=1 angular expectations using matrices and sphere quadrature.

    Spins are sigma/2. No scalar spin-orbit or tensor formula is used.
    The l=2 fixture retains existing operator-test coverage, not a new interface.
    """
    def angular_matrices(j):
        m = np.arange(j, -j - 1, -1)
        raising = np.zeros((len(m), len(m)), complex)
        for index in range(1, len(m)):
            raising[index - 1, index] = math.sqrt(j * (j + 1) - m[index] * (m[index] + 1))
        return ((raising + raising.T.conj()) / 2,
                (raising - raising.T.conj()) / (2j), np.diag(m))

    orbital = angular_matrices(l)
    total_spin = angular_matrices(1)
    individual_spin = angular_matrices(0.5)
    ls = sum(np.kron(a, b) for a, b in zip(orbital, total_spin))
    basis = [(ml, ms) for ml in range(l, -l - 1, -1) for ms in (1, 0, -1)]
    selected = [i for i, (ml, ms) in enumerate(basis) if ml + ms == 1]
    values, vectors = np.linalg.eigh(ls[np.ix_(selected, selected)])
    # J^2 = L^2 + S^2 + 2 L.S; select J=1 within the fixed-M subspace.
    column = np.argmin(abs(l * (l + 1) + 2 + 2 * values - 2))
    coefficients = np.zeros(len(basis), complex)
    coefficients[selected] = vectors[:, column]
    spinors = {1: np.array([1, 0, 0, 0]), 0: np.array([0, 1, 1, 0]) / math.sqrt(2),
               -1: np.array([0, 0, 0, 1])}
    state = np.concatenate([sum(coefficients[i] * spinors[ms]
                                   for i, (ml, ms) in enumerate(basis) if ml == m)
                            for m in range(l, -l - 1, -1)])
    dot = sum(np.kron(s, s) for s in individual_spin)
    ls1 = sum(np.kron(a, np.kron(b, np.eye(2))) for a, b in zip(orbital, individual_spin))
    ls2 = sum(np.kron(a, np.kron(np.eye(2), b)) for a, b in zip(orbital, individual_spin))
    s12 = np.vdot(state, np.kron(np.eye(2 * l + 1), dot) @ state).real

    def harmonic(m, x, phi):
        order = abs(m)
        legendre = np.polynomial.legendre.Legendre.basis(l).deriv(order)(x)
        value = ((-1) ** order * math.sqrt((2 * l + 1) / (4 * math.pi)
                                          * math.factorial(l - order) / math.factorial(l + order))
                 * (1 - x * x) ** (order / 2) * legendre * np.exp(1j * order * phi))
        return value if m >= 0 else (-1) ** order * value.conjugate()

    tensor = norm = 0.0
    nodes, weights = np.polynomial.legendre.leggauss(12)
    for x, weight in zip(nodes, weights):
        for phi in np.arange(24) * 2 * math.pi / 24:
            n = (math.sqrt(1 - x * x) * math.cos(phi), math.sqrt(1 - x * x) * math.sin(phi), x)
            sn = sum(a * b for a, b in zip(n, individual_spin))
            psi = sum(coefficients[i] * harmonic(ml, x, phi) * spinors[ms]
                      for i, (ml, ms) in enumerate(basis))
            measure = weight * 2 * math.pi / 24
            norm += measure * np.vdot(psi, psi).real
            tensor += measure * np.vdot(psi, (3 * np.kron(sn, sn) - dot) @ psi).real
    np.testing.assert_allclose(norm, 1, rtol=0, atol=1e-13)
    return (np.vdot(state, ls1 @ state).real, np.vdot(state, ls2 @ state).real, s12, tensor)


def reference_corrections(particle, trial, r):
    """Independent radial derivatives, spin matrices and normalized contact reference."""
    l = particle.azimuthal_quant_num
    m1, m2 = particle.mass1, particle.mass2
    h = particle.fundamental_values.hbarc
    g = particle.fundamental_values.e_charge
    c = 10 / particle.alpha0
    radial = trial.shift(-1)
    laplacian = radial_laplacian(radial, l)
    h1 = kinetic_fourth_coefficient((m1, m2), h)
    h1 *= radial_laplacian(laplacian, l).shift(1)(r)
    # Two momentum-to-velocity factors in the attractive orbit-orbit term.
    h2 = 0.5 * g * (h / m1) * (h / m2)
    h2 *= (laplacian + radial.derivative(2))(r)
    ls1, ls2, s12, tensor = triplet_angular_reference(l)
    w3 = g / r ** 3 * (0.5 * (h / m1) ** 2 * ls1 + 0.5 * (h / m2) ** 2 * ls2
                       + (h / m1) * (h / m2) * (ls1 + ls2))
    # Full-space normalized kernel, expressed through cosh independently.
    contact = c / (4 * math.pi * r ** 2 * math.cosh(c * r) ** 2)
    # Pauli Darwin term = sum((h/M)^2)/8 * Laplacian(-g/r).
    w4 = ((h / m1) ** 2 + (h / m2) ** 2) / 8 * (4 * math.pi * g * contact)
    w5 = g * (h / m1) * (h / m2) * (8 * math.pi / 3 * s12 * contact + tensor / r ** 3)
    return [h1, h2, w3 * trial(r), w4 * trial(r), w5 * trial(r)]


class SchrodingerOperatorAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.particle = fixture_particle()
        self.addCleanup(mock.patch.stopall)
        mock.patch("quantum_solver.quantum_system.fundamental_values",
                   self.particle.fundamental_values).start()

    def assertNumericallyEqual(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)

    def test_required_activation_witnesses(self):
        for name, expected in (("thirdersigmoid", -0.125), ("fourdersigmoid", 0),
                               ("thirdertanh", -2), ("fourdertanh", 0)):
            with self.subTest(name=name):
                self.assertEqual(getattr(NeuralNetwork, name)(0), expected)

    def test_activation_derivatives_against_recurrences(self):
        for name in ("sigmoid", "tanh", "smplus", "relu"):
            for s in (-1.3, -0.4, 0.6, 1.1):
                for order in range(5):
                    with self.subTest(name=name, s=s, order=order):
                        self.assertNumericallyEqual(
                            NeuralNetwork.activfunc(name, order)(s),
                            reference_activation(name, order, s))

    def test_trial_derivatives_with_polynomial_network(self):
        polynomial = Polynomial([0.3, -0.5, 0.7, 0.11, 0.05])
        with ExitStack() as stack:
            for order, name in enumerate(NETWORK_METHODS):
                f = polynomial.deriv(order)
                stack.enter_context(mock.patch.object(
                    NeuralNetwork, name, side_effect=lambda r, *args, f=f: f(r)))
            for k in (0, 0.3, 1.1):
                trial = ExpLaurent({p + 1: c for p, c in enumerate(polynomial.coef)}, k)
                for r in (0, 0.4, 1.7):
                    for order, name in enumerate(TRIAL_METHODS):
                        with self.subTest(k=k, r=r, order=order):
                            self.assertNumericallyEqual(
                                getattr(NeuralNetwork, name)(r, [], k, 1, "tanh"),
                                trial.derivative(order)(r))

    def test_trial_derivatives_with_actual_smooth_networks(self):
        weights = [0.7, -0.2, 1.1, -0.6, 0.3, -0.8]
        k = 0.7
        for activation in ("sigmoid", "tanh", "smplus"):
            for r in (0, 0.4, 1.2):
                n = [sum(weights[i] * weights[i + 2] ** degree * reference_activation(
                    activation, degree, weights[i + 2] * r + weights[i + 4])
                    for i in range(2)) for degree in range(5)]
                for order, name in enumerate(TRIAL_METHODS):
                    # Leibniz convolution with independently differentiated r*exp(-k*r).
                    expected = sum(math.comb(order, degree) * n[degree] * math.exp(-k * r)
                                   * (r * (-k) ** (order - degree)
                                      + ((order - degree) * (-k) ** (order - degree - 1)
                                         if order > degree else 0))
                                   for degree in range(order + 1))
                    with self.subTest(activation=activation, r=r, order=order):
                        self.assertNumericallyEqual(
                            getattr(NeuralNetwork, name)(r, weights, k, 2, activation), expected)

    def test_reduced_laplacian_and_square(self):
        for l in (0, 1, 2):
            self.particle.azimuthal_quant_num = l
            for power in (l + 1, 5):
                trial = ExpLaurent({power: 1.2}, 0.7)
                radial = trial.shift(-1)
                laplacian = radial_laplacian(radial, l)
                square = radial_laplacian(laplacian, l)
                with ExitStack() as stack:
                    trial_patches(stack, trial)
                    for r in (0.2, 0.8, 1.9):
                        with self.subTest(l=l, power=power, r=r):
                            self.assertNumericallyEqual(self.particle.radnablsq([], r), r * laplacian(r))
                            self.assertNumericallyEqual(self.particle.radnablfo([], r), r * square(r))

    def test_h2_reduced_expression(self):
        for l in (0, 1, 2):
            self.particle.azimuthal_quant_num = l
            trial = ExpLaurent({l + 1: 0.9, 5: -0.2}, 0.6)
            with ExitStack() as stack:
                trial_patches(stack, trial)
                for r in (0.25, 0.9, 1.4):
                    with self.subTest(l=l, r=r):
                        self.assertNumericallyEqual(
                            self.particle.breitcoef(r, [], [0, 1, 0, 0, 0]),
                            reference_corrections(self.particle, trial, r)[1])

    def test_h1_coefficient_from_relativistic_kinetic_expansion(self):
        p = self.particle
        for masses in ((2.0, 3.0), (2.0, 2.0), (0.5, 100.0)):
            p.mass1, p.mass2 = masses
            expected = kinetic_fourth_coefficient(masses, p.fundamental_values.hbarc)
            with mock.patch.object(p, "radnablfo", return_value=1.0):
                actual = p.breitcoef(0.8, [], [1, 0, 0, 0, 0])
            with self.subTest(masses=masses):
                self.assertLess(actual, 0)
                self.assertNumericallyEqual(actual, expected)

    def test_h2_positive_sign_from_cartesian_momentum_contraction(self):
        # With h=2, g=3, M1=2, M2=3, the coefficient is exactly one.
        # R=x^2+y^2+z^2 has Hessian 2*I. For r along z,
        # T_ij=delta_ij/r+r_i*r_j/r^3=diag(1,1,2)/r.
        # -g/(2*M1*M2) * P_i*T_ij*P_j gives +8/r on R,
        # since P=-i*h*D; multiplying by r gives +8 on U=r^3.
        p = self.particle
        p.fundamental_values.hbarc = 2.0
        p.fundamental_values.e_charge = 3.0
        p.azimuthal_quant_num = 0
        trial = ExpLaurent({3: 1.0}, 0.0)
        with ExitStack() as stack:
            trial_patches(stack, trial)
            for r in (0.25, 0.9, 1.4):
                with self.subTest(r=r):
                    self.assertNumericallyEqual(p.breitcoef(r, [], [0, 1, 0, 0, 0]), 8.0)

    def test_h1_h2_coefficient_dimensions(self):
        p = self.particle
        constants = p.fundamental_values
        original = (constants.hbarc, constants.e_charge, p.mass1, p.mass2)
        trial = ExpLaurent({2: 0.9, 5: -0.2}, 0.6)
        with ExitStack() as stack:
            trial_patches(stack, trial)
            for index, h_power, mass_power in ((0, 4, -3), (1, 2, -2)):
                mask = [int(i == index) for i in range(5)]
                baseline = p.breitcoef(0.8, [], mask)
                self.assertGreater(abs(baseline), 1e-8)
                # Numeric h has dimensions E*L, g has E*L, masses have E.
                constants.hbarc = original[0] * 1.3
                self.assertNumericallyEqual(p.breitcoef(0.8, [], mask), baseline * 1.3 ** h_power)
                constants.hbarc = original[0]
                p.mass1, p.mass2 = original[2] * 2, original[3] * 2
                self.assertNumericallyEqual(p.breitcoef(0.8, [], mask), baseline * 2 ** mass_power)
                p.mass1, p.mass2 = original[2:]
                # A change of energy unit scales h, g and both masses together.
                constants.hbarc, constants.e_charge = original[0] * 4, original[1] * 4
                p.mass1, p.mass2 = original[2] * 4, original[3] * 4
                self.assertNumericallyEqual(p.breitcoef(0.8, [], mask), baseline * 4)
                constants.hbarc, constants.e_charge, p.mass1, p.mass2 = original

    def test_h1_h2_fm_to_pm_unit_covariance(self):
        p = self.particle
        constants = p.fundamental_values
        h_fm, g_fm, a_fm = constants.hbarc, constants.e_charge, p.alpha0
        scale = 1e-3  # Numerical length in pm / numerical length in fm.
        for l in (0, 1, 2):
            p.azimuthal_quant_num = l
            trial_fm = ExpLaurent({l + 1: 0.9, 5: -0.2}, 0.6)
            # Hold unnormalized U fixed as a scalar field: U_pm(x)=U_fm(x/scale).
            trial_pm = ExpLaurent({power: value / scale ** power
                                   for power, value in trial_fm.terms.items()}, trial_fm.beta / scale)
            for r_fm in (0.25, 0.9, 1.4):
                for index in (0, 1):
                    mask = [int(i == index) for i in range(5)]
                    constants.hbarc, constants.e_charge, p.alpha0 = h_fm, g_fm, a_fm
                    with ExitStack() as stack:
                        trial_patches(stack, trial_fm)
                        in_fm = p.breitcoef(r_fm, [], mask)
                        h0_fm = p.hamiltonian([], r_fm)
                    constants.hbarc, constants.e_charge = h_fm * scale, g_fm * scale
                    p.alpha0 = a_fm * scale
                    with ExitStack() as stack:
                        trial_patches(stack, trial_pm)
                        with self.subTest(l=l, r=r_fm, term=index + 1):
                            self.assertNumericallyEqual(p.breitcoef(r_fm * scale, [], mask), in_fm)
                            self.assertNumericallyEqual(p.hamiltonian([], r_fm * scale), h0_fm)

    def test_h2_linear_coupling_and_h1_coupling_independence(self):
        p = self.particle
        original_g = p.fundamental_values.e_charge
        with ExitStack() as stack:
            trial_patches(stack, ExpLaurent({2: 0.9, 5: -0.2}, 0.6))
            h1 = p.breitcoef(0.8, [], [1, 0, 0, 0, 0])
            h2 = p.breitcoef(0.8, [], [0, 1, 0, 0, 0])
            self.assertGreater(abs(h2), 1e-8)
            for scale in (0, 0.4, 1.7):
                p.fundamental_values.e_charge = original_g * scale
                with self.subTest(scale=scale):
                    self.assertNumericallyEqual(p.breitcoef(0.8, [], [0, 1, 0, 0, 0]), scale * h2)
                    self.assertNumericallyEqual(p.breitcoef(0.8, [], [1, 0, 0, 0, 0]), h1)

    def test_h1_h2_are_independent_of_contact_regulator(self):
        p = self.particle
        with ExitStack() as stack:
            trial_patches(stack, ExpLaurent({2: 0.9, 5: -0.2}, 0.6))
            for mask in ([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0]):
                baseline = p.breitcoef(0.8, [], mask)
                for a0 in (0.4, 2.3, 9.1):
                    p.alpha0 = a0
                    with self.subTest(mask=mask, a0=a0):
                        self.assertNumericallyEqual(p.breitcoef(0.8, [], mask), baseline)

    def test_multiplication_operators_zero_scaling_and_superposition(self):
        for equal_masses in (False, True):
            self.particle = fixture_particle(equal_masses=equal_masses)
            first = ExpLaurent({2: 1.2}, 0.6)
            second = ExpLaurent({1: -0.4, 3: 0.2}, 0.6)
            combined = first.shift(0, 2.3) + second.shift(0, -0.7)
            for i in (2, 3, 4):
                mask = [int(j == i) for j in range(5)]
                for r in (0.2, 0.9, 1.6):
                    values = []
                    for trial in (ExpLaurent({}, 0.6), first, first.shift(0, -1.8), second, combined):
                        with ExitStack() as stack:
                            trial_patches(stack, trial)
                            values.append(self.particle.breitcoef(r, [], mask))
                    with self.subTest(equal_masses=equal_masses, term=i + 1, r=r):
                        self.assertEqual(values[0], 0)
                        self.assertNumericallyEqual(values[2], -1.8 * values[1])
                        self.assertNumericallyEqual(values[4], 2.3 * values[1] - 0.7 * values[3])

    def test_all_32_masks_against_independent_values_and_residuals(self):
        for equal_masses in (False, True):
            for l in (0, 1, 2):
                self.particle = fixture_particle(l, equal_masses)
                trial = ExpLaurent({l + 1: 0.9, 5: -0.2}, 0.6)
                self.particle.relative_search_points = 1
                with ExitStack() as stack:
                    trial_patches(stack, trial)
                    for r in (0.25, 0.9, 1.4):
                        terms = reference_corrections(self.particle, trial, r)
                        self.assertTrue(any(abs(value) > 1e-10 for value in terms))
                        h0 = (-self.particle.fundamental_values.hbarc ** 2 / (2 * self.particle.mass)
                              * trial.derivative(2)(r) + self.particle.coulomb_potential(r) * trial(r))
                        for mask in itertools.product((0, 1), repeat=5):
                            expected = sum(value for bit, value in zip(mask, terms) if bit)
                            self.particle.breit = list(mask)
                            with self.subTest(equal_masses=equal_masses, l=l, r=r, mask=mask):
                                self.assertNumericallyEqual(self.particle.breitcoef(r, [], mask), expected)
                                self.assertNumericallyEqual(
                                    self.particle.objective_function([], [r]),
                                    (h0 + expected - self.particle.energy * trial(r)) ** 2)

    def test_all_zero_mask_evaluates_no_corrections(self):
        self.particle.alpha0 = 0  # Even the correction regulator must not be evaluated.
        with ExitStack() as stack:
            for name in TRIAL_METHODS:
                stack.enter_context(mock.patch.object(NeuralNetwork, name, side_effect=AssertionError(name)))
            for name in ("radnablsq", "radnablfo", "deltafunc"):
                stack.enter_context(mock.patch.object(self.particle, name, side_effect=AssertionError(name)))
            self.assertEqual(self.particle.breitcoef(0, [], [0] * 5), 0)

    def test_disabled_term_exceptions_and_nonfinite_values_are_isolated(self):
        for poison in (float("nan"), float("inf"), AssertionError("disabled term evaluated")):
            kwargs = {"side_effect": poison} if isinstance(poison, Exception) else {"return_value": poison}
            for selected in range(5):
                with ExitStack() as stack:
                    trial_patches(stack, ExpLaurent({2: 1.2}, 0.6))
                    if selected != 0:
                        stack.enter_context(mock.patch.object(self.particle, "radnablfo", **kwargs))
                    if selected not in (3, 4):
                        stack.enter_context(mock.patch.object(self.particle, "deltafunc", **kwargs))
                    if selected in (2, 3, 4):
                        for name in TRIAL_METHODS[1:]:
                            stack.enter_context(mock.patch.object(NeuralNetwork, name, **kwargs))
                    mask = [int(i == selected) for i in range(5)]
                    with self.subTest(selected=selected, poison=str(poison)):
                        self.assertTrue(math.isfinite(self.particle.breitcoef(0.8, [], mask)))

    def test_h0_only_matches_existing_baseline_with_actual_network(self):
        p = self.particle
        weights = [0.7, -0.2, 1.1, -0.6, 0.3, -0.8]
        radii = [0.2, 0.8, 1.5]
        p.relative_search_points = len(radii)
        residuals = []
        for r in radii:
            u = NeuralNetwork.arfunc(r, weights, p.k_optimization_const, 2, p.activation)
            u2 = NeuralNetwork.secderarfunc(r, weights, p.k_optimization_const, 2, p.activation)
            potential = (-p.fundamental_values.e_charge / r
                         + p.fundamental_values.hbarc ** 2 * p.azimuthal_quant_num
                         * (p.azimuthal_quant_num + 1) / (2 * p.mass * r ** 2))
            expected_h0 = -p.fundamental_values.hbarc ** 2 * u2 / (2 * p.mass) + potential * u
            self.assertNumericallyEqual(p.hamiltonian(weights, r), expected_h0)
            residuals.append(expected_h0 - p.energy * u)
        p.breit = None
        old_baseline = p.objective_function(weights, radii)
        p.breit = [0] * 5
        with mock.patch.object(p, "radnablfo", side_effect=AssertionError("disabled H1")), \
                mock.patch.object(p, "deltafunc", side_effect=AssertionError("disabled contact")):
            masked_baseline = p.objective_function(weights, radii)
        self.assertEqual(masked_baseline, old_baseline)
        self.assertNumericallyEqual(masked_baseline, sum(value ** 2 for value in residuals))


if __name__ == "__main__":
    unittest.main()
