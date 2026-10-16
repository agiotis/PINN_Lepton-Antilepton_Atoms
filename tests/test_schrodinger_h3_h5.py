"""Approved diagonal-triplet H3-H5 checks; no optimization or experiment data."""

import math
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
from scipy.integrate import quad

from quantum_solver.neural_network import NeuralNetwork
from quantum_solver.quantum_system import Particle
from tests.test_schrodinger_operator_algebra import (
    ExpLaurent, fixture_particle, reference_corrections, trial_patches, triplet_angular_reference,
)


class TripletCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.p = fixture_particle()
        patcher = mock.patch("quantum_solver.quantum_system.fundamental_values", self.p.fundamental_values)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertClose(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)

    def apply(self, index, r=0.8):
        return self.p.breitcoef(r, [], [int(i == index) for i in range(5)])

    def test_independent_spin_matrix_and_tensor_angular_values(self):
        for l, ls, tensor in ((0, 0, 0), (1, -0.5, 0.5), (2, -1.5, -0.5)):
            with self.subTest(l=l):
                self.assertClose(triplet_angular_reference(l), (ls, ls, 0.25, tensor))

    def test_h3_equal_unequal_mass_coefficients_and_exchange(self):
        p = self.p
        p.fundamental_values.hbarc = 2.0
        p.fundamental_values.e_charge = 3.0
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0):
            for m1, m2, expected in ((2, 2, -4.5), (2, 3, -37 / 12), (3, 2, -37 / 12)):
                p.mass1, p.mass2 = float(m1), float(m2)
                for r in (0.25, 1.0, 1.4):
                    with self.subTest(masses=(m1, m2), r=r):
                        self.assertClose(self.apply(2, r), expected / r ** 3)
            # H3 must not consult the incompatible individual-j assignments.
            p.angular_momentum1 = p.angular_momentum2 = float("nan")
            self.assertClose(self.apply(2, 1.0), -37 / 12)

    def test_h3_zero_for_both_s_states(self):
        p = self.p
        p.azimuthal_quant_num = 0
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.3):
            for n in (1, 2):
                p.primary_quant_num = n
                for masses in ((2.0, 2.0), (2.0, 3.0)):
                    p.mass1, p.mass2 = masses
                    for r in (0.2, 0.8, 1.5):
                        with self.subTest(n=n, masses=masses, r=r):
                            self.assertEqual(self.apply(2, r), 0)

    def test_h4_positive_darwin_coefficient(self):
        p = self.p
        p.fundamental_values.hbarc = 2.0
        p.fundamental_values.e_charge = 3.0
        # A raw kernel value 2*pi represents a locally normalized value of one.
        # Pauli Darwin: 4*pi*g times sum((h/M)^2)/8 = 13*pi/6.
        with mock.patch.object(p, "deltafunc", return_value=2 * math.pi), \
                mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0):
            self.assertClose(self.apply(3), 13 * math.pi / 6)
            self.assertGreater(self.apply(3), 0)

    def test_h5_contact_and_tensor_signs_separately(self):
        p = self.p
        p.fundamental_values.hbarc = 2.0
        p.fundamental_values.e_charge = 3.0
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0):
            for l, tensor_at_one in ((0, 0), (1, 1)):
                p.azimuthal_quant_num = l
                with mock.patch.object(p, "deltafunc", return_value=0.0):
                    tensor = self.apply(4, 1.0)
                with mock.patch.object(p, "deltafunc", return_value=2 * math.pi):
                    combined = self.apply(4, 1.0)
                with self.subTest(l=l):
                    self.assertClose(tensor, tensor_at_one)
                    self.assertClose(combined - tensor, 4 * math.pi / 3)
                    self.assertClose(combined, 4 * math.pi / 3 + tensor_at_one)

    def test_h4_local_full_space_normalization_and_global_kernel_preserved(self):
        p = self.p
        p.fundamental_values.hbarc = 2.0
        p.fundamental_values.e_charge = 3.0
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0):
            for kappa in (0.2, 1.0, 7.5):
                p.alpha0 = 10 / kappa
                # x=kappa*r maps the entire positive radial domain, not the grid.
                raw_norm = quad(lambda x: 4 * math.pi * (x / kappa) ** 2
                                * p.deltafunc(kappa, x / kappa) / kappa, 0, math.inf)[0]
                local_norm = quad(lambda x: 4 * math.pi * (x / kappa) ** 2
                                  * self.apply(3, x / kappa) / (13 * math.pi / 6) / kappa,
                                  0, math.inf)[0]
                with self.subTest(kappa=kappa):
                    self.assertClose(raw_norm, 2 * math.pi)
                    self.assertClose(local_norm, 1)

    def test_contact_kernel_shape_origin_and_large_argument(self):
        for kappa in (0.2, 1.0, 7.5):
            for x in (0.01, 0.3, 2.0):
                r = x / kappa
                expected = kappa / (2 * r ** 2 * math.cosh(x) ** 2)
                self.assertClose(self.p.deltafunc(kappa, r), expected)
            r = 1e-6 / kappa
            self.assertClose(r * r * self.p.deltafunc(kappa, r), kappa / 2)
            with np.errstate(over="raise", invalid="raise"):
                self.assertEqual(self.p.deltafunc(kappa, 1000 / kappa), 0)

    def test_frozen_target_functions_one_hot_and_combined(self):
        # Unnormalized regular 1s, 2s (including its node), and 2p fixtures.
        fixtures = ((1, 0, ExpLaurent({1: 1}, 0.7)),
                    (2, 0, ExpLaurent({1: 1, 2: -0.35}, 0.35)),
                    (2, 1, ExpLaurent({2: 1}, 0.35)))
        for n, l, trial in fixtures:
            self.p.primary_quant_num = n
            self.p.azimuthal_quant_num = l
            with ExitStack() as stack:
                trial_patches(stack, trial)
                for r in (0.2, 0.8, 1.5):
                    expected = reference_corrections(self.p, trial, r)
                    for i in (2, 3, 4):
                        with self.subTest(n=n, l=l, r=r, term=i + 1):
                            self.assertClose(self.apply(i, r), expected[i])
                    self.assertClose(self.p.breitcoef(r, [], [0, 0, 1, 1, 1]), sum(expected[2:]))

    def test_linear_coupling_and_mass_exchange(self):
        p = self.p
        original_g = p.fundamental_values.e_charge
        with ExitStack() as stack:
            trial_patches(stack, ExpLaurent({2: 1.1}, 0.6))
            for i in (2, 3, 4):
                baseline = self.apply(i)
                self.assertGreater(abs(baseline), 1e-8)
                for scale in (0, 0.4, 1.7):
                    p.fundamental_values.e_charge = original_g * scale
                    self.assertClose(self.apply(i), baseline * scale)
                p.fundamental_values.e_charge = original_g
                p.mass1, p.mass2 = p.mass2, p.mass1
                self.assertClose(self.apply(i), baseline)
                p.mass1, p.mass2 = p.mass2, p.mass1

    def test_energy_length_units_and_fm_pm_covariance(self):
        p = self.p
        constants = p.fundamental_values
        originals = (constants.hbarc, constants.e_charge, p.mass1, p.mass2, p.alpha0)
        scale = 1e-3  # fm to pm.
        for l in (0, 1):
            p.azimuthal_quant_num = l
            trial = ExpLaurent({l + 1: 1.1}, 0.6)
            converted = ExpLaurent({power: value / scale ** power for power, value in trial.terms.items()},
                                   trial.beta / scale)
            for i in (2, 3, 4):
                constants.hbarc, constants.e_charge, p.mass1, p.mass2, p.alpha0 = originals
                with ExitStack() as stack:
                    trial_patches(stack, trial)
                    baseline = self.apply(i)
                    # A change of energy unit must scale H_i U by one energy power.
                    constants.hbarc *= 4
                    constants.e_charge *= 4
                    p.mass1 *= 4
                    p.mass2 *= 4
                    self.assertClose(self.apply(i), 4 * baseline)
                constants.hbarc, constants.e_charge, p.mass1, p.mass2, p.alpha0 = originals
                constants.hbarc *= scale
                constants.e_charge *= scale
                p.alpha0 *= scale
                with ExitStack() as stack:
                    trial_patches(stack, converted)
                    with self.subTest(l=l, term=i + 1):
                        self.assertClose(self.apply(i, 0.8 * scale), baseline)

    def test_disabled_contact_not_called_by_h3_and_s_wave_contact_remains(self):
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0), \
                mock.patch.object(self.p, "deltafunc", side_effect=AssertionError("disabled contact")):
            self.assertTrue(math.isfinite(self.apply(2)))
        self.p.azimuthal_quant_num = 0
        with mock.patch.object(NeuralNetwork, "arfunc", return_value=1.0):
            self.assertGreater(self.apply(4), 0)
            # Changing the contact width cannot change the s-wave tensor (zero).
            with mock.patch.object(self.p, "deltafunc", return_value=0.0):
                self.assertEqual(self.apply(4), 0)


if __name__ == "__main__":
    unittest.main()
