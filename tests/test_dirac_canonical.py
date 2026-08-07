from __future__ import annotations

import inspect
import os
import sys
import unittest

import numpy as np

from quantum_solver import dirac_canonical as dc
from quantum_solver.dirac_canonical_adapter import DiracStateMetadata, GroundStateMetadata, ShadowDiracAdapter


def fixed_networks_h2():
    f = np.array([0.8, -0.3, 0.02, -0.015, 0.1, -0.2], dtype=float)
    g = np.array([-0.4, 0.25, -0.01, 0.012, -0.3, 0.15], dtype=float)
    return f, g


def vectors_h2():
    f, g = fixed_networks_h2()
    fixed = np.concatenate((f, g))
    learnable = np.concatenate((f, [1.0], g, [1.0]))
    return fixed, learnable


class CanonicalDiracTests(unittest.TestCase):
    def setUp(self):
        self.r = np.linspace(0.1, 40.0, 401)
        self.fixed, self.learnable = vectors_h2()
        self.fixed_adapter = ShadowDiracAdapter(2, 0.03, dc.FIXED_UNIT_COEFFICIENTS)
        self.learnable_adapter = ShadowDiracAdapter(2, 0.03, dc.LEARNABLE_COEFFICIENTS)

    def test_01_trial_function_identity(self):
        f, _ = fixed_networks_h2()
        r = np.array([0.0, 0.1, 1.0, 3.0])
        value, _ = dc.reduced_component(r, f, 2, 0.4, -0.7)
        a, b, d = f[:2], f[2:4], f[4:6]
        expected = -0.7 * r * np.exp(-0.4 * r) * np.sum(a * np.tanh(r[:, None] * b + d), axis=1)
        np.testing.assert_allclose(value, expected, rtol=1e-14, atol=1e-15)
        self.assertEqual(value[0], 0.0)

    def test_02_derivative_finite_difference(self):
        f, _ = fixed_networks_h2()
        r0, step = 0.7, 1e-5
        plus = dc.reduced_component(np.array([r0 + step]), f, 2, 0.4, -0.7)[0][0]
        minus = dc.reduced_component(np.array([r0 - step]), f, 2, 0.4, -0.7)[0][0]
        analytic = dc.reduced_component(np.array([r0]), f, 2, 0.4, -0.7)[1][0]
        self.assertAlmostEqual(analytic, (plus - minus) / (2 * step), delta=2e-10)

    def test_03_derivative_at_origin(self):
        f, _ = fixed_networks_h2()
        coefficient = -0.7
        _, derivative = dc.reduced_component(np.array([0.0]), f, 2, 0.4, coefficient)
        neural, _ = dc.neural_value_and_derivative(np.array([0.0]), f, 2)
        self.assertAlmostEqual(derivative[0], coefficient * neural[0], delta=1e-15)

    def test_04_reduced_nonreduced_transformation(self):
        r = self.r
        f = np.exp(-0.02 * r) * (1 + 0.001 * r)
        fp = np.exp(-0.02 * r) * (0.001 - 0.02 * (1 + 0.001 * r))
        g = -0.01 * np.exp(-0.03 * r) * (1 - 0.0005 * r)
        gp = -0.01 * np.exp(-0.03 * r) * (-0.0005 - 0.03 * (1 - 0.0005 * r))
        F, G = r * f, r * g
        dF, dG = f + r * fp, g + r * gp
        V = dc.pure_coulomb_potential(r)
        RF, RG = dc.coupled_residuals(r, F, G, dF, dG, -1, dc.ELECTRON_REST_ENERGY_MEV,
                                      dc.GROUND_REFERENCE_ENERGY_MEV, potential_mev=V)
        rf_non = fp + (1 - 1) * f / r - (dc.ELECTRON_REST_ENERGY_MEV + dc.GROUND_REFERENCE_ENERGY_MEV - V) * g / dc.HBARC_MEV_FM
        rg_non = gp + (1 + 1) * g / r - (dc.ELECTRON_REST_ENERGY_MEV - dc.GROUND_REFERENCE_ENERGY_MEV + V) * f / dc.HBARC_MEV_FM
        np.testing.assert_allclose(RF / r, rf_non, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(RG / r, rg_non, rtol=1e-12, atol=1e-13)

    def test_05_kappa_minus_one_angular_terms(self):
        F = self.r * np.exp(-0.02 * self.r)
        G = -0.01 * F
        zeros = np.zeros_like(F)
        V = dc.pure_coulomb_potential(self.r)
        RF, RG = dc.coupled_residuals(self.r, F, G, zeros, zeros, -1, 0.0, 0.0,
                                      hbarc_mev_fm=1.0, potential_mev=np.zeros_like(V))
        np.testing.assert_allclose(RF, -F / self.r)
        np.testing.assert_allclose(RG, G / self.r)

    def test_06_coulomb_sign_and_units(self):
        r = np.array([1.0, 10.0, 100.0])
        expected = -dc.E_SQUARED_MEV_FM / r
        np.testing.assert_allclose(dc.pure_coulomb_potential(r), expected, rtol=0, atol=0)
        alpha_v = dc.E_SQUARED_MEV_FM / dc.HBARC_MEV_FM
        np.testing.assert_allclose(expected, -alpha_v * dc.HBARC_MEV_FM / r, rtol=1e-15)
        self.assertTrue(np.all(expected < 0))

    def test_07_first_residual_sign(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        RF, _ = dc.coupled_residuals(self.r, spinor["F"], spinor["G"], spinor["dF"], spinor["dG"], -1,
                                     dc.ELECTRON_REST_ENERGY_MEV, dc.GROUND_REFERENCE_ENERGY_MEV, potential_mev=V)
        manual = spinor["dF"] - spinor["F"] / self.r - (
            dc.ELECTRON_REST_ENERGY_MEV + dc.GROUND_REFERENCE_ENERGY_MEV - V) * spinor["G"] / dc.HBARC_MEV_FM
        np.testing.assert_allclose(RF, manual, rtol=1e-14, atol=1e-15)

    def test_08_second_residual_sign(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        _, RG = dc.coupled_residuals(self.r, spinor["F"], spinor["G"], spinor["dF"], spinor["dG"], -1,
                                     dc.ELECTRON_REST_ENERGY_MEV, dc.GROUND_REFERENCE_ENERGY_MEV, potential_mev=V)
        manual = spinor["dG"] + spinor["G"] / self.r - (
            dc.ELECTRON_REST_ENERGY_MEV - dc.GROUND_REFERENCE_ENERGY_MEV + V) * spinor["F"] / dc.HBARC_MEV_FM
        np.testing.assert_allclose(RG, manual, rtol=1e-14, atol=1e-15)

    def test_09_positive_normalization(self):
        F = self.r * np.exp(-self.r)
        G = 0.1 * F
        Fn, Gn = dc.normalize_spinor(F, G, self.r)
        self.assertAlmostEqual(dc.joint_spinor_norm(Fn, Gn, self.r), 1.0, delta=1e-12)
        np.testing.assert_allclose(Gn / Fn, G / F, rtol=1e-13)

    def test_10_negative_normalization(self):
        F = -self.r * np.exp(-self.r)
        G = -0.1 * self.r * np.exp(-self.r)
        Fn, Gn = dc.normalize_spinor(F, G, self.r)
        self.assertTrue(np.all(Fn < 0))
        self.assertTrue(np.all(Gn < 0))
        self.assertAlmostEqual(dc.joint_spinor_norm(Fn, Gn, self.r), 1.0, delta=1e-12)

    def test_11_sign_changing_normalization(self):
        F = np.sin(self.r) * np.exp(-0.1 * self.r)
        G = np.cos(self.r) * np.exp(-0.1 * self.r)
        Fn, Gn = dc.normalize_spinor(F, G, self.r)
        self.assertEqual(np.count_nonzero(np.diff(np.signbit(F))), np.count_nonzero(np.diff(np.signbit(Fn))))
        self.assertEqual(np.count_nonzero(np.diff(np.signbit(G))), np.count_nonzero(np.diff(np.signbit(Gn))))

    def test_12_common_scale_invariance(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        reference = dc.canonical_integral_energy(self.r, spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        for scale in (2.0, -3.0, 0.2, 1e100):
            value = dc.canonical_integral_energy(self.r, scale * spinor["F"], scale * spinor["G"], V,
                                                 dc.ELECTRON_REST_ENERGY_MEV)
            self.assertAlmostEqual(value, reference, delta=2e-13)

    def test_13_global_sign_invariance(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        positive = dc.canonical_integral_energy(self.r, spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        negative = dc.canonical_integral_energy(self.r, -spinor["F"], -spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        self.assertEqual(positive, negative)

    def test_14_independent_component_scale_sensitivity(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        base = dc.canonical_integral_energy(self.r, spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        f_scaled = dc.canonical_integral_energy(self.r, 2 * spinor["F"], spinor["G"], V,
                                                dc.ELECTRON_REST_ENERGY_MEV)
        g_scaled = dc.canonical_integral_energy(self.r, spinor["F"], 2 * spinor["G"], V,
                                                dc.ELECTRON_REST_ENERGY_MEV)
        self.assertGreater(abs(base - f_scaled), 1e-10)
        self.assertGreater(abs(base - g_scaled), 1e-10)

    def test_15_zero_norm_rejection(self):
        with self.assertRaisesRegex(ValueError, "zero spinor norm"):
            dc.normalize_spinor(np.zeros_like(self.r), np.zeros_like(self.r), self.r)

    def test_16_near_zero_norm_rejection(self):
        tiny = np.full_like(self.r, 1e-200)
        with self.assertRaisesRegex(ValueError, "near-zero spinor norm"):
            dc.normalize_spinor(tiny, tiny, self.r)

    def test_17_nonfinite_rejection(self):
        bad = np.ones_like(self.r)
        bad[4] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dc.normalize_spinor(bad, np.ones_like(self.r), self.r)

    def test_18_learnable_layout_h17_h6(self):
        for h, length, cf, cg in ((17, 104, 51, 103), (6, 38, 18, 37)):
            layout = dc.branch_layout(h, dc.LEARNABLE_COEFFICIENTS)
            self.assertEqual((layout.expected_length, layout.c_f_index, layout.c_g_index), (length, cf, cg))
            unpacked = dc.unpack_vector(np.ones(length), h, dc.LEARNABLE_COEFFICIENTS)
            self.assertEqual((unpacked.f_network.size, unpacked.g_network.size), (3 * h, 3 * h))

    def test_19_fixed_layout_h17_h6(self):
        for h, length in ((17, 102), (6, 36)):
            layout = dc.branch_layout(h, dc.FIXED_UNIT_COEFFICIENTS)
            self.assertEqual(layout.expected_length, length)
            unpacked = dc.unpack_vector(np.ones(length), h, dc.FIXED_UNIT_COEFFICIENTS)
            self.assertEqual((unpacked.c_f, unpacked.c_g), (1.0, 1.0))

    def test_20_stale_vector_rejection(self):
        with self.assertRaises(ValueError):
            dc.unpack_vector(np.ones(102), 17, dc.LEARNABLE_COEFFICIENTS)
        with self.assertRaises(ValueError):
            dc.unpack_vector(np.ones(104), 17, dc.FIXED_UNIT_COEFFICIENTS)
        with self.assertRaises(ValueError):
            dc.unpack_vector(np.ones(103), 17, dc.LEARNABLE_COEFFICIENTS)

    def test_21_energy_decomposition(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        terms = dc.canonical_energy_terms(self.r, spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        energy = dc.canonical_integral_energy(self.r, spinor["F"], spinor["G"], V,
                                              dc.ELECTRON_REST_ENERGY_MEV)
        self.assertAlmostEqual(terms["numerator"] / terms["denominator"], energy, delta=1e-15)
        self.assertAlmostEqual(terms["numerator"], terms["potential_term"] + dc.ELECTRON_REST_ENERGY_MEV * terms["spinor_sum"], delta=1e-15)

    def test_22_no_pointwise_clipping(self):
        source = inspect.getsource(dc)
        self.assertNotIn("np.clip", source)

    def test_23_no_exponential_energy_regularization(self):
        source = inspect.getsource(dc.canonical_energy_terms)
        self.assertNotIn("exp(", source)
        self.assertNotIn("alpha", source)

    def test_24_export_ordering(self):
        labels, table = self.fixed_adapter.export_data(self.r, self.fixed, normalized=False)
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        self.assertEqual(labels, ("r_fm", "F_reduced", "G_reduced"))
        np.testing.assert_array_equal(table[:, 0], self.r)
        np.testing.assert_array_equal(table[:, 1], spinor["F"])
        np.testing.assert_array_equal(table[:, 2], spinor["G"])

    def test_25_plotting_data_identity(self):
        labels, table = self.fixed_adapter.export_data(self.r, self.fixed, normalized=True)
        F, G = self.fixed_adapter.normalized(self.r, self.fixed)
        np.testing.assert_array_equal(table[:, 1], F)
        np.testing.assert_array_equal(table[:, 2], G)
        self.assertEqual(labels[1:], ("F_reduced", "G_reduced"))

    def test_26_branch_equivalence_unit_coefficients(self):
        fixed_spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        learned_spinor = self.learnable_adapter.evaluate(self.r, self.learnable)
        for key in ("F", "G", "dF", "dG"):
            np.testing.assert_array_equal(fixed_spinor[key], learned_spinor[key])
        self.assertEqual(self.fixed_adapter.objective(self.r, self.fixed),
                         self.learnable_adapter.objective(self.r, self.learnable))

    def test_27_ground_state_metadata_and_reference(self):
        metadata = GroundStateMetadata()
        metadata.validate()
        self.assertAlmostEqual(dc.dirac_energy_eigenvalue(1, -1), dc.GROUND_REFERENCE_ENERGY_MEV, delta=5e-16)
        with self.assertRaises(ValueError):
            GroundStateMetadata(n=2).validate()

    def test_28_component_sign_flip_energy_only(self):
        spinor = self.fixed_adapter.evaluate(self.r, self.fixed)
        V = dc.pure_coulomb_potential(self.r)
        base = dc.canonical_integral_energy(self.r, spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        f_flip = dc.canonical_integral_energy(self.r, -spinor["F"], spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        g_flip = dc.canonical_integral_energy(self.r, spinor["F"], -spinor["G"], V, dc.ELECTRON_REST_ENERGY_MEV)
        self.assertEqual(base, f_flip)
        self.assertEqual(base, g_flip)
        RF, RG = dc.coupled_residuals(self.r, spinor["F"], spinor["G"], spinor["dF"], spinor["dG"], -1,
                                      dc.ELECTRON_REST_ENERGY_MEV, dc.GROUND_REFERENCE_ENERGY_MEV)
        RFf, RGf = dc.coupled_residuals(self.r, -spinor["F"], spinor["G"], -spinor["dF"], spinor["dG"], -1,
                                        dc.ELECTRON_REST_ENERGY_MEV, dc.GROUND_REFERENCE_ENERGY_MEV)
        self.assertGreater(np.max(np.abs(RF - RFf)) + np.max(np.abs(RG - RGf)), 1e-10)

    def test_29_zero_vector_primary_penalty(self):
        zero = np.zeros(12)
        result = self.fixed_adapter.objective(self.r, zero)
        self.assertEqual(result["residual_term"], 0.0)
        self.assertEqual(result["norm_penalty"], 1.0)
        self.assertEqual(result["total_objective"], 1.0)
        self.assertTrue(result["zero_penalized"])

    def test_30_fallback_rejects_zero(self):
        with self.assertRaises(ValueError):
            self.fixed_adapter.fallback_objective(self.r, np.zeros(12))


    def test_31_excited_state_metadata(self):
        energy = dc.dirac_energy_eigenvalue(2, -1)
        metadata = DiracStateMetadata(n=2, l=0, j=0.5, kappa=-1.0, reference_energy_mev=energy)
        metadata.validate()
        adapter = ShadowDiracAdapter(2, 0.02, dc.FIXED_UNIT_COEFFICIENTS, metadata=metadata)
        self.assertEqual(adapter.metadata.n, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
