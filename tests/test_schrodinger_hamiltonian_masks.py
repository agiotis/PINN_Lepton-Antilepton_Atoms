import unittest
from unittest import mock

from quantum_solver.quantum_system import Particle, normalize_hamiltonian_correction_terms


def minimal_hyperparameters(mask):
    return {
        'diff_equation': 'Schrodinger',
        'arguments.parent_equation': 'Schrodinger',
        'hamiltonian_correction_terms': mask,
        'num_of_hidden_units': 2,
        'num_of_equidistant_points': 7,
        'fuc': 1,
        'optimization_method': 'BFGS',
        'minimization_tol_lvl': 1e-8,
        'primary_quantum_number': 1,
        'azimuthal_quantum_number': 0,
        'total_angular_momentum': None,
        'k_operator': None,
        'coupling_constant': 1.0,
        'two_body_system': 'combined_hamiltonian',
        'k_optimization_constant': 0.16,
        'u0_init_method': 'fixed',
        'activation_function': 'tanh',
        'user_search_grid': 0,
        'overlay_plot': False,
        'input_file': True,
        'verbose': False,
        'experiment_index': 99,
        'mass_relation': 'm1>>m2',
        'subsystem': 'Muonium',
        'numerical_method': 'NeuralNetworks',
        'manual_u0': None,
        'dirac_coefficient_mode': None,
    }


class SchrodingerHamiltonianMaskTests(unittest.TestCase):
    def test_spaced_string_masks_normalize_to_five_ints(self):
        cases = {
            "0 0 0 0 0": [0, 0, 0, 0, 0],
            "1 0 0 0 0": [1, 0, 0, 0, 0],
            "0 1 0 0 0": [0, 1, 0, 0, 0],
            "0 0 1 0 0": [0, 0, 1, 0, 0],
            "0 0 0 1 0": [0, 0, 0, 1, 0],
            "0 0 0 0 1": [0, 0, 0, 0, 1],
            "1 1 1 1 1": [1, 1, 1, 1, 1],
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_hamiltonian_correction_terms(raw), expected)

    def test_internal_sequence_masks_normalize_to_five_ints(self):
        self.assertEqual(
            normalize_hamiltonian_correction_terms([0, 0, 0, 1, 0]),
            [0, 0, 0, 1, 0],
        )
        self.assertEqual(
            normalize_hamiltonian_correction_terms(["0", "0", "0", "1", "0"]),
            [0, 0, 0, 1, 0],
        )
        self.assertEqual(
            normalize_hamiltonian_correction_terms(("0", "0", "0", "1", "0")),
            [0, 0, 0, 1, 0],
        )

    def test_invalid_masks_raise_value_error(self):
        invalid_masks = [
            "00010",
            "0 0 0 1",
            "0 0 0 1 0 0",
            "0 0 0 2 0",
            "",
            None,
        ]
        for mask in invalid_masks:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    normalize_hamiltonian_correction_terms(mask)

    def test_particle_stores_normalized_breit_mask(self):
        with mock.patch.object(Particle, "save_parameters", return_value=None):
            particle = Particle(minimal_hyperparameters("0 0 0 1 0"), 1)

        self.assertEqual(particle.breit, [0, 0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
