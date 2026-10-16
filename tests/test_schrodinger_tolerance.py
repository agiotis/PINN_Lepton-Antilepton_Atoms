"""Tolerance wiring checks; all optimizer calls and file output are mocked."""

import contextlib
import io
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from quantum_solver.main_solver import MainSolver
from quantum_solver.quantum_system import Electron, Particle
from tests.test_schrodinger_hamiltonian_masks import minimal_hyperparameters


MAIN = Path(__file__).resolve().parents[1] / "main.py"


class OptimizerCaptured(Exception):
    """Stop at the optimizer boundary, before any optimization or postprocessing."""


class SchrodingerToleranceTests(unittest.TestCase):
    def capture_cli(self, extra=(), equation="Schrodinger"):
        captured = {}

        def capture(solver, particle):
            captured.update(parameters=solver.hyperparameters, particle=particle)

        argv = [str(MAIN), "--diff_equation", equation, "--mass_relation", "m1>>m2",
                "--subsystem", "Muonium", "--num_of_hidden_units", "6",
                "--num_of_equidistant_points", "150", "--optimization_method", "BFGS",
                "--activation_function", "relu", "--u0_init_method", "random",
                "--hamiltonian_correction_terms", "0 0 0 0 0"]
        if equation == "Dirac":
            argv += ["--dirac-coefficient-mode", "fixed"]
        state = SimpleNamespace(n=1, l=0, j=0.5, k=-1, system_index=1)
        with mock.patch("sys.argv", argv + list(extra)), \
                mock.patch("quantum_solver.quantum_utils.Schrodinger", return_value=state), \
                mock.patch("quantum_solver.quantum_utils.Dirac", return_value=state), \
                mock.patch.object(Particle, "save_parameters"), \
                mock.patch.object(Electron, "save_parameters"), \
                mock.patch.object(MainSolver, "solve_system", autospec=True, side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(MAIN), run_name="__main__")
        return captured["parameters"], captured["particle"]

    def capture_minimize(self, parameters, particle):
        initial = np.linspace(0.1, 0.9, 3 * parameters["num_of_hidden_units"])
        grid = np.linspace(1, 2, parameters["num_of_equidistant_points"])
        with mock.patch.object(particle, "initialize_optimization_params", return_value=initial), \
                mock.patch.object(particle, "save_initial_point_u0"), \
                mock.patch.object(particle, "RelativeDistanceSearchGrid", return_value=grid), \
                mock.patch("quantum_solver.quantum_system.optimize.minimize",
                           side_effect=OptimizerCaptured) as minimize, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OptimizerCaptured):
                particle.solve_schrodinger_equation(parameters)
        minimize.assert_called_once()
        args, kwargs = minimize.call_args
        self.assertIs(args[1], initial)
        self.assertTrue(callable(args[0]))
        self.assertEqual(kwargs["options"], {"disp": True})
        self.assertEqual(set(kwargs), {"method", "tol", "options"})
        return kwargs

    def test_explicit_cli_tolerances_reach_particle_and_bfgs(self):
        for raw in ("1e-8", "2.75e-6"):
            with self.subTest(tolerance=raw):
                hp, particle = self.capture_cli(["--minimization_tol_lvl", raw])
                self.assertIsInstance(hp["minimization_tol_lvl"], float)
                self.assertTrue(hp["minimization_tol_explicit"])
                self.assertEqual(particle.tolerance_level, float(raw))
                kwargs = self.capture_minimize(hp, particle)
                self.assertEqual(kwargs["method"], "BFGS")
                self.assertEqual(kwargs["tol"], float(raw))

    def test_omitted_cli_tolerance_preserves_parser_default_and_optimizer_none(self):
        hp, particle = self.capture_cli()
        self.assertEqual(hp["minimization_tol_lvl"], 1e-8)
        self.assertFalse(hp["minimization_tol_explicit"])
        self.assertIsNone(particle.tolerance_level)
        self.assertIsNone(self.capture_minimize(hp, particle)["tol"])

    def test_supported_schrodinger_optimizers_receive_same_explicit_tolerance(self):
        for method in ("BFGS", "L-BFGS-B", "trust-constr"):
            with self.subTest(method=method):
                hp, particle = self.capture_cli([
                    "--optimization_method", method, "--minimization_tol_lvl=2.75e-6"])
                kwargs = self.capture_minimize(hp, particle)
                self.assertEqual(kwargs["method"], method)
                self.assertEqual(kwargs["tol"], 2.75e-6)

    def test_omitted_tolerance_preserved_for_all_supported_optimizers(self):
        for method in ("BFGS", "L-BFGS-B", "trust-constr"):
            with self.subTest(method=method):
                hp, particle = self.capture_cli(["--optimization_method", method])
                kwargs = self.capture_minimize(hp, particle)
                self.assertEqual(kwargs["method"], method)
                self.assertIsNone(kwargs["tol"])

    def test_direct_hyperparameter_value_missing_and_none(self):
        for mode, expected in (("explicit", 2.75e-6), ("missing", None), ("none", None)):
            hp = minimal_hyperparameters("0 0 0 0 0")
            if mode == "missing":
                del hp["minimization_tol_lvl"]
            else:
                hp["minimization_tol_lvl"] = expected
            with mock.patch.object(Particle, "save_parameters"):
                particle = Particle(hp, 1)
            with self.subTest(mode=mode):
                self.assertEqual(particle.tolerance_level, expected)
                self.assertEqual(self.capture_minimize(hp, particle)["tol"], expected)

    def test_explicit_tolerance_preserves_masks_and_unrelated_parameters(self):
        for mask in ("0 0 0 0 0", "0 0 0 1 0", "1 1 1 1 1"):
            hp, particle = self.capture_cli([
                "--hamiltonian_correction_terms", mask, "--minimization_tol_lvl", "2.75e-6"])
            with self.subTest(mask=mask):
                self.assertEqual(hp["hamiltonian_correction_terms"], mask)
                self.assertEqual(particle.breit, [int(bit) for bit in mask.split()])
                self.assertEqual(hp["num_of_hidden_units"], 6)
                self.assertEqual(hp["num_of_equidistant_points"], 150)
                self.assertEqual(hp["activation_function"], "relu")
                self.assertEqual(hp["u0_init_method"], "random")
                self.assertEqual(hp["optimization_method"], "BFGS")
                self.assertEqual(hp["primary_quantum_number"], 1)
                self.assertEqual(hp["azimuthal_quantum_number"], 0)

    def test_dirac_tolerance_initialization_unchanged(self):
        for extra in ([], ["--minimization_tol_lvl", "2.75e-6"]):
            with self.subTest(extra=extra):
                hp, electron = self.capture_cli(extra, equation="Dirac")
                self.assertIsInstance(electron, Electron)
                self.assertIsNone(electron.tolerance_level)
                self.assertEqual(hp["minimization_tol_lvl"], 2.75e-6 if extra else 1e-8)


if __name__ == "__main__":
    unittest.main()
