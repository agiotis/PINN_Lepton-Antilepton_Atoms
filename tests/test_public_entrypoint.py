"""Public CLI checks that stop before solver execution and reject private imports."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
NO_PRIVATE_IMPORTS = """
import importlib.abc
import sys
class NoPrivateImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ablation_experiment' or fullname.startswith('ablation_experiment.'):
            raise ModuleNotFoundError('Private package excluded by entrypoint test', name=fullname)
sys.meta_path.insert(0, NoPrivateImports())
"""


class PublicEntrypointTests(unittest.TestCase):
    def run_without_private(self, code):
        result = subprocess.run(
            [sys.executable, "-B", "-c", NO_PRIVATE_IMPORTS + code],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def parse_without_solving(self, arguments):
        # Exercise the actual __main__ parser; stop immediately after parsing.
        code = """
import argparse
import json
import runpy
from unittest import mock
original_parse = argparse.ArgumentParser.parse_args
class ParsedEnough(Exception):
    pass
def capture(parser, *args, **kwargs):
    parsed = original_parse(parser, *args, **kwargs)
    snapshot = vars(parsed).copy()
    snapshot['mask_choices'] = next(action.choices for action in parser._actions
                                    if action.dest == 'hamiltonian_correction_terms')
    print(json.dumps(snapshot))
    raise ParsedEnough()
sys.argv = ['main.py'] + ARGUMENTS
with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
    try:
        runpy.run_path('main.py', run_name='__main__')
    except ParsedEnough:
        pass
"""
        return json.loads(self.run_without_private(code.replace("ARGUMENTS", repr(arguments))))

    def test_import_main_without_private_packages(self):
        output = self.run_without_private("\nimport main\nprint('public import succeeded')\n")
        self.assertIn("public import succeeded", output)

    def test_all_32_binary_masks_preserve_tuple_format_and_order(self):
        output = self.run_without_private("""
import json
import main
masks = main.generate_breit_terms_combinations()
assert isinstance(masks, list) and all(isinstance(mask, tuple) for mask in masks)
print(json.dumps(masks))
""")
        masks = json.loads(output)
        expected = [list(format(number, '05b')) for number in range(32)]
        self.assertEqual(masks, expected)
        self.assertEqual(len({tuple(mask) for mask in masks}), 32)
        for mask in masks:
            self.assertEqual(len(mask), 5)
            self.assertTrue(all(bit in ("0", "1") for bit in mask))

    def test_default_mask_and_all_cli_choices(self):
        parsed = self.parse_without_solving(["--mass_relation", "m1>>m2", "--subsystem", "Muonium"])
        self.assertEqual(parsed["hamiltonian_correction_terms"], "0 0 0 0 0")
        self.assertEqual(parsed["mask_choices"], [" ".join(format(number, '05b')) for number in range(32)])
        self.assertEqual(parsed["diff_equation"], "Schrodinger")

    def test_cli_help_without_private_packages(self):
        output = self.run_without_private("""
import runpy
sys.argv = ['main.py', '--help']
runpy.run_path('main.py', run_name='__main__')
""")
        for option in ("--diff_equation", "--hamiltonian_correction_terms", "--dirac-coefficient-mode"):
            self.assertIn(option, output)

    def test_schrodinger_arguments_without_private_packages(self):
        parsed = self.parse_without_solving([
            "--diff_equation", "Schrodinger", "--mass_relation", "m1>>m2",
            "--subsystem", "Muonium", "--hamiltonian_correction_terms", "0 0 0 1 0", "--input_file",
        ])
        self.assertEqual(parsed["diff_equation"], "Schrodinger")
        self.assertEqual(parsed["mass_relation"], "m1>>m2")
        self.assertEqual(parsed["subsystem"], "Muonium")
        self.assertEqual(parsed["hamiltonian_correction_terms"], "0 0 0 1 0")
        self.assertTrue(parsed["input_file"])

    def test_dirac_arguments_without_private_packages(self):
        parsed = self.parse_without_solving([
            "--diff_equation", "Dirac", "--mass_relation", "m1>>m2", "--subsystem", "Muonium",
            "--dirac-coefficient-mode", "fixed", "--u0_init_method", "physics_conditioned",
        ])
        self.assertEqual(parsed["diff_equation"], "Dirac")
        self.assertEqual(parsed["dirac_coefficient_mode"], "fixed")
        self.assertEqual(parsed["u0_init_method"], "physics_conditioned")
        self.assertEqual(parsed["hamiltonian_correction_terms"], "0 0 0 0 0")


if __name__ == "__main__":
    unittest.main()
