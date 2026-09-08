# PINN Lepton-Antilepton Atoms

This repository contains the source code accompanying the article:

**A Physics-Informed Neural Network Framework for Solving Schrödinger and Dirac Equations in Lepton-Antilepton Bound States**

Computer Physics Communications, 2026  
DOI: https://doi.org/10.1016/j.cpc.2026.110335

The code implements physics-informed neural-network solvers for quantum-mechanical and relativistic bound-state problems in lepton-antilepton atomic systems, including Schrödinger and Dirac equation examples.

## Repository structure

```text
main.py                         Main command-line entry point
requirements.txt                Python package requirements
quantum_solver/                 Core solver implementation
uinput/                         Example input/configuration files
tests/                          Focused regression tests
```

Main source files:

```text
quantum_solver/quantum_system.py
quantum_solver/neural_network.py
quantum_solver/plotting.py
quantum_solver/fundamental_values.py
quantum_solver/quantum_utils.py
quantum_solver/dirac_canonical.py
quantum_solver/dirac_canonical_adapter.py
```

## Installation

A clean Python environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Alternatively, use any Python environment with the packages listed in `requirements.txt`.

## Basic usage

The main script is:

```powershell
python .\main.py --help
```

The program prompts the user for the quantum numbers after launch. For example:

```text
1 0
```

corresponds to `n=1, l=0`.

## Hamiltonian correction-term argument

The command-line argument `--hamiltonian_correction_terms` must be passed as a quoted five-entry spaced mask.

Correct examples:

```powershell
--hamiltonian_correction_terms "0 0 0 0 0"
--hamiltonian_correction_terms "0 0 0 1 0"
```

Do not pass compact masks such as:

```text
00010
```

The mask `"0 0 0 0 0"` corresponds to the baseline Hamiltonian term. Other masks activate the corresponding Hamiltonian correction-term combinations implemented in the code.

## Schrödinger equation examples

### Ground state, baseline Hamiltonian

```powershell
python .\main.py `
  --diff_equation Schrodinger `
  --mass_relation "m1>>m2" `
  --subsystem Muonium `
  --hamiltonian_correction_terms "0 0 0 0 0" `
  --num_of_hidden_units 6 `
  --num_of_equidistant_points 150 `
  --optimization_method BFGS `
  --minimization_tol_lvl 1e-8 `
  --activation_function relu `
  --u0_init_method random `
  --overlay_plot
```

When prompted, enter:

```text
1 0
```

### Ground state with a correction-term mask

```powershell
python .\main.py `
  --diff_equation Schrodinger `
  --mass_relation "m1>>m2" `
  --subsystem Muonium `
  --hamiltonian_correction_terms "0 0 0 1 0" `
  --num_of_hidden_units 6 `
  --num_of_equidistant_points 150 `
  --optimization_method trust-constr `
  --minimization_tol_lvl 1e-8 `
  --activation_function relu `
  --u0_init_method random `
  --overlay_plot
```

When prompted, enter:

```text
1 0
```

## Dirac equation examples

For stable Dirac benchmark execution, the recommended profile is:

```text
--u0_init_method physics_conditioned
--dirac-coefficient-mode fixed
```

### Dirac ground state, n=1, l=0

```powershell
python .\main.py `
  --diff_equation Dirac `
  --mass_relation "m1>>m2" `
  --subsystem Muonium `
  --num_of_hidden_units 6 `
  --num_of_equidistant_points 450 `
  --optimization_method L-BFGS-B `
  --minimization_tol_lvl 1e-8 `
  --activation_function tanh `
  --u0_init_method physics_conditioned `
  --dirac-coefficient-mode fixed `
  --overlay_plot
```

When prompted, enter:

```text
1 0
```

### Dirac example, n=2, l=0

```powershell
python .\main.py `
  --diff_equation Dirac `
  --mass_relation "m1>>m2" `
  --subsystem Muonium `
  --num_of_hidden_units 20 `
  --num_of_equidistant_points 500 `
  --optimization_method L-BFGS-B `
  --minimization_tol_lvl 1e-8 `
  --activation_function tanh `
  --u0_init_method physics_conditioned `
  --dirac-coefficient-mode fixed `
  --overlay_plot
```

When prompted, enter:

```text
2 0
```

## Outputs

The program writes numerical values and plots under the local `data/` directory. Typical outputs include CSV/TXT numerical files and PDF plots comparing analytical and numerical solutions.

## Tests

Focused regression tests can be executed with:

```powershell
python -m unittest -v tests.test_dirac_canonical tests.test_dirac_runtime_wiring
```

## Reproducibility note

The examples above are intended to provide practical entry points for reproducing representative Schrödinger and Dirac runs associated with the article. A more detailed validation and reproducibility description may be provided in a future release.

## Citation

If you use this code, please cite:

```bibtex
@article{GIOTIS2026110335,
  title   = {A unified Python framework for solving Schr{\"o}dinger and Dirac equations in exotic lepton--antilepton atoms},
  journal = {Computer Physics Communications},
  volume  = {328},
  pages   = {110335},
  year    = {2026},
  issn    = {0010-4655},
  doi     = {10.1016/j.cpc.2026.110335},
  url     = {https://www.sciencedirect.com/science/article/pii/S0010465526003176},
  author  = {Giotis, Angelos P. and Gkrepis, Athanasios and Kosmas, Odysseas and Kosmas, Theocharis S.}
}
```

## License

This repository is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See `LICENSE` for details.

