# PINN Lepton-Antilepton Atoms

This repository contains the source code accompanying the article:

**A unified Python framework for solving Schrödinger and Dirac equations in exotic lepton–antilepton atoms**

Angelos P. Giotis, Athanasios Gkrepis, Odysseas Kosmas and Theocharis S. Kosmas

Computer Physics Communications **328** (2026), 110335

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

Use Python 3.10 and a clean virtual environment. Run the following commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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

Run commands from the repository root so the input and output paths resolve correctly. The `--input_file` flag reads the state from `uinput/inpsch.csv` for Schrödinger runs or `uinput/inpdc.csv` for Dirac runs instead of prompting for quantum numbers.

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

For Schrödinger runs, the five binary entries select **H1, H2, H3, H4 and H5**, in that order. **H0 is always included**: `"0 0 0 0 0"` selects H0 only, while `"0 0 0 1 0"` selects H0 + H4. All 32 masks are supported; only enabled correction terms are evaluated.

### Implementation updates

This maintenance release corrects correction-mask parsing, reduced radial operators for `U(r) = r R(r)`, higher-order trial-function and activation derivatives, and H1–H5 physical prefactors and angular/contact contributions. These changes make the selected correction operators act consistently on the trial function within the existing Schrödinger formulation.

The public entrypoint no longer requires private batch modules. Explicitly supplied Schrödinger minimization tolerances are passed to the selected SciPy optimizer. These refinements preserve the existing solver architecture and reference-energy evaluation.

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

The following examples use random initialization and learnable F/G component coefficients. Dirac runs require an explicit `--dirac-coefficient-mode`. These are supported single-run commands; convergence depends on the initialization and numerical configuration.

### Dirac ground state, n=1, l=0

```powershell
python .\main.py `
  --diff_equation Dirac `
  --mass_relation "m1>>m2" `
  --subsystem Muonium `
  --num_of_hidden_units 6 `
  --num_of_equidistant_points 450 `
  --optimization_method L-BFGS-B `
  --activation_function tanh `
  --u0_init_method random `
  --dirac-coefficient-mode learnable `
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
  --activation_function tanh `
  --u0_init_method random `
  --dirac-coefficient-mode learnable `
  --overlay_plot
```

When prompted, enter:

```text
2 0
```

## Outputs

The program writes numerical values and plots under the local `data/` directory. Typical outputs include CSV/TXT numerical files and PDF plots comparing analytical and numerical solutions.

For Schrödinger runs, the printed `Energy` is the H0 reference binding energy. `Wave function Energy` evaluates H0 on the fitted wavefunction, including when correction terms are selected for the optimization residual. Its distance from the reference is therefore an H0 energy diagnostic, rather than a full corrected-Hamiltonian eigenenergy.

Dirac runs can load existing `minimizer_params_*.pkl` files from the working directory. Use a separate checkout without saved minimizers for an independent run.

## Tests

Focused regression tests can be executed with:

```powershell
python -m unittest -v tests.test_dirac_canonical tests.test_dirac_runtime_wiring
```

## Reproducibility note

The examples document supported single-run usage. Record the source revision, command, quantum-state input and initial parameters when comparing runs.

## Citation

If you use this code, please cite:

Angelos P. Giotis, Athanasios Gkrepis, Odysseas Kosmas and Theocharis S. Kosmas, **A unified Python framework for solving Schrödinger and Dirac equations in exotic lepton–antilepton atoms**, Computer Physics Communications **328** (2026), 110335.

https://doi.org/10.1016/j.cpc.2026.110335

```bibtex
@article{Giotis2026QuantumSolver,
  author  = {Giotis, Angelos P. and Gkrepis, Athanasios and
             Kosmas, Odysseas and Kosmas, Theocharis S.},
  title   = {A unified {Python} framework for solving {Schr{\"o}dinger} and
             {Dirac} equations in exotic lepton--antilepton atoms},
  journal = {Computer Physics Communications},
  volume  = {328},
  year    = {2026},
  pages   = {110335},
  doi     = {10.1016/j.cpc.2026.110335}
}
```

## License

This repository is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See `LICENSE` for details.
