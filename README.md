# End-to-end Physics Informed Neural Network (PINN) Solvers for lepton - antilepton biraries 
This repository corresponds to the final project's deliverables, concerning:
- Milestone-2: Software_m1m2 providing the boundstates of 2-Dirac particles with $m_1≠m_2$.
  - Deliverable D.4.1. "The advanced DiracBreitDarwinAlgo algorithm for purely leptonic atoms".
- Milestone-4: Software-mm providing the bound states of 2-Dirac particles with $m_1=m_2=m$.
  - D.6.1. "The Software_mm routines giving bound states in 2-Dirac particles of equal masses $m1=m2=m$"
- Milestone-6: Software_$m_1$mm providing the bound states of 3-Dirac particles with $m_2=m_3=m, m_1≠m$
  - Deliverable D.8.1. "The Software_$m_1$mm routines providing the bound states of the 3-particle ($m_1>>m_2=m_3=m$) muonium ion, $Mu^-$".
  - This case is partially implemented for the solution of the Schrödinger differential equation.
    
All deliverables are integrated within our proposed end-to-end Python framework for solving differential equations (Schrödinger, Dirac-Coulomb, Klein-Gordon) for purely leptonic systems.

- We consider two basic axes for the leptonic particle systems by selecting corresponding **system index** at run time:

1. **Lepton-antilepton binary systems with $m1>>m2$ are the following:**
 - **1.1** Muonium $(μ^+ e^-)$,
 - **1.2** e_Tauonium $(τ^+ e^-)$,
 - **1.3** μ_Tauonium $(τ^+ μ^-)$
 - **1.4 (beta-phase)** NewSystemDiffMass (e.g., $m_1 = 2GeV, m_2 = 1 Gev$)

**Axis 1**: Solves one-body *Schrödinger* or *Dirac* equation for each of the first 3 systems selected at input phase. Additional Solver is on beta testing phase to solve a new system for particles with different masses (two-body system).
- Extended *non-relativistic description* of the Schrödinger equation with *relativistic* augmented *Hamiltonian correction terms* (i.e., Breit-Darwin terms, see `example 7` and `note 4` for further information).
- System index, quantum numbers and other options are defined at run time as additional arguments of the main.py entry point. 
- **Even though the underlying code has the potential to provide results for the above 3 systems (1.1 to 1.3), within the context of this project, we have thoroughly checked only the case of "Muonium" purely leptonic atom**.
2. **Lepton-antilepton binary systems with of equal masses: $m_1 = m_2$**
 - **2.1** Positronium $(e^+ e^-)$,
 - **2.2** True_Muonium $(μ^+ μ^-)$,
 - **2.3** True_Tauonium $(τ^+ τ^-)$
 - **2.4 (beta-phase)** NewSystemEqMass (e.g., $m_1 = m_2 = 0.1 GeV$)

**Axis 2**: Solvers two-body Dirac-Coulomb equation that is actually reduced for each of the 4 abovementioned systems (2.1 to 2.4) into coupled ODE:
- Klein-Gordon equation wherein rest masses $m_1 = m_2$ are actually reduced to $M_0 = 2 * m_1$ for the calculation of the integral energy. 
- **Again, our code has the potential to provide results all of the above 4 systems (2.1 to 2.4). However, within the context of this project, we have thoroughly checked only the case of "Positronium" purely leptonic atom**.
## Summary of Quantum Numbers

- **n**: Principal quantum number $n = 1, 2, 3, \ldots$.
- **l**: Azimuthal (orbital angular momentum) quantum number $l = 0, 1, 2, \ldots, n - 1$.
- **s**: Spin quantum number $s = \frac{1}{2}$ for leptons.
- **j**: Total angular momentum quantum number $j = l \pm \frac{1}{2}$.
- **k**: Related to $j$ and $l$ by $k = \pm (j + \frac{1}{2})$. This is related to the Dirac or Klein-Gordon equation.

## Installation Requirements - Deployment

1. **Install Anaconda**:
   If you do not have Anaconda installed, download and install it from [here](https://www.anaconda.com/products/distribution).
   - For Windows users, [Anaconda Navigator](https://docs.anaconda.com/navigator/index.html) is available as a GUI option for easier code deployment and package management.

2. **Clone the Repository**:
   Open a terminal and clone the repository to your local machine. You can either clone the original repository or your forked version (recommended if you plan to make changes).
   
   - **Clone the original repository**:
     ```sh
     git clone https://github.com/agiotis/End-to-end_PINN_Solvers_For_Lepton-Antilepton_Binaries.git
     ```

   - **Clone your forked repository**:
     ```sh
     git clone https://github.com/OPRA-UoI-UoJ/End-to-end_PINN_Solvers_For_Lepton-Antilepton_Binaries.git
     ```

   Navigate to the project directory:
   ```sh
   cd End-to-end_PINN_Solvers_For_Lepton-Antilepton_Binaries
   ```

3. **Create a New Environment**:
   Use Anaconda to create a new environment named `quantum_solver` with Python 3.10.9:
   ```sh
   conda create -n quantum_solver python=3.10.9
   ```

4. **Activate the Environment**:
   Activate the newly created environment:
   ```sh
   conda activate quantum_solver
   ```

5. **Install Required Packages**:
   Install all the necessary Python packages listed in the `requirements.txt` file:
   ```sh
   pip install -r requirements.txt
   ```
After completing these steps, your environment should be set up, and you’ll be ready to run the code.

# End-to-end software updates for D.4.1. and D.6.1.

## DiracBreitDarwinAlgoCode extended implementation (D.4.1.):
- 2-layer feed-forward Neural networks were used to estimate trial equations (e.g., u(r)) and minimize estimation errors, for solving the Schrödinger and Dirac-Coulomb Differential Equations, initially applied to a 2-particle system and extendable to other quantum systems.
- The integration and debugging of Hamiltonian correction terms, such as Breit and Darwin terms, into the reduced radial part of the Schrödinger wave function, have been completed.
- Enriched with the addition of batch experiment execution capabilities:
  - ablation-experiment/**batch_schrodinger.py** 
  - ablation-experiment/**batch_dirac.py**

  in order to determine the limits of the system's hyperparameters, for which innovative computational approximation results of the advanced solutions of the above differential equations are achieved.

[//]: # ( Also, assessing their impact on optimizing the objective function for the Schrödinger equation whereas extending these corrections to be readily applicable for the Dirac-Coulomb equation &#40;which actually reduces to a Schrodinger-like wave function&#41; has been incorporated into a unified framework, as outlined in Deliverable D6.1.&#41;)


## Software_mm_routines updated implementation (D.6.1.):
- 2-layer feed-forward Neural networks were used again to estimate the trial equation u(r) and minimize the error function concerning Klein-Gordon equation for the two-body abovementioned systems (2.1 to 2.4).
- Integral energy calculation has been updated to account for overflow warnings when computing the contribution of each term of the Hamiltonian. 
- Numerical stability of the integral energy is ensured by selecting the proper normalization constant after ablative analysis of the available normalization constant options.

[//]: # (- The results of this work lay the foundation for extending the "DiracBreitDarwinAlgo" algorithm to 3-leptonic particle systems, specifically the muonium ion Mu-, as detailed in Deliverable D8.1, which focuses on providing bound states of the 3-particle system &#40;m1>>m2=m3=m&#41;.)
# Package Quantum Solver

This package integrates the solutions of Schrödinger, Dirac and Klein-Gordon equations for quantum systems using numerical methods and neural networks.

## Overview

The `main.py` script serves as the entry point for solving quantum equations. It takes various input parameters to configure the solver and select the type of differential equation to solve (Schrödinger, Dirac, KG).

### UML Class Diagram

Below are the UML class diagrams generated from the code using `$pyreverse`:

- [Packages Diagram (packages.pdf)](class_diagram/packages_quantum_solver.pdf)
- [Classes Diagram (classes.pdf)](class_diagram/classes_quantum_solver.pdf)

These diagrams provide an overview of the package and class structure of the code.

## Parameters

- **`--diff_equation`**: Choose the type of differential equation to solve (`options`: Schrodinger, Dirac, Klein-Gordon, `default`: Schrodinger).

- **`--hamiltonian_correction_terms`**: Choices for Hamiltonian correction terms (i.e., **Breit-Darwin**) that enter the Hamiltonian $H_0$). Use 'custom' to provide your own values from all possible combinations (32 in total). Default is no correction terms at all, i.e., $H_0$ (`default`: "0 0 0 0 0").

- **`--num_of_hidden_units`**: Number of hidden units/neurons in the first hidden layer of the neural network (integer, `default`: 10).
  
- **`--num_of_equidistant_points`**: Number of equidistant points in the grid from e.g., $r=0$ to $r=b$ (integer, `default`: 150).

- **`--coupling_constant`**: User can provide coupling constant during input. If not provided then $self.fsc = 1/137$ is set beforehand. (float, `default`: 0.0072973525693).

- **`--fuc`**: Regularizing heuristic constant for each distinct term of the error function for Schrödinger (float, `default`: 1.0).

- **`--optimization_method`**: Optimization method (scipy.optimize - local multivariate) for finding minimizers (`options`: BFGS/trust-constr/L-BFGS-B, `default`: trust-constr).
  
- **`--minimization_tol_lvl`**: Tolerance level for the optimization method. The interpretation may vary depending on the optimization method and problem characteristics (float, `default`: 1e-8).

- **`--u0_init_method`**: Initialization method for ($u_0 = [u_0, v_0, w_0]$) values (`options`: random/xavier/fixed/bounded, `default`: random).

- **`--activation_function`**: Activation function for the neural network (`options`: sigmoid/relu/tanh/smplus, `default`: relu).
  
- **`--search_grid`**: Only for Dirac Analytical solution: Adaptive Quadrature or user-defined method for ($r_i$) values required by optimize.quad for integral calculations (`options`: 0/1/2, `default`: 0).

- **`--overlay_plot`**: Optional argument for plotting behavior (`options`: action='store_true', `default`: False, `help`: "Enable overlayed plot").

- **`--input_file`**: Optional argument for reading default values from **.csv file**: `\uniput\inpdc.csv` for Dirac, `\uniput\inpKG.csv` for Klein-Gordon, or `\uniput\inpsch.csv` for Schrödinger (`options`: action='store_true', `default`: False, `help`: "Read default values for quantum numbers and system index from file /uinput/file.csv").

- **`--verbose`**: This parameter controls the output/plot information shown to the user (`options`: action='store_true', `default`: False, `help`: "Print analytical and numerical plots (Dirac/Schrödinger) and optimization progress bar if used").

- **`--experiment_index`**: Ablation experiment index to be passed to the Plotting module (integer, `default`: 2).

- **`--mass_relation`**: Specify the mass relation between particles (`choices`: "m1>>m2", "m1=m2", "dynamic", `required`: True).

- **`--subsystem`**: Specify the subsystem of the quantum system (`choices`: "Muonium", "e-Tauonium", "μ-Tauonium", "Positronium", "True_Muonium", "True_Tauonium", "Other", `default`: "Other").

### Note 1 (--search_grid)
- Option 0 refers to Adaptive Quadrature wherein `optimize.quad` dynamically adjusts the number of points based on the local behavior of the integrand. This can be beneficial for integrands with varying levels of complexity.
- Option 1 is a user pre-defined number of points for integral calculation, e.g., `integrate.quad(function, 1e-5, 1e5, points=num_of_equidistant_points)`
- Option 2 is a user defined method for selecting $r_i$ points used for integral calculation, e.g., `search_grid = numpy.linspace(start=1e-5, end=1e5, num_of_equidistant_points)` so that integral is calculated as `optimize.quad(function, search_grid[0], search_grid[-1])`
- **search_grid** values related to Options 1 and 2 are working for the analytical solution of the Dirac equation only! The numerical solution outputs an integral overflow calculation and is still under investigation.

### Note 2 (--overlay_plot)
If --overlay_plot is used for sovling:
- Schrödinger equation: analytical and numerical plots are overlayed (superimposed) on the same figure.
- Dirac equation: Plots for f and g components are overlayed on the same figure either for analytical or numerical solutions.
  
Else: 
- Schrödinger equation: analytical and numerical plots are displayed on separate figures
- Dirac equation: Plots for f and g components are displayed on separate figures for both analytical and numerical solutions.

### Note 3 (--input_file)
If --input_file is **used**, default values are loaded from csv file. These values correspond to `quantum numbers`, i.e., $n =$ primary quantum number (Sch/Dirac/KG), $l =$ azimuthal (orbital angular momentum) quantum number (Sch/Dirac/KG), $j =$ total angular momentum (Dirac/KG), $k =$ k operator (Dirac/KG), and `quantum system index`, i.e., $1 :$ "Muonium", $2 :$ "e-Tauonium", $3 :$ "μ-Tauonium", $4 :$ "NewSystemDiffMass", $5 :$ "Positronium", $6 :$ "True Muonium", $7 :$ "True Tauonium, $8 :$ "NewSystemEqMass"

If --input_file is **not used**, then the user **interactively** (console) inserts values for `quantum numbers` ($n, l$ for Schrödinger,  $n, l$ and $spin = up$ or $spin = down$ for Dirac/KG) and `quantum system index`, respectively.

#### Example "inpsch.csv" Description:

The file `uinput/inpsch.csv` contains the following data structure:

- **First row**: System index. In this case, the value `1` corresponds to "Muonium".
- **Second row**: Quantum number $n$, where $n=1$.
- **Third row**: Quantum number $l$, where $l=0$.

## Examples

1. **Solve Schrödinger equation with default parameters for Muonium one body system of different masses:**
   ```sh
   python main.py --mass_relation "m1>>m2" --subsystem Muonium --overlay_plot

- [This example implements the code related to the paper of Gkrepis et al. (IC-MSQUARE 2023, accepted)](papers/GkrepisKosmasVlachosKosmas-Schrodinger-IC-MSQUARE2023.pdf)

2. **Solve Schrödinger equation with default parameters for selected (interactive session) one body system of different masses:**
   ```sh
   python main.py --mass_relation dynamic
- This will open interactive runtime mode: "Non-Relativistic Description"
- "Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, 5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit): "
- The code will continue asking quantum numbers, after selecting quantum system corresponding to $m_1$ >> $m_2$ (system indices 1 to 4) or $m_1 = m_2$ (system indices 5 to 8).

3. **Solve one-body Positronium system of equal masses using (default) Schrödinger equation (Milestone 4 - stable solution)**
    ```sh
    python main.py --mass_relation m1=m2 --subsystem Positronium

4. **Solve two-body Positronium system of equal masses using Klein-Gordon equation (Milestone 4 - room for improvement in numerical stability)**
    ```sh
    python main.py --mass_relation m1=m2 --subsystem Positronium --diff_equation KleinGordon
   
- this case is still an active area of research in order to achieve numerical stability (even for the ground energy state).

5. **Solve Dirac equation with custom parameters and export overlayed analytical and numerical plots for f,g, respectively:**
    ```sh
    python main.py --mass_relation "m1>>m2" --subsystem Muonium --diff_equation Dirac --num_of_hidden_units 8 --num_of_equidistant_points 500 --optimization_method L-BFGS-B --u0_init_method xavier --activation_function tanh --overlay_plot

- [This example implements the code related to the paper of Gkrepis et al. (MEDEX 2023)](papers/GkrepisKosmasKosmas-Dirac-Coulomb-Medex2023.pdf)

6. **Solve Dirac equation with default values loaded from input file, show optimization progress bar and plot output figures (verbose=True)**
    ```sh
    python main.py --mass_relation "m1>>m2" --subsystem Muonium --diff_equation Dirac --num_of_hidden_units 8 --num_of_equidistant_points 300 --optimization_method L-BFGS-B --activation_function tanh --verbose --input_file

7. **Solve Dirac equation with additional options for search grid ($r_i$ values), i.e., user-defined points instead of *Adaptive Quadrature* (optimize.quad) default behaviour**
    ```sh
    python main.py --mass_relation "m1>>m2" --subsystem Muonium --diff_equation Dirac --num_of_hidden_units 8 --num_of_equidistant_points 300 --optimization_method L-BFGS-B --activation_function tanh --search_grid 1 --verbose --input_file

8. **Solve Schrödinger equation with `all` Hamiltonian correction (including Breit-Darwin) terms and default parameters loaded from file:**
    ```sh
    python main.py --mass_relation "m1>>m2" --subsystem Muonium --hamiltonian_correction_terms '1 1 1 1 1' --input_file
    
- This option integrates the following terms into the total Hamiltonian operator (1.5):

1.1) '0 0 0 0 0': $H_0$

1.2) '1 0 0 0 0': $H_0 + H_1$

1.3) '1 1 0 0 0': $H_0 + H_1 + H_2$

$\ldots$

1.5) '1 1 1 1 1': $H_0 + H_1 + \dots + H_5$

Where:

- **$H_0$**:  
  $H_0 = - \frac{\hbar^2}{2\mu} \nabla^2 + V$  
  Non-relativistic Hamiltonian, where $\mu$ is the reduced mass.

- **$H_1$**:  
  $H_1 = - \frac{(\hbar c)^4}{8c^6} \left( \frac{1}{m_1^3} + \frac{1}{m_2^3} \right) \nabla^4$
  Relativistic kinetic energy correction term.

- **$H_2$**:  
  $H_2 = \frac{e_1 e_2 (\hbar c)^2}{2 m_1 m_2 c^4 r} \left( \nabla^2 + \frac{\partial^2}{\partial r^2} \right)$  
  Retarded potential terms accounting for relativistic corrections to the interaction potentials.

- **$H_3$** (Spin-Orbit Coupling):
  - For $m_1 = m_2$:  
    $H_3 = \frac{3 e^2 (\hbar c)^2}{4 m^2 c^4 r^3} \left[ j(j+1) - l(l+1) - s(s+1) \right]$
    
  - For $m_1 \neq m_2$:  
    $H_3 = \frac{e^2 (\hbar c)^2}{2c^4 r^3} \left[ \frac{j(j+1) - l(l+1) - s(s+1)}{m_1 m_2} + \frac{j_1(j_1 + 1) - l(l+1) - \frac{3}{4}}{2m_1^2} + \frac{j_2(j_2 + 1) - l(l+1) - \frac{3}{4}}{2m_2^2} \right]$

- **$H_4$** (Breit-Darwin term):  
  $H_4 = \frac{e^2 (\hbar c)^2}{2c^4} \left( \frac{1}{m_1^2} + \frac{1}{m_2^2} \right) \delta(r)$  
  Darwin term accounting for relativistic quantum corrections.

- **$H_5$** (Spin-Spin interaction):  
  $H_5 = \frac{e_1 e_2 (\hbar c)^2}{m_1 m_2 c^4} \left( H_5^{(1)} + H_5^{(2)} \right)$  
  Describes the interaction between the spins of two particles.

- Note that the more terms we include inside the hamiltonian operator, the longer the code requires to produce expected results.

9. **Solve Schrödinger equation with `custom` selected Breit-Darwin terms and default parameters loaded from file:**
    ```sh
    python main.py --mass_relation "m1>>m2" --subsystem Muonium --hamiltonian_correction_terms custom --input_file
   
- example use of custom: $'0 0 0 1 0'$: reflects the choice $H_0 + H_4$ Hamiltonian operator (it actually yields better results than those of the standard $H_0$ (non-relativistic) operator, which is confirmed after extensive ablative analysis for the Schrödinger equation).
   
10. **NEW:** **Solve Klein-Gordon equation for the Positronium system of equal masses and overlay numerical energy (of the two-body system) vs analytical (one body-system for reference only):**
    ```sh
    python main.py --mass_relation "m1=m2" --subsystem Positronium --diff_equation KleinGordon --overlay_plot --num_of_equidistant_points 300

11. **NEW (OPRA-Test-1 - Interactive):** **Solve Klein-Gordon equation for a new system of equal masses (`NewSystemEqMass`) set dynamically by the user at run time:**
    ```sh
    python main.py --mass_relation dynamic --overlay_plot --num_of_equidistant_points 200

- This will open interactive runtime mode: "Two-body Relativistic Description"
- "Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, 5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit): "
- The user should select SystemIndex '8' (NewSystemEqMass) and then he will be asked to insert either of the two equal masses in MeV ($m_1$ or $m_2$).
- The code will continue asking quantum numbers, solving Klein-Gordon for the custom quantum system of equal masses.

12. **NEW (OPRA-Test-1 - User_select):** **Solve Schrödinger equation for a new system of equal masses (`NewSystemEqMass`) set dynamically by the user as input:**
    ```sh
    python main.py --mass_relation "m1=m2" --subsystem NewSystemEqMass --overlay_plot --num_of_equidistant_points 200
    
-The user is requested to directly insert either of the two equal masses in MeV ($m_1$ or $m_2$) and then the quantum numbers $n$ and $l$.

13. **NEW (OPRA-Test-2 - User_select):** **Solve Klein-Gordon equation for a new system of equal masses (`NewSystemEqMass`) set dynamically by the user as input:**
    ```sh
    python main.py --mass_relation "m1=m2" --diff_equation KleinGordon --subsystem NewSystemEqMass --overlay_plot --num_of_equidistant_points 200
    
-The user is requested to directly insert either of the two equal masses in MeV ($m_1$ or $m_2$) and then the quantum numbers $n$ and $l$.

14. **NEW (OPRA-Test-3 - User_select): Solver Klein Gordon equation for a custom system of equal masses and coupling constant (`NewSystemEqMass`) set by the user as input:**
    ```sh
    python main.py --mass_relation "m1=m2" --subsystem NewSystemEqMass --coupling_constant 0.07 --overlay_plot --num_of_equidistant_points 500

- **Note**: The user must manually update the code regarding the `search grid` to account for larger values of coupling constant (i.e., 0.07) and particle masses (in the order of GeV):
- `search_grid = self.RelativeDistanceSearchGrid(1e-1, 300)`
### Note 4 (related to examples 7 and 8):

In `python main.py --hamiltonian_correction_terms custom --input_file`
custom is a manual selection for Hamiltonian correction terms, i.e., '1 0 1 0 1', typically employed for ablation parameter sensitivity analyses. All possible combinations include:
- {0 0 0 0 0, 0 0 0 0 1, 0 0 0 1 0, 0 0 0 1 1, 0 0 1 0 0, 0 0 1 0 1, 0 0 1 1 0, 0 0 1 1 1, 0 1 0 0 0, 0 1 0 0 1, 0 1 0 1 0, 0 1 0 1 1, 0 1 1 0 0, 0 1 1 0 1, 0 1 1 1 0, 0 1 1 1 1, 1 0 0 0 0, 1 0 0 0 1, 1 0 0 1 0, 1 0 
0 1 1, 1 0 1 0 0, 1 0 1 0 1, 1 0 1 1 0, 1 0 1 1 1, 1 1 0 0 0, 1 1 0 0 1, 1 1 0 1 0, 1 1 0 1 1, 1 1 1 0 0, 1 1 1 0 1, 1 1 1 1 0, 1 1 1 1 1}
- choice '0 0 0 0 0]' corresponds to standard $H_0$ non-relativistic description.
- other choices such as '1 0 0 0 1' correspond to the use of $H_0 + H_1 + H_5$ Hamiltonian terms combined. 

### Note 5:

If you are unsure about the available options, you can type the following command for more details:
- ```sh 
  python main.py --help

This will provide you with a list of valid options and arguments.

## Reproducing the Schrödinger results

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the retained Figure 4–6 data mapping, plotting conventions, reproduction levels, and known v1.0 boundary. Regenerating a plot from a retained numerical array is different from rerunning an optimization: exact plot regeneration can use the retained `ablation_experiment/Experiment_<id>_<n>_<l>.txt` files, while a fresh optimization may start from a different `u0` and converge to a close but non-identical minimizer or curve.

The plotting-only utility is [`ablation_experiment/plot_experiment_overlay_ratio.py`](ablation_experiment/plot_experiment_overlay_ratio.py). It reads retained analytical and numerical columns and writes overlay, ratio, and relative-error PDFs; it does not run an optimization.

### Schrödinger output-folder logic

The output layout reflects several experiment families. Configuration-specific logs are stored under `ablation_experiment/Sch_Breit_<configuration>/`; curated TXT arrays and Figure 4–6 PDFs are stored directly under `ablation_experiment/`; CSV, energy, initialization, and parameter files are stored where available in `ablation_experiment/data/Muonium-Sch/` or `data/Muonium-Sch/`. The root `uinput/` and `ablation_experiment/uinput/` directories are separate runtime contexts and must not be merged or treated as duplicates. See the linked reproducibility document for the exact panel-to-artifact table.

# Package ablation_experiment

`ablation_experiment/batch-schrodinger.py` script addresses the batch experiments required for the ablative analysis over all possible combinations of Hamiltonian correction terms per user-defined configuration of hyperparameters (i.e., activation function, optimizer, number of hidden unints, etc.).

- Routines `folder_name_parser.py`, `read_energy_runtime.py` and `plot_best_results.py` are the post-processing operations actually required for the parameter sensitivity (ablation) analysis. 

`ablation_experiment/batch-dirac.py` script addresses the batch experiments required for the ablative analysis over all corresponding hyperparameters for the Dirac-Coulomb equation.

- Similarly, routines `Dirac_folder_name_parser.py`, `Dirac_read_energy_runtime.py` and `Dirac_plot_best_results.py` are the post-processing operations required for the parameter sensitivity (ablation) analysis of the Dirac case.

- **This package extends current implementation out of the scope of the project's deliverables. However, it is required to probe the end-to-end system's capacity to solve for various systems with different eigen-energy states and various hyperparameter configurations through an extensive ablative analysis.**

## Ablation experiment examples

**Assuming you have changed directory to `ablation_experiment/`**:
- ```sh
  cd ablation_experiment/
**Run example commands (for each distinct equation) while respecting the suggested order.**

1. **Solve Schrödinger equation with a user-defined hyper-parameter configuration for all Hamiltonian correction terms:**
   ```sh
   python batch_schrodinger.py
- Reads Schrödinger's quantum system selection (e.g., 1. Muonium) and corresponding quantum numbers from `uinput/inpsch.csv`.
- Writes output experiments (from index 1 up to 32) inside `ablation_experiment/` child folder.
- Writes output overlayed plots (`data/Muonium-Sch/Experiment_index_overlayed.pdf`) corresponding to the wave functions of the numerical and analytic solutions per experiment index.
   ```sh
   python folder_name_parser.py
- Reads all ``ablation_experiment/experiment_index.txt`` output files of previous `batch_schrodinger.py` results.
- Outputs/creates the folder `Sch_Breit_.../` named after the parameter configuration selection for the corresponding batch experiment.
   ```sh
   python read_energy_runtime.py Sch_Breit_.../
- Reads the numerical and analytical energy for all 32 experiments inside `ablation_experiment/experiment_index.txt` and
- Outputs the file `energy_vs_time.txt` inside the `Sch_Breit_.../` folder corresponding to a specific ablation experiment. 

- example row of this file is formatted as `analytical` `numerical` `runtime (seconds)` `hamiltonian terms combination`, i.e.,
-1.354020818510234e-05 -1.3540431088312487e-05 243.1402497291565 0 1 0 1 0
   ```sh
   python plot_best_results.py Sch_Breit_.../
- Reads all ``ablation_experiment/experiment_index.txt`` output files of previous `batch_schrodinger.py` results and the `energy_vs_time.txt` inside the specified `Sch_Breit_.../` folder (argument)
- Outputs/creates the file `best_results.txt` inside the specified argument/folder as well as the `energy_vs_time_plot.png` figure which actually showcases the **automated ablation analysis** of the best results obtained that lie within a pre-defined accuracy threshold (i.e., $10^-9$).

2. **Solve Dirac equation with a user-defined hyper-parameter configuration for a different network hidden unit width (ranging from 2 to 22 neurons) and different choices of model architecture:**
    ```sh
    python batch_dirac.py

- Reads Dirac one-body quantum system selection (e.g., 1. Muonium) and corresponding quantum numbers from `uinput/inpdc.csv`.
- Writes output experiments (from index 2 up to 21, 20 experiments in total) inside `ablation_experiment/` child folder.
- Writes output energies (`data/Muonium-DC/Experiment_index_F_analytical_n_l_j.csv`) corresponding to the wave functions of the analytic solutions per experiment index. 
- Writes output energies (`data/Muonium-DC/Experiment_index_G_analytical_n_l_j.csv`) corresponding to the wave functions of the analytic solutions per experiment index.
- Writes output energies (`data/Muonium-DC/Experiment_index_F_numerical_n_l_j.csv`) corresponding to the wave functions of the numerical solutions per experiment index. 
- Writes output energies (`data/Muonium-DC/Experiment_index_G_numerical_n_l_j.csv`) corresponding to the wave functions of the numerical solutions per experiment index.
- Writes output overlayed plots (`data/Muonium-DC/F_overlayed_n_l_j.csv`) corresponding to the wave functions of the analytical and numerical solutions per experiment index.
- Writes output overlayed plots (`data/Muonium-DC/G_overlayed_n_l_j.csv`) corresponding to the wave functions of the analytical and numerical solutions per experiment index.

  ```sh
   python Dirac_folder_name_parser.py
- Reads all ``ablation_experiment/experiment_index.txt`` output files of previous `batch_dirac.py` results.
- Outputs/creates the folder `Dirac_2-22_neurons_.../` named after the parameter configuration selection for the corresponding batch experiment.
   ```sh
   python Dirac_read_energy_runtime.py Dirac_2-22_neurons_.../
- Reads the numerical and analytical energy for all 32 experiments inside `ablation_experiment/experiment_index.txt` and
- Outputs the file `energy_vs_time.txt` inside the `Dirac_2-22_neurons_.../` folder corresponding to a specific ablation experiment.
- example row of this file is formatted as `analytical_distance` `numerical_distance` `analytical_without_g` `numerical_without_g` `runtime (seconds)` `number of neurons`, i.e.,
-6.098254256570179e-06 -1.970057243483314e-05 1.346149151371101e-05 -1.3605735969313137e-05 11.747704982757568 3
   ```sh
   python Dirac_plot_best_results.py Dirac_2-22_.../
- Reads all ``ablation_experiment/experiment_index.txt`` output files of previous `batch_dirac.py` results and the `energy_vs_time.txt` inside the specified `Dirac_2-22_.../` folder (argument)
- Outputs/creates the files `distances_from_energy_eigenvalue_plot.png`, `numerical_distances_from_energy_eigenvalue_plot.png`, `numerical_only_from_energy_eigenvalue_plot.png` inside the specified argument/folder as well as the `energy_vs_time` figures which actually showcase the **automated ablation analysis** of the best results obtained, that lie within a pre-defined accuracy threshold (i.e., $10^-5$).

## License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Reproducibility command examples

The command-line argument `--hamiltonian_correction_terms` must be passed as a quoted five-entry spaced mask. For example, use `"0 0 0 1 0"` or `'0 0 0 1 0'`, not `00010`.

### Schrodinger equation, ground state, H0 only

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

### Schrodinger equation with an H4 correction-term mask

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

### Dirac equation, stable reproduction profile

For stable Dirac benchmark reproduction, use the fixed-coefficient representation together with the physics-conditioned initializer:

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

For the Dirac ground state, enter:

```text
1 0
```

For the first excited Dirac state considered in the examples, use 20 hidden units and 500 grid points:

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

The historical initialization modes `random`, `fixed`, `bounded`, `xavier`, and `manual` remain available. The `physics_conditioned` mode is intended as the recommended stable Dirac initialization profile.
