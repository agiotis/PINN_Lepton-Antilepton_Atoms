import os
import pickle
import csv
from math import factorial
from typing import Any

import numpy as np
from scipy import optimize, special, integrate
from scipy.integrate import quad
from scipy.linalg import solve
from tqdm import tqdm

from .fundamental_values import FundamentalValues
from .neural_network import NeuralNetwork
from .plotting import SchrodingerPlotter, DiracPlotter, KGPlotter
from . import dirac_canonical
from .dirac_canonical_adapter import DiracStateMetadata, ShadowDiracAdapter

fundamental_values = FundamentalValues()


def normalize_hamiltonian_correction_terms(mask):
    if isinstance(mask, str):
        if not mask.strip():
            raise ValueError("Hamiltonian correction-term mask must not be empty.")
        values = mask.split()
    elif isinstance(mask, (list, tuple)):
        values = list(mask)
    else:
        raise ValueError("Hamiltonian correction-term mask must be a spaced string, list, or tuple.")

    if len(values) != 5:
        raise ValueError("Hamiltonian correction-term mask must contain exactly five entries.")

    normalized = []
    for value in values:
        if isinstance(value, str):
            if value not in ("0", "1"):
                raise ValueError("Hamiltonian correction-term mask entries must be 0 or 1.")
            normalized.append(int(value))
        elif isinstance(value, int) and value in (0, 1):
            normalized.append(value)
        else:
            raise ValueError("Hamiltonian correction-term mask entries must be 0 or 1.")

    return normalized


def save_minimizer(ures, filename="minimizer_params.pkl"):
    """Αποθηκεύει το minimizer (parameters) σε αρχείο."""
    with open(filename, 'wb') as f:
        pickle.dump(ures, f)
    print(f"Minimizer parameters saved to {filename}.")


def load_minimizer(filename="minimizer_params.pkl"):
    """Φορτώνει το minimizer (parameters) από αρχείο."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            ures = pickle.load(f)
        print(f"Minimizer parameters loaded from {filename}.")
        return ures
    else:
        print(f"No saved minimizer parameters found in {filename}.")
        return None


class QuantumSystem:
    def __init__(self, hyperparameters, system_index):
        self.fundamental_values = fundamental_values
        [mass1, mass2, reduced_mass, alpha0, particle] = self.fundamental_values.twobody(system_index)
        self.mass1 = mass1
        self.mass2 = mass2
        self.mass = reduced_mass
        self.coupling_constant = hyperparameters['coupling_constant']
        self.alpha0 = alpha0
        self.particle = particle
        self.primary_quant_num = hyperparameters['primary_quantum_number']
        self.azimuthal_quant_num = hyperparameters['azimuthal_quantum_number']
        self.total_angular_momentum = 1  # these are now included for the system of equal masses (Klein-Gordon)
        self.total_spin_momentum = 1  # these are now included for the system of equal masses (Klein-Gordon)
        self.angular_momentum1 = self.azimuthal_quant_num - 1 / 2  # Angular Momentum of Particle 1
        self.angular_momentum2 = self.azimuthal_quant_num + 1 / 2  # Angular Momentum of Particle 2
        self.k_operator = None
        self.energy = None
        self.k_optimization_const = hyperparameters["k_optimization_constant"]
        self.u0_init_method = None
        self.initial_value_u0 = None
        self.initial_value_v0 = None
        self.num_hidden_units = None
        self.relative_search_points = None
        # algorithm employed for minimization is either "BFGS", "L-BFGS-B" or "trust-constr"
        self.optimization_method = hyperparameters['optimization_method']
        self.tolerance_level = None  # Set tolerance level for minimization algorithm
        self.u_v_w_minimizer = None  # Initialize as None the minimization params for numerical solutions
        self.activation = hyperparameters['activation_function']
        self.overlay_plot = hyperparameters['overlay_plot']
        self.verbose = hyperparameters['verbose']
        self.breit = None
        self.experiment_index = hyperparameters['experiment_index']
        self.manual_u0 = hyperparameters.get("manual_u0", None)

    @staticmethod
    def create_folders():
        path = f'./uinput/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_parameters(self, hyperparameters):
        # Saving of physical parameters for corresponding equation
        if hyperparameters['diff_equation'] == 'Schrodinger':
            self.create_folders()
            with open(f'./uinput/schparam.txt', 'w', newline='') as param:
                writer = csv.writer(param)
                writer.writerow([f"Particles_System     :{self.particle}"])
                writer.writerow([f"Reduced_mass         :{self.mass}"])
                writer.writerow([f"Bohr_Radius          :{self.alpha0}"])
                writer.writerow([f"Prim._Quant._numb.   :{self.primary_quant_num}"])
                writer.writerow([f"Azimuth._Quant._numb.:{self.azimuthal_quant_num}"])

    # """Creation of the grid"""
    def RelativeDistanceSearchGrid(self):
        return np.linspace(1e-2 * self.alpha0, self.alpha0, self.relative_search_points)
        # we might need to account for the OPRA TEST of GeV masses for m1 = 1GeV and m2 = 2GeV for coupling = 0.07
        # return np.linspace(1e-2 * self.alpha0, self.alpha0, self.relative_search_points)

    def SchrodingerRadialFunction(self, r: float) -> float:
        """computation of the radial part of the exact solution"""
        scoef1 = np.power(2 / (self.primary_quant_num * self.alpha0), 3)
        scoef2 = factorial(self.primary_quant_num - self.azimuthal_quant_num - 1) / (
                2 * self.primary_quant_num * factorial(self.primary_quant_num + self.azimuthal_quant_num))
        coeff = np.sqrt(scoef1 * scoef2)
        argp = r / (self.primary_quant_num * self.alpha0)
        expon = np.exp(-argp)
        power = np.power(2 * argp, self.azimuthal_quant_num)
        lang = special.genlaguerre(self.primary_quant_num - self.azimuthal_quant_num - 1,
                                   2 * self.azimuthal_quant_num + 1, monic=False)(2 * argp)
        rnl = coeff * expon * power * lang
        return rnl

    # def coulomb_potential(self, r: float) -> float:
    #     """The Coulomb potential with smoothing for small r"""
    #     r_clamped = np.clip(r, 1e-10, None)
    #     if r < 1e-5:  # Χρήση εξομάλυνσης για μικρό r
    #         v_p = -self.fundamental_values.e_charge / r_clamped  # Dominant term
    #     else:
    #         v_p = (-self.fundamental_values.e_charge / r_clamped +
    #                (np.power(self.fundamental_values.hbarc, 2) * self.azimuthal_quant_num * (
    #                            self.azimuthal_quant_num + 1))
    #                / (2 * self.mass * np.power(r_clamped, 2)))  # in MeV
    #     return v_p

    # ORIGINAL IMPLEMENTATION
    def coulomb_potential(self, r: float) -> float:
        """The Coulomb potential"""
        v_p = (-self.fundamental_values.e_charge / r + (
                np.power(self.fundamental_values.hbarc, 2) * self.azimuthal_quant_num * (self.azimuthal_quant_num + 1))
               / (2 * self.mass * np.power(r, 2)))  # in MeV
        return v_p

    def bidning_energy(self):

        return -np.power(self.fundamental_values.hbarc, 2) / (
                2 * self.mass * np.power(self.alpha0, 2) * np.power(self.primary_quant_num, 2))

    # def solve_dirac_equation(self, hyperparameters):
    #     # Implement the Dirac equation solver here
    #     pass


class Particle(QuantumSystem):
    def __init__(self, hyperparameters, system_index):
        super().__init__(hyperparameters, system_index)
        self.save_parameters(hyperparameters)
        self.normalization_constant = None  # Initialize as None the norm_constant squared
        self.breit = normalize_hamiltonian_correction_terms(hyperparameters['hamiltonian_correction_terms'])
        if (hyperparameters['diff_equation'] == 'Schrodinger'
                and hyperparameters.get('minimization_tol_explicit', 'minimization_tol_lvl' in hyperparameters)):
            self.tolerance_level = hyperparameters.get('minimization_tol_lvl')

    def save_initial_point_u0(self, u0):
        directory = f'./data/{self.particle}-Sch/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}initpoint{self.primary_quant_num}{self.azimuthal_quant_num}.txt', 'w',
                  newline='') as file:
            file.write(f"Initial_Point: {u0}")

    """Hamiltonian Operators (non-Reduced and Reduced)"""

    def hamiltonian(self, u: list, r: float) -> float:
        """Non-Relativistic (non-reduced) Hamiltonian Operator H0"""
        v_p = self.coulomb_potential(r)
        hterm1 = (-np.power(fundamental_values.hbarc, 2) / (2 * self.mass) *
                  NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        hterm2 = v_p * NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
        return hterm1 + hterm2

    def rhamiltonian(self, u: list, r: float) -> float:
        """Non-Relativistic (reduced) Hamiltonian Operator H0"""
        v_p = self.coulomb_potential(r)
        hterm1 = (-(np.power(fundamental_values.hbarc, 2)) / (2 * self.mass)) * NeuralNetwork.secderarfunc(
            r, u, self.k_optimization_const, self.num_hidden_units, self.activation
        )
        # hterm1 = ((-np.power(fundamental_values.hbarc) / (2 * self.mass)) *
        #           NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        hterm2 = (np.power(fundamental_values.hbarc, 2) * self.azimuthal_quant_num * (self.azimuthal_quant_num + 1) /
                  (2 * self.mass * np.power(r, 2)) *
                  NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        hterm3 = v_p * NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        return hterm1 + hterm2 + hterm3

    """Breit Part (reduced and non-reduced)"""

    def radnablsq(self, u: list, r: float) -> float:
        """Reduced radial Laplacian A_l U = U'' - l(l+1) U/r^2."""
        term1 = NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
        term2 = (- self.azimuthal_quant_num * (self.azimuthal_quant_num + 1) / r ** 2
                 * NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        return term1 + term2

    def radnablfo(self, u: list, r: float) -> float:
        """Squared reduced radial Laplacian A_l^2 acting on U = r R."""
        angular = self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)
        term1 = NeuralNetwork.fourderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
        term2 = (-2 * angular / r ** 2
                 * NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        term3 = (4 * angular / r ** 3
                 * NeuralNetwork.firderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        term4 = (angular * (angular - 6) / r ** 4
                 * NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation))
        return term1 + term2 + term3 + term4

    @staticmethod
    def deltafunc(c: float, r: float) -> float:
        """Delta function"""
        term1 = np.power(r, -2)
        term2 = (2 * c * np.exp(- 2 * c * r)) / np.power(1 + np.exp(-2 * c * r), 2)
        return term1 * term2

    def breitcoef(self, r: float, u: list, v: list) -> float:
        """Apply only selected Breit corrections to the reduced radial trial function."""
        if not any(v):
            return 0.0
        cterm = 10 / self.alpha0
        correction = 0.0

        if v[0]:
            # Relativistic kinetic energy Η1
            breit1 = ((-np.power(fundamental_values.hbarc, 4) / 8
                       * (np.power(self.mass1, -3) + np.power(self.mass2, -3))) * self.radnablfo(u, r))
            correction += breit1

        if v[1]:
            # Retarded potentials Η2
            breit2 = (fundamental_values.e_charge * np.power(fundamental_values.hbarc, 2) /
                      (2 * self.mass1 * self.mass2 * r)
                      * (2 * NeuralNetwork.secderarfunc(r, u, self.k_optimization_const,
                                                             self.num_hidden_units, self.activation)
                          - 2 / r * NeuralNetwork.firderarfunc(r, u, self.k_optimization_const,
                                                             self.num_hidden_units, self.activation)
                          + (2 - self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / r ** 2
                          * NeuralNetwork.arfunc(r, u, self.k_optimization_const,
                                                 self.num_hidden_units, self.activation)))
            correction += breit2

        if v[2]:
            # Spin-Orbit coupling Η3
            chi = (self.total_angular_momentum * (self.total_angular_momentum + 1)
                   - self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)
                   - self.total_spin_momentum * (self.total_spin_momentum + 1))
            if self.mass1 == self.mass2:
                breit3 = (3 * fundamental_values.e_charge * np.power(fundamental_values.hbarc, 2) /
                          (4 * np.power(self.mass1, 2) * np.power(r, 3)) * chi)  # case m1 = m2
            else:
                # Diagonal triplet LSJ: <L.s1> = <L.s2> = chi/4.
                term1 = chi / (self.mass1 * self.mass2)
                term2 = chi / (4 * np.power(self.mass1, 2))
                term3 = chi / (4 * np.power(self.mass2, 2))
                breit3 = (fundamental_values.e_charge * np.power(fundamental_values.hbarc, 2) /
                          (2 * np.power(r, 3)) * (term1 + term2 + term3))  # case m1 != m2
            correction += breit3 * NeuralNetwork.arfunc(
                r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        if v[3]:
            # Darwin term Η4
            # The unchanged global kernel has full-space integral 2*pi.
            delta_normalized = self.deltafunc(cterm, r) / (2 * np.pi)
            breit4 = ((np.pi * fundamental_values.e_charge * np.power(fundamental_values.hbarc, 2) / 2)
                      * (np.power(self.mass1, -2) + np.power(self.mass2, -2)) * delta_normalized)
            correction += breit4 * NeuralNetwork.arfunc(
                r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        if v[4]:
            # Spin-Spin  Η5
            delta_normalized = self.deltafunc(cterm, r) / (2 * np.pi)
            slterm = ((self.total_angular_momentum * (self.total_angular_momentum + 1)
                       - self.azimuthal_quant_num * (self.azimuthal_quant_num + 1))
                      - self.total_spin_momentum * (self.total_spin_momentum + 1))  # chi = 2<L.S>
            breit51 = - 4 * np.pi / 3 * (self.total_spin_momentum *
                                         (self.total_spin_momentum + 1) - 3 / 2) * delta_normalized
            breit520 = ((self.azimuthal_quant_num * self.total_spin_momentum *
                         (self.azimuthal_quant_num + 1) * (self.total_spin_momentum + 1)) -
                        3 / 4 * (slterm + np.power(slterm, 2)))  # ls(l+1)(s+1) - 3/4 (SL^2 + SL)
            breit52 = (-1 / ((2 * self.azimuthal_quant_num + 3) * (2 * self.azimuthal_quant_num - 1) * np.power(r, 3))
                       * breit520)
            breit5 = (- fundamental_values.e_charge * np.power(fundamental_values.hbarc, 2) /
                      (self.mass1 * self.mass2) * (breit51 + breit52))
            correction += breit5 * NeuralNetwork.arfunc(
                r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        return correction

    def objective_function(self, u: list, r0: list) -> float:
        """Error function (with/without Breit terms)"""
        errfc0 = 0
        progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
                            disable=not self.verbose)
        iterations = 0
        if self.breit is not None:
            for i in progress_bar:
                iterations += 1
                errfc0 += np.power(self.hamiltonian(u, r0[i]) + self.breitcoef(r0[i], u, self.breit) - self.energy
                                   * NeuralNetwork.arfunc(r0[i], u, self.k_optimization_const,
                                                          self.num_hidden_units, self.activation), 2)
                # suppress output to show the loss every 1000 iterations only
                if iterations % 100 == 0:
                    print(f'Error Function: {errfc0}', end='\r')
            return errfc0
        else:
            for i in progress_bar:
                iterations += 1
                errfc0 += np.power(self.hamiltonian(u, r0[i]) - self.energy
                                   * NeuralNetwork.arfunc(r0[i], u, self.k_optimization_const,
                                                          self.num_hidden_units, self.activation), 2)
                # suppress output to show the loss every 300 iterations only
                if iterations % 100 == 0:
                    print(f'Error Function: {errfc0}', end='\r')
            return errfc0

    def integrand(self,
                  r: float) -> float:  # in many cases this integrand is zero (especially for low number of hidden_units
        """To be used for the normalization of the numerical solution"""
        if self.u_v_w_minimizer is not None:
            return np.power(
                NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                     self.activation), 2)
        else:
            raise ValueError("Minimizer not available. Run solve_schrodinger_equation first.")

    def normalized_arfunc(self, r: float) -> float:
        """Normalized arithmetic solution"""
        if self.normalization_constant is not None:
            return NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                        self.activation) / (
                    r * self.normalization_constant)
        else:
            raise ValueError("Normalization constant not available. Run solve_schrodinger_equation first.")

    # Equation (19) Gkrepis et al. 2023 (ICM) 2nd term
    def numerical_energy_integral(self, r: float) -> float:
        """used for the calculation of the energy"""
        return (np.power(fundamental_values.hbarc, 2) / (2 * self.mass)
                * np.power(NeuralNetwork.firderarfunc(r, self.u_v_w_minimizer, self.k_optimization_const,
                                                      self.num_hidden_units, self.activation), 2)
                + self.coulomb_potential(r) * np.power(NeuralNetwork.arfunc(r, self.u_v_w_minimizer,
                                                                            self.k_optimization_const,
                                                                            self.num_hidden_units, self.activation), 2))

    def initialize_optimization_params(self, hyperparameters, f_g_model):

        u0_init_method = hyperparameters['u0_init_method']

        if u0_init_method == 'random':  # Case 1: default (30 random values in [0..1])
            initial_value_u0 = NeuralNetwork.random_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        elif u0_init_method == 'fixed':
            initial_value_u0 = NeuralNetwork.fixed_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        elif u0_init_method == 'bounded':
            initial_value_u0 = NeuralNetwork.bounded_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        elif u0_init_method == 'manual':
            if self.manual_u0 is None:
                raise ValueError(
                    "u0_init_method='manual' but manual_u0 was not provided."
                )

            initial_value_u0 = NeuralNetwork.manual_initial_value_u0(
                self.manual_u0,
                self.num_hidden_units,
                f_g_model
            )
        elif u0_init_method == 'xavier':  # Case 2: structured initialization of parameters for a 2-layer network
            # with 2 input units (v*r + w), 10 hidden units ( o = f(v*r+w)) 1 output unit (u * o)
            # u_xavier = xavier_init_u0((2, 10))  # Parameters for the input to hidden layer
            # u * f(v*r+w) actually has 20 params for v + w
            # v_xavier = xavier_init_u0((10, 1)) # Parameters for the hidden to output layer
            # u * f(v*r+w) actually has 10 params for size(u) linear combinations of sigmoid activations per hidden unit
            # that actually correspond to 10 weights for u that linearly combine output of each hidden unit.
            if f_g_model != 0:
                u_f = NeuralNetwork.xavier_init_u0((self.num_hidden_units + 1, 1))
                vw_f = NeuralNetwork.xavier_init_u0((2, self.num_hidden_units))
                # ravel values of column vector u and row vector vw to flatten them into 1D array
                initial_value_f = np.concatenate((u_f.ravel(), vw_f.ravel()))
                u_g = NeuralNetwork.xavier_init_u0((self.num_hidden_units + 1, 1))
                vw_g = NeuralNetwork.xavier_init_u0((2, self.num_hidden_units))
                initial_value_g = np.concatenate((u_g.ravel(), vw_g.ravel()))
                initial_value_u0 = np.concatenate((initial_value_f, initial_value_g))
            else:
                u = NeuralNetwork.xavier_init_u0((self.num_hidden_units, 1))
                vw = NeuralNetwork.xavier_init_u0((2, self.num_hidden_units))
                # ravel values of column vector u and row vector vw to flatten them into 1D array
                initial_value_u0 = np.concatenate((u.ravel(), vw.ravel()))
                # [u v w], 10 hidden units, and 1 output unit
                # self.xavier_initial_value_u0 = NeuralNetwork.xavier_init_u0((30, 10))  # input to hidden layer
                # self.xavier_initial_value_v0 = NeuralNetwork.xavier_init_u0((10, 1))   # hidden to output layer
            print(initial_value_u0)
        else:
            raise ValueError(f"Unsupported u0_init_method: {u0_init_method}")
        return initial_value_u0

    def solve_schrodinger_equation(self, hyperparameters):
        # Implement the Schrödinger equation solver here
        # Use fundamental values here if needed

        # Calculate Energy eigenvalue
        # energy = -np.power(fundamental_values.hbarc, 2) / (2 * rmass * pow(alpha0, 2) * pow(nq, 2))
        # self.energy = (-np.power(self.fundamental_values.hbarc, 2) /
        #                (2 * self.mass * np.power(self.alpha0, 2) * np.power(self.primary_quant_num, 2)))
        self.energy = self.bidning_energy()
        # k optimization constant overrides main.py default argument for Schrödinger equation
        self.k_optimization_const = np.sqrt(-2 * self.energy * self.mass) / self.fundamental_values.hbarc  # in fm^{-1}

        # Following code prints several global values that may be inherited by Schrödinger or Dirac implementations
        # Print Energy eigenvalue
        print("Energy: ", self.energy)
        # Print k_optimization_constant value as calculated above
        print("k_optimization_constant: ", self.k_optimization_const)
        # Access hyperparameters here
        print("Hyperparameters:", hyperparameters)
        self.num_hidden_units = hyperparameters['num_of_hidden_units']
        self.relative_search_points = hyperparameters['num_of_equidistant_points']

        # ------------------------------------ Numerical Approximation -------------------------------------------------
        # ------------------------------------- Gkrepis et al. 2023 (IC-MSQUARE)----------------------------------------

        # Implement the Schrödinger equation solver for particles
        """calculation of the exact (radial) wave function"""

        # """Optimization Initial value for "u(r)=u(0)" (may not be used)"""
        self.initial_value_u0 = self.initialize_optimization_params(hyperparameters, 0)
        # supress outputs print("Number of hidden units/neurons of 2-layer feed forward NN per component u, v, w:",
        # self.num_hidden_units)
        print("Number of hidden units/neurons of 2-layer feed forward NN in total:", 3 * self.num_hidden_units)

        # supress I/O operations that are not needed
        # Save/keep values of decade triplets for u, v and w, respectively.
        self.save_initial_point_u0(self.initial_value_u0)

        """Creation of the grid"""
        search_grid = self.RelativeDistanceSearchGrid()
        # Convert NumPy arrays to lists in search_grid
        search_grid_list = [grid.tolist() if isinstance(grid, np.ndarray) else grid for grid in search_grid]

        # ------------------------------Minimization------------------------------------------------
        if self.optimization_method == "BFGS":
            # An issue is that the optimize.minimize() function expects the objective function to take a single
            # argument (the optimization variable), but we've modified it to also take hyperparameters. To resolve
            # this, we use a lambda function to pass the additional arguments to the objective function.
            res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                     self.initial_value_u0, method='BFGS', tol=self.tolerance_level,
                                     options={'disp': True})
        elif self.optimization_method == "L-BFGS-B":
            # An issue is that the optimize.minimize() function expects the objective function to take a single
            # argument (the optimization variable), but we've modified it to also take hyperparameters. To resolve
            # this, we use a lambda function to pass the additional arguments to the objective function.
            res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                     self.initial_value_u0, method='L-BFGS-B', tol=self.tolerance_level,
                                     options={'disp': True})
            # res1 = optimize.minimize(self.objective_function, self.random_initial_value_u0, args=search_grid,
            #                          method='BFGS', options={'disp': True})  # , 'maxiter': 100})
        elif self.optimization_method == "trust-constr":
            res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                     self.initial_value_u0, method='trust-constr', tol=self.tolerance_level,
                                     options={'disp': True})
            # res1 = optimize.minimize(self.objective_function, self.random_initial_value_u0, args=search_grid,
            #                          method='trust-constr', options={'disp': True})
        else:
            res1 = optimize.shgo(self.objective_function, bounds=[], args=search_grid)
        ures = res1.x  # Minimization parameters
        print("Minimization Parameters: ", ures)
        print("distance from u_0: \n", ures - self.initial_value_u0)
        # -------------------------------------------------------------------------------------------
        self.u_v_w_minimizer = ures  # Minimization parameters is now saved as an attribute of Particle class

        # Calculate final error function value for the optimizer
        final_error = self.objective_function(ures, search_grid_list)
        print(f"Final Error Function Value (unnormalized): {final_error}")

        # -------------------Normalization Constant Calculation---------------------------------------
        # The quad function returns a tuple with the first element being the result of the integration and the second
        # element being an estimate of the absolute error.
        normalization_constant_squared = quad(self.integrand, 0, 1e7)[0]  # Normalization constant squared
        self.normalization_constant = np.power(normalization_constant_squared, 1 / 2)  # Save as an attribute
        # self.normalization_constant = np.sqrt(normalization_constant_squared)  # Save as an attribute
        print(f"Normalization Constant: {self.normalization_constant}")

        # Check Relu and Sigmoid in n=2, l=1 why they diverge that much due to V_eff (effective potential)
        norm2 = quad(self.integrand, 0, 1e7)[0]

        T_num = quad(lambda rr: (fundamental_values.hbarc ** 2) / (2 * self.mass) *
                                (NeuralNetwork.firderarfunc(rr, self.u_v_w_minimizer, self.k_optimization_const,
                                                            self.num_hidden_units, self.activation) ** 2),
                     0, 1e7)[0] / norm2

        V_num = quad(lambda rr: self.coulomb_potential(rr) *
                                (NeuralNetwork.arfunc(rr, self.u_v_w_minimizer, self.k_optimization_const,
                                                      self.num_hidden_units, self.activation) ** 2),
                     0, 1e7)[0] / norm2

        print("T:", T_num, "Veff:", V_num, "E=T+V:", T_num + V_num)
        # End of check!

        numer_energy = quad(self.numerical_energy_integral, 0, 1e7)[0] / normalization_constant_squared
        print(f"Wave function Energy: {numer_energy}")
        print(f"Distance from analytical energy: {np.abs(numer_energy - self.energy)}")
        print(f"Breit combinations: {self.breit}")

        # Instantiate SchrodingerPlotter
        schrodinger_plotter = SchrodingerPlotter(partl=self.particle, nq=self.primary_quant_num,
                                                 lq=self.azimuthal_quant_num, alpha0=self.alpha0, verbose=self.verbose,
                                                 index=self.experiment_index)

        """ Plotting and Exporting function values to .csv and .txt formats """

        # Analytical and Numerical Solution Overlayed Plot
        schrodinger_plotter.plot(
            func_analytical=self.SchrodingerRadialFunction,
            func_numerical=self.normalized_arfunc,
            anum='quantum_numbers: ', lb=0, rb=0, ipoints=0, overlay=self.overlay_plot, idx=self.experiment_index
        )

        """Exporting Initial Point"""
        # with open(f'./data/{self.particle}-Sch/init_point{self.primary_quant_num}{self.azimuthal_quant_num}.txt', 'w',
        #          newline='') as file:
        #    file.write(f"Initial point: {u0}")
        # ----------------------------------------   SUPRESS    -------------------------------------------------
        # supress I/O operations that are not needed for ablations
        self.save_initial_point_u0(self.initial_value_u0)

        """Exporting Minimization Parameters"""
        with open(f'./data/{self.particle}-Sch/param{self.primary_quant_num}{self.azimuthal_quant_num}.csv', 'w',
                  newline='') as file:
            writer = csv.writer(file)
            for p in self.u_v_w_minimizer:
                writer.writerow([p])

        """Exporting Energy Value"""
        with open(f'./data/{self.particle}-Sch/energ{self.primary_quant_num}{self.azimuthal_quant_num}.txt', 'w',
                  newline='') as file:
            file.write(f"Anal_Energy: {self.energy}\n")
            file.write(f"Numer_Energy: {numer_energy}")

    # Additional methods specific to particles


def split_quad(func, lower_limit, upper_limit, points=None):
    """Σπάσιμο ολοκληρώματος σε δύο μέρη για σταθερότητα."""
    mid_point = (lower_limit + upper_limit) / 2

    # Χωρισμός ολοκληρώματος σε δύο μέρη
    result1, err1 = quad(func, lower_limit, mid_point, **(points or {}), epsabs=1e-10, epsrel=1e-10)
    result2, err2 = quad(func, mid_point, upper_limit, **(points or {}), epsabs=1e-10, epsrel=1e-10)

    # Συνολικό αποτέλεσμα
    total_result = result1 + result2

    print(f"Intermediate Integrals: [{result1}, {result2}], Total: {total_result}, Errors: [{err1}, {err2}]")
    return total_result


# def split_quad(func, lower_limit, upper_limit, points=None):
#     """Σπάσιμο ολοκληρώματος σε δύο μέρη για σταθερότητα."""
#     mid_point = (lower_limit + upper_limit) / 2
#     result1 = quad(func, lower_limit, mid_point, **(points or {}))[0]
#     result2 = quad(func, mid_point, upper_limit, **(points or {}))[0]
#     return result1 + result2


class Electron(Particle):
    def __init__(self, hyperparameters, system_index):
        super().__init__(hyperparameters, system_index)
        # Separate normalization constants per component F and G
        self.intnormfg = None  # Υπολειμματική, αν χρησιμοποιείται αλλού
        self.intnormf = None  # Κανονικοποίηση για F
        self.intnormg = None  # Κανονικοποίηση για G
        self.ufinf = None
        self.ufing = None
        self.numerical_solution = None
        self.total_neurons = None
        self.hidden_neurons_f = None
        self.hidden_neurons_g = None
        self.mass = fundamental_values.m_e
        # fsc = fine structure constant which is set to default value unless user defined during input
        if hyperparameters['coupling_constant'] is not None:
            self.fsc = hyperparameters['coupling_constant']
        else:
            self.fsc = fundamental_values.fsc
        self.total_angular_momentum = hyperparameters['total_angular_momentum']
        self.k_operator = hyperparameters['k_operator']
        self.grid = hyperparameters['user_search_grid']
        self.epsilon = None
        self.lambda_param = None
        self.Nspinor_normfactor = None
        self.gamma_param = None
        self.n_corrected = None
        self.coeff1 = None
        self.coeff2 = None
        self.e_charge = fundamental_values.e_charge
        self.beta = None
        self.dirac_coefficient_mode = dirac_canonical.normalize_coefficient_mode(
            hyperparameters.get("dirac_coefficient_mode")
        )
        self.dirac_canonical_adapter = None
        self.dirac_coeff_f = None
        self.dirac_coeff_g = None
        self.dirac_energy_terms = None
        self.dirac_initialization_metadata = None

    # override method save_parameters of Quantum system for Dirac instance
    def save_parameters(self, hyperparameters):
        self.create_folders()
        with open(f'./uinput/dirac_param.txt', 'w', newline='') as param:
            writer = csv.writer(param)
            writer.writerow([f"Particles_System     :{self.particle}"])
            writer.writerow([f"Electron_rest_mass         :{self.mass}"])
            writer.writerow([f"Fine Structure Const :{self.fundamental_values.fsc}"])
            writer.writerow([f"Prim._Quant._numb.   :{self.primary_quant_num}"])
            writer.writerow([f"Azimuth._Quant._numb.    :{self.azimuthal_quant_num}"])
            writer.writerow([f"total_angular_momentum    :{self.total_angular_momentum}"])
            writer.writerow([f"k_operator    :{self.k_operator}"])
            dirac_mode = dirac_canonical.normalize_coefficient_mode(hyperparameters.get("dirac_coefficient_mode"))
            if dirac_mode is not None:
                writer.writerow([f"dirac_coefficient_mode    :{dirac_canonical.external_coefficient_mode(dirac_mode)}"])


    def require_dirac_coefficient_mode(self):
        self.dirac_coefficient_mode = dirac_canonical.normalize_coefficient_mode(self.dirac_coefficient_mode)
        if self.dirac_coefficient_mode is None:
            raise ValueError("dirac_coefficient_mode is required for canonical Dirac runs")
        return self.dirac_coefficient_mode

    def configure_canonical_dirac_runtime(self):
        mode = self.require_dirac_coefficient_mode()
        metadata = DiracStateMetadata(n=int(self.primary_quant_num), l=int(self.azimuthal_quant_num),
                                      j=float(self.total_angular_momentum), kappa=float(self.k_operator),
                                      reference_energy_mev=float(self.energy),
                                      fine_structure_constant=float(self.fsc))
        self.dirac_canonical_adapter = ShadowDiracAdapter(self.num_hidden_units, self.beta, mode, self.activation,
                                                          metadata=metadata)
        layout = self.dirac_canonical_adapter.layout
        self.hidden_neurons_f = 3 * self.num_hidden_units + (1 if layout.c_f_index is not None else 0)
        self.hidden_neurons_g = 3 * self.num_hidden_units + (1 if layout.c_g_index is not None else 0)
        self.total_neurons = layout.expected_length
        return layout

    def _canonical_network_random(self):
        values = np.random.rand(3 * self.num_hidden_units)
        max_abs = np.max(np.abs(values))
        return values / max_abs if max_abs != 0 else values

    @staticmethod
    def _trapz_on_grid(values, radial):
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(values, radial))
        return float(np.trapz(values, radial))

    def _dirac_physics_component_ratio(self):
        # State-scale prior derived from the same relativistic eigenenergy already used by the Dirac residual.
        if self.epsilon is None or not (0.0 < self.epsilon < 1.0):
            raise ValueError("invalid epsilon for physics-conditioned Dirac initialization")
        return float(np.sqrt((1.0 - self.epsilon) / (1.0 + self.epsilon)))

    def _physics_conditioned_dirac_initialization(self):
        mode = self.require_dirac_coefficient_mode()
        if mode != dirac_canonical.FIXED_UNIT_COEFFICIENTS:
            raise ValueError("u0_init_method='physics_conditioned' requires --dirac-coefficient-mode fixed")
        if self.activation != 'tanh':
            raise ValueError("u0_init_method='physics_conditioned' is validated only with tanh activation")

        state = (int(self.primary_quant_num), int(self.azimuthal_quant_num), float(self.k_operator))
        if state not in ((1, 0, -1.0), (2, 0, -1.0)):
            raise ValueError("physics_conditioned Dirac initialization is validated for n=1,l=0 and n=2,l=0 with kappa=-1")

        h = self.num_hidden_units
        f_network = self._canonical_network_random()
        g_network = self._canonical_network_random()
        target_input_median = 0.5 * self.beta

        for network in (f_network, g_network):
            input_weights = network[h:2 * h]
            median_abs = float(np.median(np.abs(input_weights)))
            if median_abs <= 0 or not np.isfinite(median_abs):
                raise ValueError("invalid input-weight scale for physics-conditioned Dirac initialization")
            network[h:2 * h] *= target_input_median / median_abs

        if state == (1, 0, -1.0):
            # Ground-state large and minor components use opposite relative orientation.
            g_network[:h] *= -1.0
        else:
            # First excited s-state: preserve independent random features and supply one node carrier per component.
            f_network[:h] *= 0.02
            g_network[:h] *= 0.02
            carrier_slope = 0.5 * self.beta

            f_network[0] = -1.0
            f_network[h] = carrier_slope
            f_network[2 * h] = -carrier_slope * (1.0 / self.beta)

            g_network[0] = 1.0
            g_network[h] = carrier_slope
            g_network[2 * h] = -carrier_slope * (2.0 / self.beta)

        values = np.concatenate((f_network, g_network))
        radial = np.asarray(self.RelativeDistanceSearchGrid(1, 8e5), dtype=float)
        target_q = self._dirac_physics_component_ratio()

        spinor = self.dirac_canonical_adapter.evaluate(radial, values)
        f_norm = self._trapz_on_grid(spinor["F"] * spinor["F"], radial)
        g_norm = self._trapz_on_grid(spinor["G"] * spinor["G"], radial)
        if f_norm <= 0 or g_norm <= 0:
            raise ValueError("invalid numerical F/G norm before physics conditioning")
        q_before = float(np.sqrt(g_norm / f_norm))
        g_network = values[3 * h:6 * h].copy()
        g_network[:h] *= target_q / q_before
        values[3 * h:6 * h] = g_network

        spinor = self.dirac_canonical_adapter.evaluate(radial, values)
        joint_norm = dirac_canonical.joint_spinor_norm(spinor["F"], spinor["G"], radial)
        if joint_norm <= 0 or not np.isfinite(joint_norm):
            raise ValueError("invalid joint norm for physics-conditioned Dirac initialization")
        common_scale = 1.0 / np.sqrt(joint_norm)
        values[:h] *= common_scale
        values[3 * h:4 * h] *= common_scale

        self.dirac_initialization_metadata = {
            "method": "physics_conditioned",
            "target_input_weight_median": float(target_input_median),
            "target_minor_major_ratio": float(target_q),
            "initial_minor_major_ratio_before_conditioning": float(q_before),
            "coefficient_mode": "fixed",
        }
        return values

    def initialize_dirac_optimization_params(self, hyperparameters):
        mode = self.require_dirac_coefficient_mode()
        layout = dirac_canonical.branch_layout(self.num_hidden_units, mode)
        h = self.num_hidden_units
        init_method = hyperparameters['u0_init_method']
        if init_method == 'manual':
            if self.manual_u0 is None:
                raise ValueError("u0_init_method='manual' but manual_u0 was not provided.")
            values = np.asarray(self.manual_u0, dtype=float)
        elif init_method == 'random':
            values = np.concatenate((self._canonical_network_random(), [1.0], self._canonical_network_random(), [1.0])) if mode == dirac_canonical.LEARNABLE_COEFFICIENTS else np.concatenate((self._canonical_network_random(), self._canonical_network_random()))
        elif init_method == 'fixed':
            one = np.ones(h); network = np.concatenate((1.2 * one, 0.8 * one, one))
            values = np.concatenate((network, [1.0], network, [1.0])) if mode == dirac_canonical.LEARNABLE_COEFFICIENTS else np.concatenate((network, network))
        elif init_method == 'bounded':
            def bounded_network(): return np.concatenate((np.random.uniform(-10, 10, h), np.random.uniform(-9, 9, h), np.random.uniform(-8.9, 8.9, h)))
            values = np.concatenate((bounded_network(), [1.0], bounded_network(), [1.0])) if mode == dirac_canonical.LEARNABLE_COEFFICIENTS else np.concatenate((bounded_network(), bounded_network()))
        elif init_method == 'xavier':
            def xavier_network(): return np.concatenate((NeuralNetwork.xavier_init_u0((h, 1)).ravel(), NeuralNetwork.xavier_init_u0((2, h)).ravel()))
            values = np.concatenate((xavier_network(), [1.0], xavier_network(), [1.0])) if mode == dirac_canonical.LEARNABLE_COEFFICIENTS else np.concatenate((xavier_network(), xavier_network()))
        elif init_method == 'physics_conditioned':
            values = self._physics_conditioned_dirac_initialization()
        else:
            raise ValueError(f"Unsupported u0_init_method: {init_method}")
        values = np.asarray(values, dtype=float)
        if values.size != layout.expected_length:
            raise ValueError(f"{dirac_canonical.external_coefficient_mode(mode)} mode requires {layout.expected_length} values; got {values.size}")
        dirac_canonical.unpack_vector(values, h, mode)
        return values

    def validate_dirac_vector(self, vector):
        return dirac_canonical.unpack_vector(vector, self.num_hidden_units, self.require_dirac_coefficient_mode())

    def set_canonical_minimizer(self, vector):
        unpacked = self.validate_dirac_vector(vector)
        self.u_v_w_minimizer = np.asarray(vector, dtype=float)
        self.ufinf = unpacked.f_network.copy()
        self.ufing = unpacked.g_network.copy()
        self.dirac_coeff_f = unpacked.c_f
        self.dirac_coeff_g = unpacked.c_g
        return unpacked

    def canonical_spinor(self, radial, vector=None):
        if self.dirac_canonical_adapter is None:
            raise ValueError("canonical Dirac adapter is not configured")
        values = self.u_v_w_minimizer if vector is None else vector
        return self.dirac_canonical_adapter.evaluate(radial, values)

    def canonical_dirac_objective_terms(self, vector, radial):
        if self.dirac_canonical_adapter is None:
            self.configure_canonical_dirac_runtime()
        return self.dirac_canonical_adapter.objective(np.asarray(radial, dtype=float), vector)

    def canonical_dirac_energy(self, radial, vector=None):
        if self.dirac_canonical_adapter is None:
            raise ValueError("canonical Dirac adapter is not configured")
        values = self.u_v_w_minimizer if vector is None else vector
        self.dirac_energy_terms = self.dirac_canonical_adapter.energy_terms(np.asarray(radial, dtype=float), values)
        return float(self.dirac_energy_terms["numerator"] / self.dirac_energy_terms["denominator"])
    def DiracEnergyEigenvalue(self):
        # Calculate energy eigenvalue  (fsc is the fine structure constant): equation (14) Gkrepis et al. 2023
        termd = self.fsc / (self.primary_quant_num - np.fabs(self.k_operator) +
                            np.sqrt(np.power(self.k_operator, 2) -
                                    np.power(self.fsc, 2)))
        energy_eigenvalue = self.mass / np.sqrt(1 + np.power(termd, 2))
        return energy_eigenvalue

    def RelativeDistanceSearchGrid(self, start=1e-5, end=1e5):
        """Generate a search grid for relative distances."""
        return np.linspace(start, end, self.relative_search_points)

    # This function may return a list containing numerical values (possibly
    # float) along with values of any other type (Any).
    def DiracRadialFunction(self, r) -> list[float | Any]:
        """computation of the radial part of the exact solution"""
        self.coeff1 = np.power(2 * self.lambda_param, 3 / 2) / special.gamma(2 * self.gamma_param + 1)

        # G = reduced g radial wave function
        # F = reduced f radial wave function
        #
        funcdif0 = np.power(self.Ffunc(r), 2)
        vfuncdif0 = -self.e_charge / r * funcdif0

        # funcdif is actually the denominator of equation of 19
        funcdif = np.power(self.Ffunc(r), 2) - np.power(self.Gfunc(r), 2)

        # funcadd term is used for Equation (13) and (19)
        funcadd = np.power(self.Ffunc(r), 2) + np.power(self.Gfunc(r), 2)
        vfuncdif = -self.e_charge / r * funcdif

        # Return all values as a list
        return [funcdif0, vfuncdif0, funcdif, funcadd, vfuncdif]

    # ORIGINAL IMPLEMENTATION ------------------------------------------------------------------------------------------
    # def DiracRadialNumericalFunction(self, r) -> list[float | Any]:
    #     """computation of the radial part of the exact solution (Dirac equation)"""
    #     self.coeff1 = np.power(2 * self.lambda_param, 3 / 2) / special.gamma(2 * self.gamma_param + 1)
    #
    #     # G = reduced g radial wave function
    #     # F = reduced f radial wave function
    #     #
    #     # ----------------- CHECK HERE why funcdif0 is zero!!! maybe we need to return different value ---------------
    #     funcdif0 = np.power(self.numer_F(r), 2)
    #     vfuncdif0 = -self.e_charge / r * funcdif0
    #
    #     # funcdif is actually the denominator of equation of 19
    #     funcdif = np.power(self.numer_F(r), 2) - np.power(self.numer_G(r), 2)
    #     # we need to further check why funcdif = 0 in every iteration in numerical!!!
    #     # print('funcdif = ', funcdif)
    #
    #     # funcadd term is used for Equation (13) and (19)
    #     funcadd = np.power(self.numer_F(r), 2) + np.power(self.numer_G(r), 2)
    #     vfuncdif = -self.e_charge / r * funcdif
    #
    #     # Return all values as a list
    #     return [funcdif0, vfuncdif0, funcdif, funcadd, vfuncdif]

    def DiracRadialNumericalFunction(self, r) -> list[float | Any]:
        """computation of the radial part of the exact solution (Dirac equation)"""
        self.coeff1 = np.power(2 * self.lambda_param, 3 / 2) / special.gamma(2 * self.gamma_param + 1)

        # Calculate the radial wave functions while handling potential underflow/overflow
        numer_F_val = self.numer_F(r)  # Κανονικοποιημένο F
        numer_G_val = self.numer_G(r)  # Κανονικοποιημένο G

        funcdif0 = np.clip(np.power(numer_F_val, 2), 1e-15, None)  # Avoid zero or negative values
        vfuncdif0 = -self.e_charge / np.clip(r, 1e-15, None) * funcdif0  # Avoid division by zero

        # funcdif is the denominator of equation 19
        funcdif = np.clip(np.power(numer_F_val, 2) - np.power(numer_G_val, 2), 1e-15, None)  # Avoid zero
        vfuncdif = -self.e_charge / np.clip(r, 1e-15, None) * funcdif  # Avoid division by zero

        funcadd = np.clip(np.power(numer_F_val, 2) + np.power(numer_G_val, 2), 1e-15, None)  # Avoid zero

        # Return all values as a list
        return [funcdif0, vfuncdif0, funcdif, funcadd, vfuncdif]

    def calculate_coeff2(self, sg) -> float:
        """Second term of the exact solution"""
        coeff21 = special.gamma(2 * self.gamma_param + 1 + self.n_corrected) / (4 * self.Nspinor_normfactor * (
                self.Nspinor_normfactor - self.k_operator) * special.gamma(self.n_corrected + 1))
        if sg == 1:
            return np.sqrt((1 + self.epsilon) * coeff21)
        else:
            return np.sqrt((1 - self.epsilon) * coeff21)

    def calculate_coeff3(self, r: float) -> float:
        """Third term of the exact solution"""
        return np.power(2 * self.lambda_param * r, self.gamma_param - 1) * np.exp(-self.lambda_param * r)

    def calculate_coeff4(self, r: float, sg) -> float:
        """Fourth term of the exact solution"""
        coeff41 = (self.Nspinor_normfactor - self.k_operator) * special.hyp1f1(-self.n_corrected, 2 * self.gamma_param +
                                                                               1, 2 * self.lambda_param * r)
        coeff42 = self.n_corrected * special.hyp1f1(1 - self.n_corrected, 2 * self.gamma_param + 1,
                                                    2 * self.lambda_param * r)
        if sg == 1:
            return coeff41 - coeff42
        else:
            return coeff41 + coeff42

    # g = small component of radial part of wave function from equation (12) of Gkrepis et al. 2023
    # f = large component of radial part of wave function from equation (12) of Gkrepis et al. 2023
    # we implement them both in a single method with different sg values (1 for gfunct, 0 for ffunc)
    def fgfunc(self, r: float, sg) -> float:
        """Returns the f or g component"""
        if sg == 1:  # ------ f (large) component -------
            return self.coeff1 * self.calculate_coeff2(sg) * self.calculate_coeff3(r) * self.calculate_coeff4(r, sg)
        else:  # ------ g (small) component -------
            return -self.coeff1 * self.calculate_coeff2(sg) * self.calculate_coeff3(r) * self.calculate_coeff4(r, sg)

    def Ffunc(self, r: float) -> float:
        """reduced f component"""
        return r * self.fgfunc(r, 1)

    def Gfunc(self, r: float) -> float:
        """reduced g component"""
        return r * self.fgfunc(r, 0)

    def calculate_energy_integral_part(self, func, lower_limit, upper_limit, use_points=False, alpha=None):
        """Υπολογισμός ενεργειακού ολοκληρώματος με επιλογή κανονικοποίησης."""
        points = {'points': self.relative_search_points} if use_points else {}
        regularization = (lambda u, val: val) if alpha is None else (lambda u, val: val * np.exp(-alpha * u))

        # Λειτουργία ολοκληρώματος με split_quad
        def integrate_with_split(index):
            result = split_quad(
                lambda u: regularization(u, func(r=u)[index]), lower_limit, upper_limit, points
            )
            print(f"Index {index} integral: {result}")
            return result

        # Επιλογή ολοκληρωμάτων
        if lower_limit == 1e-8 or lower_limit == 1e-7:
            enerint1 = integrate_with_split(1) + self.mass * integrate_with_split(0)
            enerint2 = integrate_with_split(0)
            integral_type = "Integral Energy without g component"
        else:
            enerint1 = integrate_with_split(4) + self.mass * integrate_with_split(3)
            enerint2 = integrate_with_split(2)
            integral_type = "integral energy"

        # Έλεγχος για αποκλίσεις
        if np.abs(enerint2) < 1e-12:
            raise ValueError("Normalization constant too small, possible divergence.")

        enerint = enerint1 / enerint2

        # Εκτύπωση αποτελεσμάτων
        if func.__name__ == "DiracRadialFunction":
            solution_type = "Analytical"
        else:
            solution_type = "Numerical"

        if lower_limit == 1e-7:
            print(f"\n {solution_type} {integral_type}: {enerint}\n")
            print(
                f"\n {solution_type} Distance of energy (without g component) from energy eigenvalue: {enerint - self.energy}\n"
            )
        else:
            print(f"\n {solution_type} {integral_type}: {enerint}\n")
            print(
                f"\n {solution_type} distance from energy eigenvalue: {enerint - self.energy}\n"
            )

        return enerint

    # ORIGINAL IMPLEMENTATION ----------------------------------------------------------------------------------------
    # def calculate_energy_integral_part(self, func: callable, lower_limit, upper_limit, use_points=False) -> float:
    #     # calculate equation (19) - Gkrepis et al. 2023
    #     if use_points:
    #         points = {'points': self.relative_search_points}
    #     else:
    #         points = {}
    #
    #     # self.DiracRadialFunction() returns list [funcdif0, vfuncdif0, funcdif, funcadd, vfuncdif]
    #     # Example: self.DiracRadialFunction(r=u)[4] will pass r=u values to vfuncdif
    #
    #     # Check if the lower limit is 1e-7 and print integral energy without the use of g component
    #     if lower_limit == 1e-7:
    #         enerint1 = (
    #                 quad(lambda u: func(r=u)[1], lower_limit, upper_limit, **points)[0] +
    #                 self.mass * quad(lambda u: func(r=u)[0], lower_limit, upper_limit, **points)[0]
    #         )
    #
    # enerint2 = quad(lambda u: func(r=u)[0], lower_limit, upper_limit, **points)[0] enerint = enerint1 / enerint2 if
    # func.__name__ == "DiracRadialFunction": print(f"\n Analytical Integral Energy without g component: {
    # enerint}\n") print( f"\n Analytical Distance of energy (without g component) from energy eigenvalue: {enerint -
    # self.energy}\n") else: print(f"\n Numerical Integral Energy without g component: {enerint}\n") print( f"\n
    # Numerical Distance of energy (without g component) from energy eigenvalue: {enerint - self.energy}\n") else:
    # enerint1 = ( quad(lambda u: func(r=u)[4], lower_limit, upper_limit, **points)[0] + self.mass * quad(lambda u:
    # func(r=u)[3], lower_limit, upper_limit, **points)[0] )
    #
    #         enerint2 = quad(lambda u: func(r=u)[2], lower_limit, upper_limit, **points)[0]
    #         enerint = enerint1 / enerint2
    #         if func.__name__ == "DiracRadialFunction":
    #             print(f"\n Analytical integral energy: {enerint}\n")
    #             print(f"\n Analytical distance from energy eigenvalue: {enerint - self.energy}\n")
    #         else:
    #             print(f"\n Numerical integral energy: {enerint}\n")
    #             print(f"\n Numerical distance from energy eigenvalue: {enerint - self.energy}\n")
    #     return enerint

    def fg_r_squared(self, r: float) -> float:
        # return (np.power(self.fgfunc(r, 1), 2) + np.power(self.fgfunc(r, 0), 2)) * np.power(r, 2)
        # ΑΛΛΑΓΗ ΣΤΙΣ 9/12/2024
        return np.power(self.fgfunc(r, 1), 2) + np.power(self.fgfunc(r, 0), 2)

    def nGfunc(self, r: float) -> float:
        return -self.Gfunc(r) / (0.5 * self.fsc)

    # def save_initial_point_u0(self, u0):
    #     with (open(f'./data/{self.particle}-DC/initpoint{self.primary_quant_num}_{self.particle}.txt', 'w',
    #     newline='')
    #           as file):
    #         file.write(f"Initial_Point: {u0}")

    # ΤΡΕΧΟΥΣΑ ΥΛΟΠΟΙΗΣΗ  - 9/12/2024 !!!
    def objective_function(self, u: list, r0: list) -> float:
        """Canonical reduced-radial Dirac residual objective."""
        result = self.canonical_dirac_objective_terms(u, r0)
        return result["total_objective"]

    # def objective_function(self, u: list, r0: list) -> float:
    #     """Dirac Error function: equation (18)"""
    #     errfc0 = 0
    #     progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
    #                         disable=not self.verbose)
    #
    #     coeff_f = u[self.hidden_neurons_f - 1]
    #     coeff_g = u[self.total_neurons - 1]
    #     for i in progress_bar:
    #         v_p = self.coulomb_potential(r0[i])
    #         # coeff_f = 1.0
    #         # coeff_f = u[self.hidden_neurons_f - 1]
    #         # coeff_g = u[self.total_neurons - 1]
    #
    #         # f - f - g equation
    #         firterm1 = (NeuralNetwork.firderarfuncDirac(r0[i], u[:self.hidden_neurons_f - 1], self.k_optimization_const,
    #                                                     self.num_hidden_units, self.activation) +
    #                     self.k_operator / r0[i] * NeuralNetwork.arfuncDirac(r0[i], u[:self.hidden_neurons_f - 1],
    #                                                                         self.k_optimization_const,
    #                                                                         self.num_hidden_units,
    #                                                                         self.activation))
    #
    #         firterm2 = ((self.mass + self.energy - v_p) *
    #                     NeuralNetwork.arfuncDirac(r0[i], u[self.hidden_neurons_g:self.total_neurons - 1],
    #                                               self.k_optimization_const, self.num_hidden_units, self.activation)
    #                     / self.fundamental_values.hbarc)
    #
    #         firterm1_norm = firterm1 / max(abs(firterm1), 1e-15)
    #         firterm2_norm = firterm2 / max(abs(firterm2), 1e-15)
    #
    #         firterm = np.power(coeff_f * firterm1_norm - coeff_g * firterm2_norm, 2)
    #
    #         # g - g - f equation
    #         secterm1 = (NeuralNetwork.firderarfuncDirac(r0[i], u[self.hidden_neurons_g:self.total_neurons - 1],
    #                                                     self.k_optimization_const, self.num_hidden_units,
    #                                                     self.activation) -
    #                     self.k_operator / r0[i] * NeuralNetwork.arfuncDirac(r0[i],
    #                                                                         u[
    #                                                                         self.hidden_neurons_g:self.total_neurons - 1],
    #                                                                         self.k_optimization_const,
    #                                                                         self.num_hidden_units,
    #                                                                         self.activation))
    #
    #         secterm2 = ((self.mass - self.energy + v_p) *
    #                     NeuralNetwork.arfuncDirac(r0[i], u[:self.hidden_neurons_f - 1], self.k_optimization_const,
    #                                               self.num_hidden_units, self.activation)
    #                     / self.fundamental_values.hbarc)
    #
    #         secterm1_norm = secterm1 / max(abs(secterm1), 1e-15)
    #         secterm2_norm = secterm2 / max(abs(secterm2), 1e-15)
    #
    #         weight_r = np.exp(-0.001 * r0[i])
    #         # firterm = weight_r * np.power(coeff_f * firterm1_norm - coeff_g * firterm2_norm, 2)
    #         secterm = weight_r * np.power(coeff_g * secterm1_norm - coeff_f * secterm2_norm, 2)
    #
    #         errfc0 += firterm + secterm
    #
    #     return errfc0

    # ΑΡΧΙΚΗ ΥΛΟΠΟΙΗΣΗ
    # def numer_F(self, r: float) -> float:
    #     """Numerical F component (non normalized)"""
    #     return NeuralNetwork.arfunc(r, self.ufinf, self.k_optimization_const, self.num_hidden_units, self.activation)
    # def numer_G(self, r: float) -> float:
    #     """Numerical G component (non normalized)"""
    #     return NeuralNetwork.arfunc(r, self.ufing, self.k_optimization_const, self.num_hidden_units, self.activation)
    def numer_F(self, r: float) -> float:
        """Canonical numerical reduced F component."""
        return float(self.canonical_spinor(np.asarray([r], dtype=float))["F"][0])

    def numer_G(self, r: float) -> float:
        """Canonical numerical reduced G component."""
        return float(self.canonical_spinor(np.asarray([r], dtype=float))["G"][0])

    # ORIGINAL IMPLEMENTATION ----------------------------------------------------------------------------------------
    # def fgnum_squared(self, r: float) -> float:
    #     """Integrand for Normalization Condition"""
    #     return np.power(self.numer_F(r), 2) + np.power(self.numer_G(r), 2)
    def fgnum_squared(self, r: float, alpha=1e-7) -> float:
        """Canonical common normalization integrand."""
        f_val = self.numer_F(r)
        g_val = self.numer_G(r)
        return np.power(f_val, 2) + np.power(g_val, 2)

    def fgnum_squared_log(self, x: float, alpha=1e-1) -> float:
        """Integrand for Normalization Condition using log-scaling"""
        r = np.exp(x)  # Αντιστροφή του log-scaling
        f_val = np.clip(self.numer_F(r), 1e-08, None)
        g_val = np.clip(self.numer_G(r), 1e-08, None)
        return (np.power(f_val, 2) + np.power(g_val, 2)) * np.exp(x)  # Το e^x προέρχεται από το dr = e^x dx
        # return (np.power(f_val, 2) + np.power(g_val, 2)) * alpha*np.exp(-x)  # Το e^x προέρχεται από το dr = e^x dx

    # ORIGINAL IMPLEMENTATION ----------------------------------------------------------------------------------------
    # def norm_numer_f(self, r: float) -> float:
    #     """Numerical F Component (Normalized)"""
    #     return self.numer_F(r) / np.sqrt(self.intnormfg)
    def norm_numer_f(self, r: float) -> float:
        """Numerical F Component (Normalized)"""
        if self.intnormfg <= 0:
            raise ValueError("Normalization constant must be positive and non-zero.")
        return self.numer_F(r) / np.sqrt(self.intnormfg)

    def norm_numer_g(self, r: float) -> float:
        """Numerical G Component (Normalized)"""
        return self.numer_G(r) / np.sqrt(self.intnormfg)

    # def normalize_functions(self):
    #     """Κανονικοποιεί τις συναρτήσεις F και G ξεχωριστά."""
    #     # Κανονικοποίηση για F
    #     norm_constant_F = quad(lambda u: np.power(self.numer_F(u), 2), 1e-7, 1e7)[0]
    #     if norm_constant_F == 0:
    #         raise ValueError("Normalization constant for F is zero!")
    #     self.ufinf /= np.sqrt(norm_constant_F)
    #
    #     # Κανονικοποίηση για G
    #     norm_constant_G = quad(lambda u: np.power(self.numer_G(u), 2), 1e-7, 1e7)[0]
    #     if norm_constant_G == 0:
    #         raise ValueError("Normalization constant for G is zero!")
    #     self.ufing /= np.sqrt(norm_constant_G)
    #
    #     print(f"Normalization constants -> F: {norm_constant_F}, G: {norm_constant_G}")
    def normalize_functions_weighted(self, weight_f=0.7, weight_g=0.3):
        """Κανονικοποιεί τις συναρτήσεις F και G με βάση συγκεκριμένα βάρη."""
        # Υπολογισμός των ατομικών κανονικοποιητικών σταθερών
        norm_constant_F = quad(lambda u: np.power(self.numer_F(u), 2), 1e-7, 1e7)[0]
        norm_constant_G = quad(lambda u: np.power(self.numer_G(u), 2), 1e-7, 1e7)[0]

        if norm_constant_F == 0 or norm_constant_G == 0:
            raise ValueError("Normalization constant for F or G is zero!")

        # Εφαρμογή βαρυμένης κανονικοποίησης
        total_norm_constant = weight_f * norm_constant_F + weight_g * norm_constant_G
        if total_norm_constant == 0:
            raise ValueError("Total normalization constant is zero!")

        # Κανονικοποίηση παραμέτρων
        self.ufinf /= np.sqrt(total_norm_constant)  # F κανονικοποίηση
        self.ufing /= np.sqrt(total_norm_constant)  # G κανονικοποίηση

        print(f"Weighted normalization constants -> F: {weight_f * norm_constant_F}, G: {weight_g * norm_constant_G}")
        # return total_norm_constant

    def solve_dirac_equation(self, hyperparameters, lower_limit_1=1e-5, upper_limit_1=1e5, lower_limit_2=1e-7,
                             upper_limit_2=1e7):
        # Χρήση των παραμέτρων στη ρουτίνα

        # assign energy_eigenvalue for Dirac class instance
        self.energy = self.DiracEnergyEigenvalue()

        # Print Energy eigenvalue
        print("Energy Eigenvalue: ", self.energy)

        # The code that follows assumes the mathematical formulation of "An advanced algorithm to solve the
        # Dirac-Coulomb equation" (Gkrepis et al. 2023).

        # ε = E/E_0
        # E_0 =  Electron rest mass = mc^2 = self.mass
        # other parameters are calculated in corresponding methods of Electron class since they are re-used by both
        # analytical and numerical approximations
        self.epsilon = self.energy / self.mass

        # λ parameter
        self.lambda_param = self.mass / self.fundamental_values.hbarc * np.sqrt(1 - np.power(self.epsilon, 2))

        # N = normalization factor for the Dirac spinor where α represents the Dirac matrices and sqrt((1-epsilon)^2)
        # is a normalization factor.
        self.Nspinor_normfactor = self.fsc / np.sqrt(1 - np.power(self.epsilon, 2))

        # γ = sqrt(κ^2 - α^2)
        self.gamma_param = np.sqrt(np.power(self.k_operator, 2) - np.power(self.fsc, 2))

        # n' = N −γ  is the quantum number correction dubbed n_corrected
        self.n_corrected = self.epsilon * self.Nspinor_normfactor - self.gamma_param
        # nd = self.primary_quant_num - self.total_angular_momentum - (1 / 2)
        nd = self.primary_quant_num - np.abs(self.k_operator)
        # Check if n' is equal to nd
        if self.n_corrected == nd:
            # Proceed with the next part of the code
            # ...
            print("n_corrected= %.18f, nd= %.18f" % (self.n_corrected, nd))
            print("n_corrected is equal to nd. Continuing with the next part of the code.")
        else:
            # Handle the case when n' is not equal to nd
            # ...
            print("n_corrected= %.15f, nd= %.15f" % (self.n_corrected, nd))
            print("n_corrected is NOT equal to nd. Handling this case.")
            # "here we check what will happen if nd differs from n_corrected"
            # self.n_corrected = nd

        # Print k_optimization_constant value as calculated above
        print("k_optimization_constant: ", self.k_optimization_const)
        # Access hyperparameters here
        print("Hyperparameters:", hyperparameters)
        self.num_hidden_units = hyperparameters['num_of_hidden_units']
        self.relative_search_points = hyperparameters['num_of_equidistant_points']

        print("\nDirac parameters: ε = %.15f, N = %.15f, γ = %.15f, λ = %.15f, n' = %.15f\n" % (
            self.epsilon, self.Nspinor_normfactor, self.gamma_param, self.lambda_param, self.n_corrected))

        # ---------- Calculate coefficients of f, g as well as F, G, and radial function values ------------------------
        # --------------------------------- required for integral calculations -----------------------------------------

        # self.DiracRadialFunction() of returns list [funcdif0, vfuncdif0, funcdif, funcadd, vfuncdif]
        # Example: self.DiracRadialFunction(r=u)[4] will pass r=u values to vfuncdif

        # -------------------------------------------------2 OPTIONS ---------------------------------------------------
        # ----------------------------------1. Adaptive Quadrature (Default Behavior): ---------------------------------
        # optimize.quad dynamically adjusts the number of points based on the local behavior of the integrand. This can
        # be beneficial for integrands with varying levels of complexity.
        # Cons: (i) Runtime Determination: The number of points used is determined at runtime, which might be less
        # predictable. (ii) Potential for Overhead: In some cases, the algorithm might spend more time determining the
        # optimal number of points, leading to overhead. This is used when --search_grid=0

        # ----------------------- 2.1 User-Defined Search Grid: a hyperparameter is set as follows: --------------------
        # hyperparameters['user_search_grid'] = 1 or set the argument "--search_grid = 1"
        # Pros: (i) Control: You have explicit control over the grid points, allowing you to customize the integration
        # based on your understanding of the problem. (ii) Predictable: You know the number of points in advance,
        # providing predictability in terms of performance.
        # Cons: (i) Manual Tuning: You need to manually tune the grid points, which might require experimentation to
        # find an optimal value. (ii) Less Adaptive: It might not adapt well to integrands with varying complexity.
        # This options sets a user pre-defined num of points for the quad operation such as:
        # integrate.quad(function, 1e-5, 1e5, points=1000)[0]
        # OR
        # -------------------- 2.2 User-Defined Search Grid: a user-defined method sets the values for r ---------------
        # Example:  RelativeDistanceSearchGrid(self, start=1e-5, end=1e5, num_points=1000):
        #           """Generate a search grid for relative distances within given limits."""
        #               return np.linspace(start, end, num_points)

        self.relative_search_points = hyperparameters['num_of_equidistant_points']
        # ORIGINAL IMPLEMENTATION -------------------------------------------------------------------------------------
        # # Check the value of self.grid
        # if self.grid == 0:  # OPTION 1: default behavior
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-5, upper_limit=1e5,)
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-7, upper_limit=1e7)
        #
        # elif self.grid == 1:  # OPTION 2.1: user-defined points for quad
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-5, upper_limit=1e5,
        #                                         use_points=True)
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-7, upper_limit=1e7,
        #                                         use_points=True)
        # else:  # OPTION 2.2: user-defined method for getting the points for quad
        #     # Generate the search grid for relative distances
        #     search_grid = self.RelativeDistanceSearchGrid()
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=search_grid[0],
        #                                         upper_limit=search_grid[-1])
        #
        #     # Generate another search grid for relative distances
        #     search_grid2 = self.RelativeDistanceSearchGrid(1e-7, 1e7)
        #     self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=search_grid2[0],
        #                                         upper_limit=search_grid2[-1])

        # Έλεγχος της τιμής του self.grid
        if self.grid == 0:  # OPTION 1: default behavior
            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-5, upper_limit=1e5)
            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-7, upper_limit=1e7)

        elif self.grid == 1:  # OPTION 2.1: user-defined points for quad
            alpha_points = 1e-6

            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-5, upper_limit=1e5,
                                                use_points=True, alpha=alpha_points)
            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=1e-7, upper_limit=1e7,
                                                use_points=True, alpha=alpha_points)

        else:  # OPTION 2.2: user-defined method for getting the points for quad
            # Δημιουργία του search grid για τις σχετικές αποστάσεις
            search_grid = self.RelativeDistanceSearchGrid()
            alpha_custom = 1e-5  # προσαρμοσμένη τιμή alpha
            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=search_grid[0],
                                                upper_limit=search_grid[-1], alpha=alpha_custom)

            # Δημιουργία άλλου search grid για τις σχετικές αποστάσεις
            search_grid2 = self.RelativeDistanceSearchGrid(1e-7, 1e7)
            self.calculate_energy_integral_part(func=self.DiracRadialFunction, lower_limit=search_grid2[0],
                                                upper_limit=search_grid2[-1], alpha=alpha_custom)

        print(f"Integral OK Value: {quad(lambda u: self.fg_r_squared(r=u), 1e-5, 8e6)}")

        # Instantiate DiracPlotter
        dirac_plotter = DiracPlotter(partl=self.particle, nq=self.primary_quant_num,
                                     lq=self.azimuthal_quant_num, jq=self.total_angular_momentum, verbose=self.verbose)
        """ Plotting and Exporting function values to .csv and .txt formats """
        # Analytical and Numerical Solution Overlayed Plot
        # if self.overlay_plot == 1:
        #     dirac_plotter.plot(f_component=self.Ffunc, g_component=self.nGfunc, anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num, jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot)
        # else:
        #     dirac_plotter.plot(f_component=self.Ffunc, g_component=self.Gfunc, anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num, jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot)

        # save initial quantum parameters to dirac_param.txt inside uinput
        # ------------------------------------ SUPRESS FOR batch_dirac.py ----------------------------------------------
        # self.save_parameters(hyperparameters)

        # ------------------------------------ Numerical Approximation -------------------------------------------------
        # ---------------------------------- Gkrepis et al. 2023 (MEDEX) -----------------------------------------------

        # The following code assumes the use of Neural Networks to approximate F and G components (see equation 16).
        # Both F and G are modelled by a two-layer NN wherein each u, v, w employs self.num_of_hidden_units = 8 hidden
        # neurons plus 1 neuron for coefficient c_f and c_g, respectively. Our implementation combines F and G in a
        # single u0 vector of size: (3 * hidden_units + 1) * f_g_model = (3 * 8 + 1) * 2 = 50 hidden units in total.

        # ----------------------------- Optimization constants and Initialization --------------------------------------
        self.k_optimization_const = self.lambda_param
        self.beta = self.k_optimization_const
        layout = self.configure_canonical_dirac_runtime()
        self.initial_value_u0 = self.initialize_dirac_optimization_params(hyperparameters)
        print("Number of hidden units/neurons of 2-layer feed forward NN per component u, v, w:", self.num_hidden_units)
        print("Dirac coefficient mode:", dirac_canonical.external_coefficient_mode(self.dirac_coefficient_mode))
        print("Canonical Dirac vector length:", layout.expected_length)
        self.numerical_solution = 1

        # Save/keep values of two decade triplets for u, v and w, plus 1 F and 1 G coefficients, respectively.
        self.save_initial_point_u0(self.initial_value_u0)

        """Creation of the grid"""
        search_grid = self.RelativeDistanceSearchGrid(1, 8e5)
        # Convert NumPy arrays to lists in search_grid
        search_grid_list = [grid.tolist() if isinstance(grid, np.ndarray) else grid for grid in search_grid]

        # self.beta is already configured as the beta coefficient of equation (16).
        # boundaries are used to constrain optimization that follows
        boundaries = [(-10, 10) for i in range(self.total_neurons)]

        # Έλεγχος αν υπάρχουν αποθηκευμένες τιμές του minimizer
        if hyperparameters['u0_init_method'] == 'physics_conditioned':
            filename = (f"minimizer_params_{self.experiment_index}_dirac_n{int(self.primary_quant_num)}_"
                        f"l{int(self.azimuthal_quant_num)}_h{self.num_hidden_units}_p{self.relative_search_points}_"
                        f"{self.activation}_{self.optimization_method}_"
                        f"{dirac_canonical.external_coefficient_mode(self.dirac_coefficient_mode)}_physics_conditioned.pkl")
        else:
            filename = f"minimizer_params_{self.experiment_index}.pkl"
        saved_ures = load_minimizer(filename)
        if saved_ures is not None:
            ures = np.asarray(saved_ures, dtype=float)
            self.validate_dirac_vector(ures)
            print("Using saved minimizer parameters.")
        else:
            # Ενέργειες για το optimization αν δεν υπάρχουν αποθηκευμένα δεδομένα
            print("No saved parameters found. Running optimization...")
            # -------------------------------------------- Minimization-------------------------------------------------
            if self.optimization_method == "BFGS":
                res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                         self.initial_value_u0, method='BFGS', tol=self.tolerance_level,
                                         bounds=boundaries,
                                         options={'disp': True})
            elif self.optimization_method == "L-BFGS-B":
                lbfgsb_options = {'disp': True}
                dirac_tol = self.tolerance_level
                if hyperparameters['u0_init_method'] == 'physics_conditioned':
                    lbfgsb_options['maxls'] = 50
                    dirac_tol = hyperparameters['minimization_tol_lvl']
                res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                         self.initial_value_u0, method='L-BFGS-B', tol=dirac_tol,
                                         bounds=boundaries,
                                         options=lbfgsb_options)
            elif self.optimization_method == "trust-constr":
                res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
                                         self.initial_value_u0, method='trust-constr', tol=self.tolerance_level,
                                         bounds=boundaries,
                                         options={'disp': True})
            else:
                res1 = optimize.shgo(self.objective_function, bounds=[], args=search_grid)
            ures = res1.x  # Minimization parameters
            save_minimizer(ures, filename)

        print("Minimization Parameters: ", ures)
        print("distance from u_0: \n", ures - self.initial_value_u0)

        # Calculate final error function value for the optimizer
        final_error = self.objective_function(ures, search_grid_list)
        print(f"Final Error Function Value (unnormalized): {final_error}")
        # --------------------------------------------------------------------------------------------------------------
        self.set_canonical_minimizer(ures)

        print(f"Parameters u of f: {self.ufinf[:self.num_hidden_units]}\n "
              f"Parameters v of f: {self.ufinf[self.num_hidden_units: 2 * self.num_hidden_units]} \n "
              f"Parameters w of f: {self.ufinf[2 * self.num_hidden_units: self.hidden_neurons_f]}")
        print(f"Parameters u of g: {self.ufing[: self.num_hidden_units]}\n "
              f"Parameters v of g: {self.ufing[self.num_hidden_units: 2 * self.num_hidden_units]} \n "
              f"Parameters w of g: {self.ufing[2 * self.num_hidden_units: self.hidden_neurons_g]}")
        """Extraction of arithmetic results"""

        # ----------------------------------- Novelty agiotis 20-11-2024 -----------------------------------------------
        # Κανονικοποίηση των uf και ug
        # self.normalize_functions()
        # self.intnormfg = self.normalize_functions_weighted(weight_f=0.3, weight_g=0.7)

        # Υπολογισμός κανονικοποιητικής σταθεράς
        # self.intnormfg = quad(lambda u: self.fgnum_squared(r=u, alpha=1e-7), 1e-9, 1e9)[0]
        # print(f"""Normalization constant: {self.intnormfg}""")

        # Νέα κανονικοποίηση με log-scaling
        r_min, r_max = 1e-7, 1e7
        x_min, x_max = np.log(r_min), np.log(r_max)

        self.intnormfg = self.dirac_canonical_adapter.norm(np.asarray(search_grid_list, dtype=float), ures)
        print(f"Canonical joint normalization constant: {self.intnormfg}")

        # self.intnormfg = quad(lambda u: self.fgnum_squared(r=u), 1e-9, 1e3, epsabs=1e-10, epsrel=1e-10)[0]

        numer_energy = self.canonical_dirac_energy(np.asarray(search_grid_list, dtype=float), ures)
        print(f"Canonical numerical integral energy: {numer_energy}")
        print(f"Canonical distance from energy eigenvalue: {numer_energy - self.energy}")
        """ Plotting and Exporting function values to .csv and .txt formats """

        """ Plotting and Exporting function values to .csv and .txt formats """
        # if self.overlay_plot == 1:
        #     dirac_plotter.plot(analytical_F=self.Ffunc, numerical_F=self.numer_F,
        #                        # analytical_G=self.nGfunc, numerical_G=self.norm_numer_g,
        #                        analytical_G=self.Gfunc, numerical_G=self.numer_G,
        #                        anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num,
        #                        jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot, idx=self.experiment_index)
        # else:
        #     dirac_plotter.plot(analytical_F=self.Ffunc, numerical_F=self.numer_F,
        #                        analytical_G=self.Gfunc, numerical_G=self.numer_G,
        #                        anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num,
        #                        jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot, idx=self.experiment_index)
        # Keep the existing DiracPlotter sampling convention while supplying canonical numerical F/G arrays.
        plot_radial = np.linspace(1e-2, 5e5, 900)
        plot_spinor = self.dirac_canonical_adapter.evaluate(plot_radial, ures)
        plot_scale = np.sqrt(self.intnormfg)
        _, canonical_export_table = dirac_canonical.export_table(
            plot_radial, plot_spinor["F"] / plot_scale, plot_spinor["G"] / plot_scale
        )
        dirac_plotter.plot(analytical_F=self.Ffunc, numerical_F=self.norm_numer_f,
                           analytical_G=self.Gfunc, numerical_G=self.norm_numer_g,
                           anum='quantum numbers: ',
                           nq=self.primary_quant_num, lq=self.azimuthal_quant_num,
                           jq=self.total_angular_momentum,
                           overlay=self.overlay_plot, idx=self.experiment_index,
                           canonical_numerical_data=canonical_export_table,
                           dirac_coefficient_mode=dirac_canonical.external_coefficient_mode(self.dirac_coefficient_mode),
                           dirac_initialization_method=hyperparameters['u0_init_method'])
        # Numerical Solution Overlayed Plot
        # if self.overlay_plot == 1:
        #     dirac_plotter.plot(f_component=self.norm_numer_f, g_component=self.norm_numer_g, anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num, jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot, solver_mode="numerical")
        # else:
        #     dirac_plotter.plot(f_component=self.norm_numer_f, g_component=self.norm_numer_g, anum='quantum numbers: ',
        #                        nq=self.primary_quant_num, lq=self.azimuthal_quant_num, jq=self.total_angular_momentum,
        #                        overlay=self.overlay_plot, solver_mode="numerical")

        # save initial quantum parameters to dirac_param.txt inside uinput
        self.save_parameters(hyperparameters)


class Positronium(QuantumSystem):

    def __init__(self, hyperparameters, system_index):
        super().__init__(hyperparameters, system_index)
        # self.normalization_constant = hyperparameters[
        #     "integral_normalization_constant"]  # Initialize as None the norm_constant squared
        # Example: Calculate values for Positronium with given A, Z, N
        self.system = hyperparameters['two_body_system']
        self.maxv = None
        # self.mass = self.mass1   # we overide the reduced mass since for the Positronium system either m1 or m2.
        self.normalization_constant = None
        self.A = self.fundamental_values.A  # Example value for A = 137
        self.Z = self.fundamental_values.Z  # Example value for Z = 68 or Z = 1
        # fsc = fine structure constant which is set to default value unless user defined during input
        if hyperparameters['coupling_constant'] is not None:
            self.fsc = hyperparameters['coupling_constant']
        else:
            self.fsc = fundamental_values.fsc
        # N = self.A - self.Z  # Calculate N if required for the code below
        # [mass1, mass2, rmass, _, partl] = fundamental_values.twobody(4, self.A, self.Z, N) # old case for K-G
        # self.mass1 = mass1
        # self.mass2 = mass2
        # self.mass = rmass
        # self.particle = partl
        self.save_parameters(hyperparameters)
        self.method = hyperparameters['numerical_method']

    def save_initial_point_u0(self, u0):
        # Define the directory path
        directory = f'./data/{self.particle}-KG'

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the file path
        file_path = f'{directory}/initpoint{self.primary_quant_num}{self.azimuthal_quant_num}.txt'

        # Write the initial point to the file
        with open(file_path, 'w', newline='') as file:
            file.write(f"Initial_Point: {u0}")

    def save_parameters(self, hyperparameters):
        with open(f'./uinput/klein_gordon_param.txt', 'w', newline='') as param:
            writer = csv.writer(param)
            writer.writerow([f"Particles_System     :{self.particle}"])
            writer.writerow([f"{self.particle}_mass1         :{self.mass1}"])
            writer.writerow([f"{self.particle}_rest_mass         :{self.mass}"])
            writer.writerow([f"Fine Structure Const :{self.fsc}"])
            writer.writerow([f"Coupling Constant :{self.coupling_constant}"])
            writer.writerow([f"Prim._Quant._numb.   :{self.primary_quant_num}"])
            writer.writerow([f"Azimuth._Quant._numb.    :{self.azimuthal_quant_num}"])
            writer.writerow([f"total_angular_momentum    :{self.total_angular_momentum}"])
            writer.writerow([f"k_operator    :{self.k_operator}"])

    def RelativeDistanceSearchGrid(self, start=1e-6, end=30):  # we can change it directly in solve_KG_equation
        return np.linspace(start, end, self.relative_search_points)

    # Energy Calculation
    # def E_nl(self) -> float:  # equation 16
    #     alpha = self.fsc
    #     Za2 = np.power(self.Z * alpha, 2)
    #
    #     # Correct calculation for the denominator
    #     denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
    #                            np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)
    #
    #     # Keep the reduced mass (not * 2 as in Scot et al. 1992)
    #     energy_in_hartree = self.mass * np.power(1 + (Za2 / denominator), -0.5)
    #
    #     # Convert the energy from Hartree to MeV
    #     hartree_to_mev = 0.0272114  # Conversion factor: 1 Hartree = 0.0272114 MeV
    #     energy_in_mev = energy_in_hartree * hartree_to_mev
    #
    #     return energy_in_mev

    # def E_nl(self) -> float:
    #     alpha = self.fsc
    #     Za2 = np.power(self.Z * alpha, 2)
    #
    #     # Correct calculation for the denominator
    #     denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
    #                            np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)
    #
    #     # In the following case no M_0 exists for the calculation of E_nl so we keep the reduced mass only
    #     result = self.mass * np.power(1 + (Za2 / denominator), -0.5)
    #
    #     # Subtract rest energy (binding energy)
    #     mc2 = 2 * self.mass  # 2mc^2 for the two-body system
    #     binding_energy = result - mc2
    #
    #     return binding_energy

    # def E_nl(self) -> float:
    #     alpha = self.fsc
    #     Za2 = np.power(self.Z * alpha, 2)
    #
    #     # Correct calculation for the denominator
    #     denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
    #                            np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)
    #
    #     # Compute E_nl directly in MeV (if it's already supposed to be in MeV)
    #     result_mev = self.mass * np.power(1 + (Za2 / denominator), -0.5)
    #     # print("result mev = ", result_mev)
    #
    #     # Subtract rest energy (binding energy) directly in MeV
    #     mc2_mev = self.mass  # In MeV units (rest energy of the electron)
    #     binding_energy_mev = (result_mev - mc2_mev) * 1e6
    #     # binding_energy_mev = result_mev
    #     # This step is no longer needed
    #     # hartree_to_mev = 0.0272114
    #     # binding_energy_mev = binding_energy_hartree * hartree_to_mev
    #
    #     # Return the binding energy in MeV (no conversion needed)
    #     return binding_energy_mev

    def E_nl(self) -> float:
        alpha = self.fsc  # Fine structure constant
        Za2 = np.power(self.Z * alpha, 2)  # (Z * alpha)^2

        # Calculate the relativistic denominator as per the formula in your reference
        denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
                               np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)

        # Compute the relativistic energy eigenvalue E_nl
        result_mev = self.mass * np.power(1 + Za2 / denominator, -0.5)  # Result in MeV

        # Binding energy is defined as the difference from rest energy mc^2
        binding_energy_mev = result_mev - self.mass  # In MeV, expected to be negative

        # Convert binding energy to eV
        # binding_energy_ev = binding_energy_mev * 1e6  # Now in eV
        binding_energy_ev = binding_energy_mev

        # Return the binding energy in eV
        return binding_energy_ev

    # def E_nl(self) -> float:
    #     alpha = self.fsc
    #     Za2 = np.power(self.Z * alpha, 2)
    #
    #     # Correct calculation for the denominator
    #     denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
    #                            np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)
    #
    #     # Compute E_nl in Hartree units
    #     result_hartree = self.mass1 * np.power(1 + (Za2 / denominator), -0.5)
    #
    #     # Subtract rest energy (binding energy)
    #     mc2_hartree = 2 * self.mass1  # In Hartree units (we need the rest mass of the electron)
    #     # mc2_hartree = self.mc2
    #     binding_energy_hartree = result_hartree - mc2_hartree
    #     print("Binding energy = ",binding_energy_hartree)
    #     # Convert binding energy to MeV
    #     hartree_to_mev = 0.0272114
    #     binding_energy_mev = binding_energy_hartree * hartree_to_mev
    #
    #     return binding_energy_mev

    # def E_nl(self) -> float:  # equation 16  can also be computed in a simpler fashion (H. Kosmas will provide the
    # code) alpha = self.fsc Za2 = np.power(self.Z * alpha, 2)
    #
    #     # Correct calculation for the denominator
    #     denominator = np.power((self.primary_quant_num - self.angular_momentum2) +
    #                            np.sqrt(np.power(self.angular_momentum2, 2) - Za2), 2)
    #
    #     # In the following case no M_0 exists for the cacluation of E_nl so we keep the reduced mass only (not * 2 as
    #     # in Scot et al. 1992). If required then we must also make a "two_body" version of it
    #     result = self.mass1 * np.power(1 + (Za2 / denominator), -0.5)
    #     return result

    # def E_nl(self):  # equation 16
    #
    #     alpha = self.fsc  # if not set during input then by default it's the self.fundamental_values.fsc
    #     Za2 = np.power(self.Z * alpha, 2)
    #
    #     denominator = np.power((self.primary_quant_num - self.azimuthal_quant_num - 0.5) +
    #                            np.sqrt(np.power(self.azimuthal_quant_num + 0.5, 2) - Za2), 2)
    #
    #     result = self.mass1 * np.power(1 + (Za2 / denominator), -0.5)
    #
    #     return result

    def mi(self, l: int) -> float:
        return np.power(l + 0.5, 2) - np.power(self.Z * self.fsc, 2)

    def ar(self, r: float, betta: float) -> float:
        return r * betta

    def betta(self, Energy) -> float:
        # Ask Prof Kosmas if self.mass * 2 should be used here? I think not as it is in the calculation of the E_nl
        # If affected we need a "two_body" version
        return (2 / self.fundamental_values.hbarc) * np.sqrt(
            np.power(self.mass, 2) - np.power(Energy, 2))

    def l2(self, Energy, betta) -> float:
        return (2 * self.Z * self.fsc * Energy) / (self.fundamental_values.hbarc * betta)

    def R_nl(self, r: float) -> float:
        n = self.primary_quant_num
        l = self.azimuthal_quant_num
        mi_v = self.mi(l)
        betta_v = self.betta(self.E_nl())
        ar_v = self.ar(r, betta_v)

        d = 2 * mi_v + 1
        Gd = special.gamma(d)

        nprim = n - l - 1
        cnpr = d + nprim
        Gcnp = special.gamma(cnpr)

        Gnl = special.gamma(n - l)

        Frac = Gcnp / (Gnl * Gd * Gd * (self.E_nl() * (d + 2 * n - 2 * l - 2) + betta_v * self.Z * self.fsc))
        CoefN = np.sqrt(Frac) * np.power(betta_v, 1.5)

        Exps = np.exp(-ar_v / 2)
        rm = np.power(ar_v, mi_v - 0.5)
        H1F1 = special.hyp1f1(-nprim, d, ar_v)

        Rwf = CoefN * rm * Exps * H1F1
        return Rwf

    def U_nl(self, r: float) -> float:
        return self.R_nl(r) * r

    # We need to check the following!
    # def coulomb_potential(self, r: float) -> float:  # overrides QuantumSystem's V(r) for KG case
    #     """The Coulomb potential"""
    #     # v_p = -self.Z * self.fsc / r
    #     v_p = -np.power(fundamental_values.e_charge, 2) / r
    #
    #     return v_p

    # def coulomb_potential(self, r: float) -> float: # overrides QuantumSystem's V(r) for KG case
    #     """The Coulomb potential with a small cutoff to avoid singularity at r = 0"""
    #     epsilon = 1e-6  # Small cutoff value
    #     r_eff = max(r, epsilon)  # Avoid dividing by zero or very small r
    #     v_p = -np.power(fundamental_values.e_charge, 2) / r_eff
    #     return v_p

    def coulomb_potential(self, r: float) -> float:
        """The Coulomb potential with scaling for MeV units"""
        epsilon = 1e-6  # Small cutoff value
        r_eff = max(r, epsilon)  # Avoid very small r to prevent division by zero

        # Convert potential to MeV by dividing by hbarc (197.326 MeV·fm)
        v_p = -np.power(self.fundamental_values.e_charge, 2) / (r_eff * self.fundamental_values.hbarc)

        return v_p

    def U_minimizer(self, r: float) -> float:
        """Numerical solution for U(r)"""
        return NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                    self.activation)

    def dU_dr_minimizer(self, r: float) -> float:
        """Numerical solution for U(r)"""
        return NeuralNetwork.firderarfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                          self.activation)

    def d2U_dr_minimizer(self, r: float) -> float:
        """Second derivative of U(r)"""
        return NeuralNetwork.secderarfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                          self.activation)

    def hamiltonian_eq_4(self, u: list, r: float) -> float:
        """Non-Relativistic (non-reduced) Hamiltonian Operator of equation (4) - KLeinGordon"""
        v_p = self.coulomb_potential(r)

        # Compute second derivative term using numpy
        d2U_dr = NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute arfunc value using numpy
        arfunc_value = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute centrifugal term using numpy
        centrifugal_term = (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) * arfunc_value / np.power(r, 2)

        # Compute relativistic correction term using numpy
        # we need to check that self.mass is actually the mc^2 term
        relativistic_correction = (np.power((self.E_nl() - v_p), 2) - np.power(self.mass, 2)) / np.power(
            self.fundamental_values.hbarc, 2) * arfunc_value

        # Return the Hamiltonian value
        return d2U_dr - centrifugal_term - relativistic_correction

    def hamiltonian_eq_3_1_non_relativistic(self, u: list, r: float) -> float:  # first term of equation 3.1
        """Non-relativistic Hamiltonian Operator for the Klein-Gordon equation (excluding relativistic correction)."""
        V_r = self.coulomb_potential(r)
        # dV_dr = np.gradient(V_r, r)
        dV_dr = -1 / np.power(r, 2)

        # Compute second derivative term using numpy
        d2R_dr2 = NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute first derivative term using firderarfunc
        dR_dr = NeuralNetwork.firderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute R(r) using arfunc
        R_r = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute first derivative correction term
        first_derivative_correction = (2 / r + dV_dr / (self.E_nl() - V_r)) * dR_dr

        # Compute centrifugal term
        centrifugal_term = (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r, 2) * R_r

        # Return the non-relativistic Hamiltonian value
        return d2R_dr2 + first_derivative_correction - centrifugal_term

    def hamiltionian_relativistic_correction(self, u: list, r: float) -> float:  # last term of equation 3.1
        """Relativistic correction term for the Klein-Gordon equation."""
        V_r = self.coulomb_potential(r)

        # Compute R(r) using arfunc
        R_r = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute relativistic correction term wherein now M_0 = 2 x self.mass = 2 * mc^2
        M_0 = 2 * self.mass
        relativistic_correction = - ((np.power((self.E_nl() - V_r), 2) - np.power(M_0, 2)) / (
                4 * np.power(self.fundamental_values.hbarc, 2))) * R_r

        # Return the relativistic correction term
        return relativistic_correction

    def hamiltonian_eq_3_1_two_body(self, u: list, r: float) -> float:
        """Combined Hamiltonian Operator of equation (4) - Klein-Gordon"""

        # Handle singularity at r = 0
        epsilon = 1e-6
        r_eff = max(r, epsilon)

        V_r = self.coulomb_potential(r)

        # Derivative of potential with respect to r
        dV_dr = np.power(self.fundamental_values.e_charge, 2) / np.power(r_eff, 2)

        # Compute second and first derivative terms using the neural network
        d2R_dr2 = NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
        dR_dr = NeuralNetwork.firderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
        R_r = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute first derivative correction term (with singularity handling)
        first_derivative_correction = (2 / r_eff + dV_dr / (self.E_nl() - V_r)) * dR_dr

        # Compute centrifugal term
        centrifugal_term = (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r_eff, 2) * R_r

        # Relativistic correction term
        M_0 = 2 * self.mass  # Mass term for two-body system
        relativistic_correction = - ((np.power((self.E_nl() - V_r), 2) - np.power(M_0, 2)) / (
                4 * np.power(self.fundamental_values.hbarc, 2))) * R_r

        # Return the combined Hamiltonian value
        return d2R_dr2 + first_derivative_correction - centrifugal_term - relativistic_correction

    def hamiltonian_eq_10(self, u: list, r: float) -> float:
        """Relativistic (reduced) Hamiltonian Operator for Equation (10)"""

        # Compute second derivative term using numpy
        d2U_dr = NeuralNetwork.secderarfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute arfunc value using numpy
        arfunc_value = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

        # Compute new centrifugal term using numpy
        new_centrifugal_term = ((self.azimuthal_quant_num * (self.azimuthal_quant_num + 1) - np.power(self.Z * self.fsc,
                                                                                                      2)) * arfunc_value / np.power(
            r, 2))

        # Compute term with Z alpha
        Z_alpha_term = ((2 * self.Z * self.fsc * self.E_nl()) / (fundamental_values.hbarc * r)) * arfunc_value

        # Compute relativistic correction term wherein now M_0 = 2 x self.mass = 2 * mc^2
        M_0 = 2 * self.mass
        # Compute relativistic correction term using numpy
        relativistic_correction = (np.power(self.E_nl(), 2) - np.power(M_0, 2) / np.power(fundamental_values.hbarc
                                                                                          , 2)) * arfunc_value

        # Return the Hamiltonian value
        return d2U_dr - new_centrifugal_term + Z_alpha_term + relativistic_correction

    def objective_function_eq_10(self, u: list, r0: list) -> float:
        """Error function for Klein-Gordon equation"""
        errfc0 = 0
        progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
                            disable=not self.verbose)
        iterations = 0

        # Calculate the normalization factor using the current u
        # normalization_factor = self.compute_normalization_factor(u, self.k_optimization_const, self.num_hidden_units,
        #                                                          self.activation, self.E_nl())

        for i in progress_bar:
            r = r0[i]
            # U = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
            hamiltonian_value = self.hamiltonian_eq_10(u, r)

            # Summing the squared Hamiltonian value directly
            F = np.power(hamiltonian_value, 2)
            errfc0 += F

            # Suppress output to show the loss every 100 iterations only
            if iterations % 100 == 0:
                print(f'Error Function: {errfc0}', end='\r')

            iterations += 1
        return errfc0

    # MY ORIGINAL IMPLEMENTATION THAT WORKS!
    # def objective_function(self, u: list, r0: list) -> float:
    #     """Error function for Klein-Gordon equation"""
    #     errfc0 = 0
    #     progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
    #                         disable=not self.verbose)
    #     iterations = 0
    #
    #     for i in progress_bar:
    #         r = r0[i]
    #         U = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)
    #         if self.system == "combined_hamiltonian":
    #             hamiltonian_value = self.hamiltonian_eq_3_1_two_body(u, r)
    #             F = np.power(hamiltonian_value, 2)
    #         elif self.system == "relativistic":
    #             hamiltonian_value = self.hamiltionian_relativistic_correction(u, r)
    #             F = np.power(hamiltonian_value, 2)
    #         elif self.system == "non-relativistic":
    #             hamiltonian_value = self.hamiltonian_eq_3_1_non_relativistic(u, r)
    #             F = np.power(hamiltonian_value, 2)
    #         else:
    #             hamiltonian_value = self.hamiltonian_eq_4(u, r)
    #             F = np.power(hamiltonian_value * U, 2)
    #
    #         errfc0 += F
    #
    #         # Suppress output to show the loss every 100 iterations only
    #         if iterations % 100 == 0:
    #             print(f'Error Function (Iteration {iterations}): {errfc0}', end='\r')
    #
    #         iterations += 1
    #
    #     # Return the final error value after all points have been processed
    #     return errfc0

    # New objective with penalty in loss function at boundaries r=0 and r=inf
    def objective_function(self, u: list, r0: list) -> float:
        """Error function for Klein-Gordon equation with boundary conditions at r = 0 and r = maxv (∞)."""
        errfc0 = 0
        progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
                            disable=not self.verbose)
        iterations = 0

        for i in progress_bar:
            r = r0[i]
            U = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

            if self.system == "combined_hamiltonian":
                hamiltonian_value = self.hamiltonian_eq_3_1_two_body(u, r)
                F = np.power(hamiltonian_value, 2)
            elif self.system == "relativistic":
                hamiltonian_value = self.hamiltionian_relativistic_correction(u, r)
                F = np.power(hamiltonian_value, 2)
            elif self.system == "non-relativistic":
                hamiltonian_value = self.hamiltonian_eq_3_1_non_relativistic(u, r)
                F = np.power(hamiltonian_value, 2)
            else:
                hamiltonian_value = self.hamiltonian_eq_4(u, r)
                F = np.power(hamiltonian_value * U, 2)

            errfc0 += F

            iterations += 1

        # Penalty at boundaries
        lambda_0 = 0.1  # Adjust penalty at r=0
        penalty_r0 = lambda_0 * NeuralNetwork.arfunc(0, u, self.k_optimization_const, self.num_hidden_units,
                                                     self.activation) ** 2

        lambda_inf = 0.1  # Adjust penalty at r=maxv
        r_max = self.maxv
        penalty_r_max = lambda_inf * NeuralNetwork.arfunc(r_max, u, self.k_optimization_const, self.num_hidden_units,
                                                          self.activation) ** 2

        total_error = errfc0 + penalty_r0 + penalty_r_max

        # Regularization (weight decay)
        weight_decay = 1e-2  # Increase regularization
        regularization_term = weight_decay * np.sum(np.square(u))

        total_loss = total_error + regularization_term

        return total_loss

    def objective_function_advanced(self, u: list, r0: list) -> float:
        """Error function for Klein-Gordon equation under normalization constraint"""
        errfc0 = 0
        progress_bar = tqdm(range(self.relative_search_points), desc="Calculating Error", unit="point",
                            disable=not self.verbose)

        # Define the singular point based on the physical system's characteristics (similar to your previous approach)
        r_s = (self.fundamental_values.Z * self.fundamental_values.e_charge ** 2) / (
                4 * np.pi * self.fundamental_values.eps_0 * self.E_nl())

        for i in progress_bar:
            r = r0[i]
            U = NeuralNetwork.arfunc(r, u, self.k_optimization_const, self.num_hidden_units, self.activation)

            # Calculate the Hamiltonian value for the current system
            if self.system == "combined_hamiltonian":
                hamiltonian_value = self.hamiltonian_eq_3_1_two_body(u, r)
            elif self.system == "relativistic":
                hamiltonian_value = self.hamiltionian_relativistic_correction(u, r)
            elif self.system == "non-relativistic":
                hamiltonian_value = self.hamiltonian_eq_3_1_non_relativistic(u, r)
            else:
                hamiltonian_value = self.hamiltonian_eq_4(u, r)

            F = np.power(hamiltonian_value * U, 2)
            errfc0 += F

            if i % 100 == 0:
                print(f'Error Function: {errfc0}', end='\r')

        # Handling normalization with segmented integration to avoid singularities
        if self.u_v_w_minimizer is not None:
            # Segment 1: From 1e-4 to just before r_s
            normalization_part1, _ = quad(lambda r: self.integrand_with_minimizer(r), 1e-4, r_s - 1e-4, epsabs=1.49e-8,
                                          epsrel=1.49e-8)

            # Segment 2: From just after r_s to the upper bound (maxv or np.max(r0))
            normalization_part2, _ = quad(lambda r: self.integrand_with_minimizer(r), r_s + 1e-4, np.max(r0),
                                          epsabs=1.49e-8, epsrel=1.49e-8)

            # Combine the results from both segments
            normalization_constant = normalization_part1 + normalization_part2

        else:
            # Segment 1: From 1e-4 to just before r_s
            normalization_part1, _ = quad(lambda r: self.integrand_without_minimizer(r, U), 1e-4, r_s - 1e-4,
                                          epsabs=1.49e-8, epsrel=1.49e-8)

            # Segment 2: From just after r_s to the upper bound (maxv or np.max(r0))
            normalization_part2, _ = quad(lambda r: self.integrand_without_minimizer(r, U), r_s + 1e-4, np.max(r0),
                                          epsabs=1.49e-8, epsrel=1.49e-8)

            # Combine the results from both segments
            normalization_constant = normalization_part1 + normalization_part2

        if normalization_constant == 0:
            raise ValueError("Normalization constant is zero, check the integrand calculation.")

        # Normalize the error function by dividing by the normalization constant
        normalized_error = errfc0 / normalization_constant
        return normalized_error

    def integrand_without_minimizer(self, r: float, u: float) -> float:
        """Integrand function for the case where the minimizer is not available"""
        Vr = self.coulomb_potential(r)

        return np.sqrt(np.power(np.abs(self.E_nl() - Vr), 2)) * np.abs(u)

    def integrand_with_minimizer(self, r: float) -> float:
        """Integrand function when the minimizer is available"""
        Vr = self.coulomb_potential(r)
        Ur = NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                  self.activation)

        return np.sqrt(np.power(np.abs(self.E_nl() - Vr), 2)) * np.abs(Ur)

    def integrand(self, r: float) -> float:
        """Integrand function for the integral"""
        Vr = self.coulomb_potential(r)
        if self.u_v_w_minimizer is not None:
            Ur = NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                      self.activation)
        else:
            raise ValueError("Minimizer not available. Run solve_KG_equation first.")

        # Alternative 1: Direct Integration suggests to divide only by "U(r)^2" - works well
        # return Ur * Ur

        # Alternative 2: Weighted Integration
        # return (self.E_nl() - Vr) * np.abs(Ur)

        # Alternative 3: Adjusted Normalization Using Potential
        # return np.sqrt(np.abs(self.E_nl() - Vr)) * Ur

        # Alternative 4: Adjusted Normalization Using Potential
        # return np.sqrt(np.abs(self.E_nl() - Vr)) * Ur * Ur

        # Alternative 5: Adjusted Normalization Using Potential
        # return np.sqrt(np.power(np.abs(self.E_nl() - Vr), 2)) * np.abs(Ur)

        # Alternative 6: Adjusted Normalization Using Potential
        return (1 / np.abs(self.E_nl() - Vr)) * Ur

        # Alternative 7: Adjusted Normalization Using Potential
        # return (1 / np.sqrt(np.abs(self.E_nl() - Vr))) * Ur

        # CHECK THIS ON 24/07/2024 !!
        # comment out following line to test the default case as stated in Kosmas, Tsoulos et al. KG paper.
        # return (self.E_nl() - Vr) * Ur * Ur

    def normalized_arfunc(self, r: float) -> float:
        """Normalized arithmetic solution"""
        # if self.normalization_constant is not None:
        #     return NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
        #                                 self.activation) / (
        #             r * self.normalization_constant)
        if self.normalization_constant is not None:
            return NeuralNetwork.arfunc(r, self.u_v_w_minimizer, self.k_optimization_const, self.num_hidden_units,
                                        self.activation) / (r * self.normalization_constant)
        else:
            raise ValueError("Normalization constant not available. Run klein_gordon equation first.")

    def numerical_energy_integral_one_body(self, r: float) -> float:
        """Used for the calculation of the energy"""
        term1, _ = integrate.quad(lambda r: np.power(self.d2U_dr_minimizer(r), 2), 0, self.maxv)
        term2, _ = integrate.quad(
            lambda r: (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r, 2) * np.power(
                self.U_minimizer(r), 2), 0, self.maxv)
        term3, _ = integrate.quad(lambda r: self.coulomb_potential(r) * np.power(self.U_minimizer(r), 2), 0, self.maxv)
        term4, _ = integrate.quad(lambda r: np.power(self.U_minimizer(r), 2), 0, self.maxv)

        # mc_squared = self.mass ** 2   # Original G. Gkartzonikas implementation

        # DEFAULT - Adapted implementation
        # mc_squared = self.mass

        # New implementation from Scot et al. 1992 Physical Review A, V. 45, N. 7 where M_0*c^2 = 2 * m_0 * c
        mc_squared = self.mass * 2

        result = (np.sqrt((term1 + term2 + term3) / term4 + mc_squared) + (term3 / term4))
        return result

    def numerical_energy_integral_relativistic(self, r: float) -> float:
        """Used for the calculation of the energy with relativistic corrections"""
        try:
            # Define a small value to avoid singularities at r = 0
            r_values = np.linspace(1e-10, self.maxv, num=1000)  # Avoid zero to prevent division by zero
            V_r_values = self.coulomb_potential(r_values)

            # Compute the gradient of V_r with respect to r_values
            # dV_dr_values = np.gradient(V_r_values, r_values)

            dV_dr_values = np.power(fundamental_values.e_charge, 2) / np.power(r, 2)

            # Ensure no division by zero or NaN values
            E_nl_val = self.E_nl()
            safe_dV_dr = np.where(dV_dr_values == 0, 1e-10, dV_dr_values)
            safe_V_r = np.where(V_r_values == E_nl_val, E_nl_val + 1e-10, V_r_values)
            safe_V_r = np.where(np.isnan(safe_V_r), E_nl_val + 1e-10, safe_V_r)

            # Second derivative term
            term1, _ = integrate.quad(lambda r: np.power(self.d2U_dr_minimizer(r), 2), 1e-10, self.maxv, limit=1000,
                                      epsabs=1e-6, epsrel=1e-6)

            # Debugging: Check term1
            print(f"term1: {term1}")

            # First derivative correction term
            term2, _ = integrate.quad(
                lambda r: (2 / r + np.interp(r, r_values, safe_dV_dr) / (
                        E_nl_val - np.interp(r, r_values, safe_V_r))) * self.dU_dr_minimizer(r) * self.U_minimizer(
                    r),
                1e-10, self.maxv, limit=1000, epsabs=1e-6, epsrel=1e-6)

            # Debugging: Check term2
            print(f"term2: {term2}")

            # Centrifugal term
            term3, _ = integrate.quad(
                lambda r: (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r, 2) * np.power(
                    self.U_minimizer(r), 2),
                1e-10, self.maxv, limit=1000, epsabs=1e-6, epsrel=1e-6)

            # Debugging: Check term3
            print(f"term3: {term3}")

            # Relativistic correction term (with the minus sign)
            M_0 = 2 * self.mass  # Use 2 * mass for two-body problem  tested the value 4 to see if we get negative
            term4, _ = integrate.quad(
                lambda r: - ((np.power((E_nl_val - np.interp(r, r_values, safe_V_r)), 2) - np.power(M_0, 2)) / (
                        4 * np.power(self.fundamental_values.hbarc, 2))) * np.power(self.U_minimizer(r), 2),
                1e-10, self.maxv, limit=1000, epsabs=1e-6, epsrel=1e-6)

            # Debugging: Check term4
            print(f"term4: {term4}")

            # Normalization term
            term5, _ = integrate.quad(lambda r: np.power(self.U_minimizer(r), 2), 1e-10, self.maxv, limit=1000,
                                      epsabs=1e-6, epsrel=1e-6)

            # Debugging: Check term5
            print(f"term5: {term5}")

            # Compute the result
            result = np.sqrt((term1 + term2 + term3 + term4) / term5 + M_0) + (term3 / term5)

            # Debugging: Check result
            print(f"result: {result}")

            return result
        except Exception as e:
            print(f"An error occurred: {e}")
            return np.nan

    def numerical_energy_integral_non_relativistic(self, r: float) -> float:
        """Used for the calculation of the energy for the non-relativistic Hamiltonian."""
        # Second derivative term
        term1, _ = integrate.quad(lambda r: np.power(self.d2U_dr_minimizer(r), 2), 0, self.maxv)

        # First derivative correction term
        V_r = self.coulomb_potential(r)
        # Ensure V_r is scalar when passed to the integration function  - Needs checking
        if isinstance(V_r, np.ndarray):
            V_r = V_r.item()
        # # Compute dV_dr manually if necessary    ---- WE WILL CHECK THIS TOO
        # r_values = np.linspace(0, self.maxv, num=1000)
        # V_r_values = self.coulomb_potential(r_values)
        # dV_dr_values = np.gradient(V_r_values, r_values)
        # dV_dr = np.interp(r, r_values, dV_dr_values)
        # If working then go to the next line ...
        # dV_dr = np.gradient(V_r, r)
        dV_dr = np.power(fundamental_values.e_charge, 2) / np.power(r, 2)

        term2, _ = integrate.quad(
            lambda r: (2 / r + dV_dr / (self.E_nl() - V_r)) * self.dU_dr_minimizer(r) * self.U_minimizer(r), 0,
            self.maxv)

        # Centrifugal term
        term3, _ = integrate.quad(
            lambda r: (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r, 2) * np.power(
                self.U_minimizer(r), 2), 0, self.maxv)

        # Normalization term
        term4, _ = integrate.quad(lambda r: np.power(self.U_minimizer(r), 2), 0, self.maxv)

        # Compute the result
        result = (term1 + term2 + term3) / term4
        return result

    def numerical_energy_integral_combined(self, r: float) -> float:
        """Improved calculation of the energy for the combined Hamiltonian with better handling of singularities."""

        # Define singular point r_s for handling singularities
        r_s = (self.fundamental_values.Z * self.fundamental_values.e_charge ** 2) / (
                4 * np.pi * self.fundamental_values.eps_0 * self.E_nl())

        # Tolerances and subdivision limits
        new_epsabs = 1e-6  # Relaxed absolute tolerance
        new_epsrel = 1e-6  # Relaxed relative tolerance
        subdivision_limit = 500  # Increased subdivision limit

        # Split integration at r_s and handle singularities carefully
        # Part 1: From 1e-4 to just before r_s
        term1_part1_left, err_left = integrate.quad(
            lambda r: np.power(self.d2U_dr_minimizer(r), 2),
            1e-4, r_s - 1e-4,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        # Part 2: From just after r_s to 1.0
        term1_part1_right, err_right = integrate.quad(
            lambda r: np.power(self.d2U_dr_minimizer(r), 2),
            r_s + 1e-4, 1.0,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        # Combine both segments for term1
        term1_part1 = term1_part1_left + term1_part1_right

        # Part 3: From 1.0 to maxv (handling the tail of the integral)
        term1_part2, _ = integrate.quad(
            lambda r: np.power(self.d2U_dr_minimizer(r), 2),
            1.0, self.maxv,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        # Combine all parts to get final result for term1
        term1 = term1_part1 + term1_part2

        # First derivative correction term
        V_r = self.coulomb_potential(r)
        if isinstance(V_r, np.ndarray):
            V_r = V_r.item()

        dV_dr = np.power(self.fundamental_values.e_charge, 2) / np.power(r, 2)

        # Handle term2 with extra splitting and relaxed tolerances
        # Term2: Splitting near r_s
        term2_part1_left, err_left = integrate.quad(
            lambda r: (2 / r + dV_dr / (self.E_nl() - V_r)) * self.dU_dr_minimizer(r),
            1e-4, r_s - 1e-4,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        term2_part1_right, err_right = integrate.quad(
            lambda r: (2 / r + dV_dr / (self.E_nl() - V_r)) * self.dU_dr_minimizer(r),
            r_s + 1e-4, 1.0,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        term2_part1 = term2_part1_left + term2_part1_right

        # Term 2: From r_s + 1e-4 to r_max
        term2_part2, err2 = integrate.quad(
            lambda r: (2 / r + dV_dr / (self.E_nl() - V_r)) * self.dU_dr_minimizer(r),
            r_s + 1e-4, self.maxv,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        # Combine term2 results
        term2 = term2_part1 + term2_part2

        # Centrifugal term
        term3, _ = integrate.quad(
            lambda r: (self.azimuthal_quant_num * (self.azimuthal_quant_num + 1)) / np.power(r, 2) * np.power(
                self.U_minimizer(r), 2), 0, self.maxv,
            epsabs=new_epsabs,
            epsrel=new_epsrel,
            limit=subdivision_limit
        )

        # Relativistic correction term
        M_0 = self.mass  # Use self.mass for more consistent results
        term4, _ = integrate.quad(lambda r: - ((np.power((self.E_nl() - V_r), 2) - np.power(M_0, 2)) / (
                4 * np.power(self.fundamental_values.hbarc, 2))) * np.power(self.U_minimizer(r), 2), 0, self.maxv,
                                  epsabs=new_epsabs,
                                  epsrel=new_epsrel,
                                  limit=subdivision_limit
                                  )

        # Normalization term
        term5, _ = integrate.quad(lambda r: np.power(self.U_minimizer(r), 2), 0, self.maxv,
                                  epsabs=new_epsabs,
                                  epsrel=new_epsrel,
                                  limit=subdivision_limit
                                  )

        # Combine all terms and ensure non-negativity
        combined_term = term1 + term2 + term3 + term4
        if combined_term + M_0 * term5 < 0:
            combined_term = max(combined_term, 0)  # Ensure non-negative value

        # Compute final result
        result = np.sqrt(combined_term / term5 + M_0) + (term3 / term5)
        return result

    def numerical_energy_integral_eq10(self, r: float) -> float:
        """Used for the calculation of the energy for Equation 10 Hamiltonian."""

        # Second derivative term
        term1, _ = integrate.quad(lambda r: np.power(self.d2U_dr_minimizer(r), 2), 0, self.maxv)

        # Centrifugal term adjusted with (ℓ(ℓ + 1) − (Zα)^2), applied directly to U(r)
        term2, _ = integrate.quad(
            lambda r: ((self.azimuthal_quant_num * (self.azimuthal_quant_num + 1) - np.power(self.Z * self.fsc, 2)) /
                       np.power(r, 2)) * self.U_minimizer(r), 0, self.maxv)

        # Zα term applied directly to U(r)
        term3, _ = integrate.quad(
            lambda r: ((2 * self.Z * self.fsc * self.E_nl()) / (fundamental_values.hbarc * r)) * self.U_minimizer(r),
            0, self.maxv)

        # Relativistic correction term applied directly to U(r)
        term4, _ = integrate.quad(
            lambda r: ((np.power(self.E_nl(), 2) - np.power(self.mass, 2)) / np.power(fundamental_values.hbarc, 2)) *
                      self.U_minimizer(r), 0, self.maxv)

        # Normalization term, still squared
        term5, _ = integrate.quad(lambda r: np.power(self.U_minimizer(r), 2), 0, self.maxv)

        # Check if the sum of the terms inside the square root is non-negative
        combined_term = (term1 + term2 + term3 + term4)
        if combined_term + np.power(self.mass, 2) * term5 < 0:
            combined_term = max(combined_term, 0)  # Ensure non-negative value

        # Compute the result
        result = (np.sqrt(combined_term / term5 + np.power(self.mass, 2)) + (term2 / term5))
        return result

    def initialize_optimization_params(self, hyperparameters, f_g_model):

        u0_init_method = hyperparameters['u0_init_method']

        if u0_init_method == 'random':  # Case 1: default (30 random values in [0..1])
            initial_value_u0 = NeuralNetwork.random_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        elif u0_init_method == 'fixed':
            initial_value_u0 = NeuralNetwork.fixed_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        elif u0_init_method == 'bounded':
            initial_value_u0 = NeuralNetwork.bounded_initial_value_u0(self.num_hidden_units, f_g_model)
            print(initial_value_u0)
        else:
            raise ValueError(f"Unsupported u0_init_method: {u0_init_method}")
        return initial_value_u0

    def analytical_energy_integral(self, r0: list) -> float:
        """Calculate the analytical energy as the integral over R_nl and potential."""

        def integrand(r):
            Rwf = self.R_nl(r)
            Vr = self.coulomb_potential(r)
            return Rwf ** 2 * Vr  # Energy density at each r

        # Perform the integration over the same grid as the numerical energy
        E_analytical, _ = quad(integrand, 0, np.max(r0))
        return E_analytical

    def normalize_R_nl(self, r0: list) -> float:
        """Normalize the radial wave function R_nl."""

        def integrand(r):
            Rwf = self.R_nl(r)
            return Rwf ** 2

        normalization_constant, _ = quad(integrand, 0, np.max(r0))
        return normalization_constant

    def solve_KG_equation(self, hyperparameters):
        # Implement Klein-Gordon equation solver

        # Print k_optimization_constant value as calculated above
        print("k_optimization_constant: ", self.k_optimization_const)
        # Access hyperparameters here
        print("Hyperparameters:", hyperparameters)
        self.num_hidden_units = hyperparameters['num_of_hidden_units']
        self.relative_search_points = hyperparameters['num_of_equidistant_points']

        # ---------------------- A. Giotis  16/09/2024 --------------------------
        # Now we exploit the r_i points to calculate integral energy of the analytical approach for our two-body system.
        search_grid = self.RelativeDistanceSearchGrid()  # this case is for leptonic systems (1e-6 to 10, 20, or 30)

        # Convert NumPy arrays to lists in search_grid
        search_grid_list = [grid.tolist() if isinstance(grid, np.ndarray) else grid for grid in search_grid]

        # Calculate Energy eigenvalue
        # Comment out the following line to run standard model. Otherwise, follow up A. Giotis 16/09/2024
        self.energy = self.E_nl()

        # Calculate Energy eigenvalue in Hartree
        # binding_energy = self.E_nl()
        # # Convert the energy from Hartree to MeV
        # hartree_to_mev = 0.0272114  # Conversion factor: 1 Hartree = 0.0272114 MeV
        # self.energy = binding_energy * hartree_to_mev
        #
        # print(f"Energy in Hartree: {binding_energy}")
        # print(f"Energy converted to MeV: {self.energy}")

        # ---------------------- A. Giotis  16/09/2024 --------------------------
        # Big change in analytical energy calculation
        # Calculate the analytical energy based on the integral
        # analytical_energy = self.analytical_energy_integral(r0)

        # k optimization constant
        # self.k_optimization_const = hyperparameters["k_optimization_constant"]

        # Original approach considers either of the following 2 lines to Print Energy eigenvalue without requiring to
        # define the search grid beforehand....
        # print("Total energy (E_nl): ", self.energy)
        # binding_energy_mev = self.energy - self.mass
        print("Binding energy (E_nl - 2mc^2) for given quantum numbers: ", self.energy)

        # ---------------------- A. Giotis  16/09/2024 --------------------------
        # Big change in analytical energy calculation
        # Calculate the analytical energy based on the integral
        self.maxv = np.max(search_grid_list)
        # print("self.maxv: ", self.maxv)

        # Step 1: Compute Analytical Energy (integral-based)
        # analytical_energy = self.analytical_energy_integral(search_grid_list)
        # print(f"Analytical Energy (integral-based): {analytical_energy}")

        # Step 2: Compute Normalization Constant for Radial Wave Function
        # analytical_norm_constant = self.normalize_R_nl(search_grid_list)
        # print(f"Analytical normalization constant (integral-based): {analytical_norm_constant}")

        # Step 3: Ensure normalization constant is not zero
        # if analytical_norm_constant == 0:
        #     raise ValueError("Normalization constant is zero. Check the radial wave function and integral calculation.")

        # Step 4: Print Normalized Analytical Energy
        # norm_anal_energy = analytical_energy / analytical_norm_constant
        # print(f"Normalized Analytical (integral) Energy: {norm_anal_energy}")

        # ------------------------------------ Numerical Approximation -------------------------------------------------
        # ------------------------------------------- A. Giotis --------------------------------------------------------

        # Here we use the reduced radial part of the wave function that has an exact solution
        """calculation of the exact (radial) wave function"""

        # """Optimization Initial value for "u(r)=u(0)" (may not be used)"""
        self.initial_value_u0 = self.initialize_optimization_params(hyperparameters, 0)
        # supress outputs print("Number of hidden units/neurons of 2-layer feed forward NN per component u, v, w:",
        # self.num_hidden_units)
        print("Number of hidden units/neurons of 2-layer feed forward NN in total:", 3 * self.num_hidden_units)

        # supress I/O operations that are not needed
        # Save/keep values of decade triplets for u, v and w, respectively.
        self.save_initial_point_u0(self.initial_value_u0)

        """Creation of the grid"""
        # ---------------------------SPECIAL CASE of OPRA TESTING -------------------------------------------

        # search_grid = self.RelativeDistanceSearchGrid(1e-1, 300)  # # this case tests the 0.1 GeV mass with 0.07
        # coupling_constant. Also seems to be working for mass 1GeV and 0.07 coupling constant: python main.py
        # --mass_relation "m1=m2" --subsystem "NewSystemEqMass" --overlay_plot --num_of_equidistant_points 500
        # --coupling_constant 0.07

        # ------------------------------Minimization------------------------------------------------
        if self.optimization_method == "BFGS":
            # An issue is that the optimize.minimize() function expects the objective function to take a single
            # argument (the optimization variable), but we've modified it to also take hyperparameters. To resolve
            # this, we use a lambda function to pass the additional arguments to the objective function.
            res1 = optimize.minimize(
                lambda u: self.objective_function(u, search_grid_list),
                self.initial_value_u0,
                method='BFGS',
                tol=self.tolerance_level,  # Tight tolerance level
                options={
                    'disp': True,  # Display optimization details
                    'gtol': 1e-6,  # Gradual gradient stopping
                    'maxiter': 2000  # Increase the maximum iterations if needed
                }
            )
        elif self.optimization_method == "L-BFGS-B":
            bounds = [(-1, 1)] * len(self.initial_value_u0)  # Example bounds for all variables
            res1 = optimize.minimize(
                lambda u: self.objective_function(u, search_grid_list),
                self.initial_value_u0,
                method='L-BFGS-B',
                bounds=bounds,  # Add bounds to avoid large parameter values
                tol=self.tolerance_level,
                options={
                    'disp': True,  # Display optimization details
                    'gtol': 1e-6,  # Slightly relaxed gradient tolerance
                    'maxiter': 2000,  # Maximum number of iterations
                    'maxfun': 3000,  # Maximum number of function evaluations
                    'ftol': 1e-9,  # Tolerance for change in function value
                    'eps': 1e-9  # Smaller step size for gradient approximation
                }
            )
        elif self.optimization_method == "trust-constr":

            res1 = optimize.minimize(
                lambda u: self.objective_function(u, search_grid_list),
                self.initial_value_u0,
                method='trust-constr',
                tol=self.tolerance_level,  # Global tolerance for stopping criteria
                options={
                    'disp': True,  # Display the optimization details
                    'xtol': 1e-8,  # Very tight tolerance on x (parameter changes)
                    'gtol': 1e-6,  # Tighter tolerance on gradient (stops when gradient norm is small)
                    'barrier_tol': 1e-6,  # Tolerance for the barrier term (constraints)
                    'maxiter': 2000,  # Increase maximum number of iterations
                    'initial_tr_radius': 0.1  # Start with a smaller trust region radius
                    # 'factorization_method': 'NormalFactorization'  # Change to NormalFactorization
                }
            )
            # ## comment out following lines that return minimizer according to standard objective of equation (3.1) from
            # # Scot et al.
            # res1 = optimize.minimize(lambda u: self.objective_function(u, search_grid_list),
            #                          self.initial_value_u0, method='trust-constr', tol=self.tolerance_level,
            #                          options={'disp': True})
            #
            # # The following lines reflect the objective function of equation (10) from the KG paper of Kosmas O. et al.
            # # res1 = optimize.minimize(lambda u: self.objective_function_eq_10(u, search_grid_list),
            # #                          self.initial_value_u0, method='trust-constr', tol=self.tolerance_level,
            # #                          options={'disp': True})
        else:
            res1 = optimize.shgo(self.objective_function, bounds=[], args=search_grid)
        ures = res1.x  # Minimization parameters
        print("Minimization Parameters: ", ures)
        print("distance from u_0: \n", ures - self.initial_value_u0)
        # -------------------------------------------------------------------------------------------
        self.u_v_w_minimizer = ures  # Minimization parameters is now saved as an attribute of Particle class

        # -------------------Normalization Constant Calculation---------------------------------------
        # The quad function returns a tuple with the first element being the result of the integration and the second
        # element being an estimate of the absolute error.

        self.normalization_constant = quad(self.integrand, 0, self.maxv)[0]  # Normalization constant divides F error
        print(f"Numerical Normalization Constant: {self.normalization_constant}")

        ########   --- I NEED TO CHECK WHAT IS THE IMPLEMENTATION OF numerical_energy_integral  (paper???)
        # Professor Kosmas stressed a negative sign in the original implementation to be checked on 30/07/2024 - DONE

        final_minimization_error = self.objective_function(ures, search_grid_list)

        # CASE 1: un-normalized error function
        # # Calculate final error function value for the optimizer - This requires original implementation of objective
        # final_error = self.objective_function(ures, search_grid_list)
        print(f"Unnormalized Error Function Value: {final_minimization_error}")

        # CASE 2: normalized error function by division with the number of r_i points (length of search grid)
        # This also requires original implementation of objective_function
        print(
            f"Normalized Error Function Value (by dividing the accumulated error by the size of the search grid): {final_minimization_error / self.relative_search_points}")

        # CASE 3: normalized error function by dividing with the normalization_constant (integral calculation)
        print(
            f"Normalized Error Function Value (by dividing the accumulated error by the normalization constant): {final_minimization_error / self.normalization_constant}")

        # CASE 4: SCIENTIFIC NOVELTY - Normalized error function by with built-in integrand-based normalization
        # (advanced case needs further research) Calculate the final normalized error function value
        # final_error1 = self.objective_function_advanced(ures, search_grid_list)
        # print(f"Final Error Function Value (with integrand-based normalization): {final_error1}")

        if self.system == "combined_hamiltonian":
            # Comment out the line that follows for checking default case of numerical/integral energy calculation.
            numer_energy = quad(self.numerical_energy_integral_combined, 0, self.maxv)[0]  # equation (4)

            # ------------- Following integral energy calculation is on beta-testing phase -----------------------------
            # In what follows we test the integral energy obtained when the Hamiltonian is defined by equation (10) from
            # Klein-Gordon paper of Tsoulos, Kosmas et al. The above standard case is defined by equation (3.1).
            # numer_energy = quad(self.numerical_energy_integral_eq10, 0, self.maxv)[0]
        elif self.system == "relativistic":
            numer_energy = quad(self.numerical_energy_integral_relativistic, 0, self.maxv)[0]
        elif self.system == "non-relativistic":
            numer_energy = quad(self.numerical_energy_integral_non_relativistic, 0, self.maxv)[0]
        else:
            numer_energy = quad(self.numerical_energy_integral_one_body, 0, self.maxv)[0]

        # normalized_numer_energy = numer_energy * self.normalization_constant
        normalized_numer_energy = numer_energy / self.normalization_constant
        print("Binding (E_nl - mc^2) energy: ", self.energy)
        # print("Normalized Analytical (integral) function energy: ", norm_anal_energy)
        print(f"Numerical (wave) function Energy: {numer_energy}")
        print(f"Normalized numerical (wave) function Energy: {normalized_numer_energy}")
        print(f"Distance of numerical from binding energy: {np.abs(numer_energy) - np.abs(self.energy)}")
        print(
            f"Distance of normalized numerical from binding energy (system error): {np.abs(normalized_numer_energy) - np.abs(self.energy)}")
        # print(
        #     f"Distance of normalized numerical from normalized (integral) analytical energy: {np.abs(normalized_numer_energy) - np.abs(norm_anal_energy)}")
        # print(f"Breit combinations: {self.breit}")

        # Instantiate Klein-Gordon Plotter
        # Neural Network Solution Plotting
        KG_plotter = KGPlotter(
            partl=self.particle,
            nq=self.primary_quant_num,
            lq=self.azimuthal_quant_num,
            alpha0=self.alpha0,
            verbose=self.verbose,
            index=self.experiment_index,
            method='NeuralNetworks'
        )

        KG_plotter.plot(
            func_analytical=self.U_nl,
            func_numerical=self.normalized_arfunc,
            anum='quantum_numbers: ', lb=0, rb=0, ipoints=400, overlay=self.overlay_plot, idx=self.experiment_index
        )
        #
        # KG_plotter = KGPlotter(partl=self.particle, nq=self.primary_quant_num,
        #                        lq=self.azimuthal_quant_num, alpha0=self.alpha0, verbose=self.verbose,
        #                        index=self.experiment_index, method=self.method)
        #
        # """ Plotting and Exporting function values to .csv and .txt formats """
        # # Analytical and Numerical Solution Overlayed Plot
        # KG_plotter.plot(
        #     func_analytical=self.U_nl,
        #     func_numerical=self.normalized_arfunc,
        #     anum='quantum_numbers: ', lb=0, rb=0, ipoints=400, overlay=self.overlay_plot, idx=self.experiment_index
        # )
        """Exporting Initial Point"""
        # with open(f'./data/{self.particle}-Sch/init_point{self.primary_quant_num}{self.azimuthal_quant_num}.txt', 'w',
        #          newline='') as file:
        #    file.write(f"Initial point: {u0}")
        # ----------------------------------------   SUPRESS    -------------------------------------------------
        # supress I/O operations that are not needed for ablations
        self.save_initial_point_u0(self.initial_value_u0)

        """Exporting Minimization Parameters"""
        with open(f'./data/{self.particle}-KG/param_NN_{self.primary_quant_num}{self.azimuthal_quant_num}.csv', 'w',
                  newline='') as file:
            writer = csv.writer(file)
            for p in self.u_v_w_minimizer:
                writer.writerow([p])

        """Exporting Energy Value"""
        with open(f'./data/{self.particle}-KG/energ_NN_{self.primary_quant_num}{self.azimuthal_quant_num}.txt', 'w',
                  newline='') as file:
            file.write(f"Anal_Energy: {self.energy}\n")
            file.write(f"Numer_Energy: {numer_energy}")

    def solve_KG_FEM(self, hyperparameters):
        """Solve the Klein-Gordon equation using Finite Element Method (FEM)"""

        # Print k_optimization_constant value as calculated above
        print("k_optimization_constant: ", self.k_optimization_const)
        # Access hyperparameters here
        print("Hyperparameters:", hyperparameters)

        self.num_hidden_units = hyperparameters['num_of_hidden_units']
        self.relative_search_points = hyperparameters['num_of_equidistant_points']

        # Calculate Energy eigenvalue
        # Comment out the following line to run standard model. Otherwise, follow up A. Giotis 16/09/2024
        self.energy = self.E_nl()

        def exponential_search_grid(start=1e-6, end=30, num_points=100):
            return np.exp(np.linspace(np.log(start), np.log(end), num_points))

        def chebyshev_search_grid(start=1e-6, end=30, num_points=100):
            return 0.5 * (start + end) + 0.5 * (end - start) * np.cos(
                np.linspace(np.pi, 0, self.relative_search_points))

        # Define the search grid (for leptonic systems, typical grid may range from 1e-6 to larger values)
        # search_grid = self.RelativeDistanceSearchGrid()

        # An exponential grid places more points near the origin where the wave function is rapidly changing.
        search_grid = chebyshev_search_grid()

        # Convert NumPy arrays to lists in search_grid (this is necessary for FEMSolver)
        search_grid_list = [grid.tolist() if isinstance(grid, np.ndarray) else grid for grid in search_grid]

        # Coulomb potential for two-body system
        def V_coulomb_fem(r):
            scaling_factor = 1e-2  # Lower scaling factor
            return self.coulomb_potential(r) * scaling_factor

        def V_harmonic_oscillator(r):
            k = 1.0  # Spring constant or force constant
            return 0.5 * k * r ** 2

        # Increase number of elements for finer grid resolution
        fem_solver = FEMSolver(
            search_grid=search_grid_list,
            num_elements=self.relative_search_points,
            potential_func=V_coulomb_fem,
            mass=self.mass,
            angular_momentum=self.azimuthal_quant_num,
            energy=self.E_nl(),  # Current energy value
            E_nl_func=self.E_nl  # Passing the function to calculate energy eigenvalue
        )

        # Solve using FEM
        r_values, R_values = fem_solver.solve_fem()

        # ---- Normalization ---- #
        epsilon = 1e-10  # A small value to prevent division by zero
        normalization_constant = np.sqrt(np.trapz(R_values ** 2, r_values) + epsilon)

        if normalization_constant == 0 or np.isclose(normalization_constant, 0):
            raise ValueError("Normalization constant is too small or zero, leading to invalid normalization.")

        normalized_R_values = R_values / normalization_constant

        # ---- Calculate Numerical Energy ---- #
        def numerical_energy_integral(r):
            R_r = np.interp(r, r_values, normalized_R_values)
            dR_dr = np.gradient(normalized_R_values, r_values)
            dR_r = np.interp(r, r_values, dR_dr)
            V_r = V_coulomb_fem(r)
            return (dR_r ** 2 / (2 * self.mass)) + V_r * (R_r ** 2)

        # Integrate the numerical energy over the domain
        numer_energy = quad(numerical_energy_integral, 0, np.max(r_values))[0]

        print(f"Binding (E_nl - mc^2) energy: {self.energy}")
        print(f"Numerical (wave) function Energy using FEM: {numer_energy}")
        print(f"Normalized numerical (wave) function Energy: {numer_energy / normalization_constant}")
        print(f"Distance of numerical from binding energy: {np.abs(numer_energy) - np.abs(self.energy)}")
        print(f"Distance of normalized numerical from binding energy (system error): "
              f"{np.abs(numer_energy / normalization_constant) - np.abs(self.energy)}")

        # ---- Exporting Energy Value ---- #
        with open(f'./data/{self.particle}-KG/energ_FEM_{self.primary_quant_num}_{self.azimuthal_quant_num}.txt', 'w',
                  newline='') as file:
            file.write(f"Anal_Energy: {self.energy}\n")
            file.write(f"Numer_Energy: {numer_energy}\n")  # Write the numerical energy

        # FEM Solution Plotting
        KG_plotter = KGPlotter(
            partl=self.particle,
            nq=self.primary_quant_num,
            lq=self.azimuthal_quant_num,
            alpha0=self.alpha0,
            verbose=self.verbose,
            index=self.experiment_index,
            method='FEM'
        )

        KG_plotter.plot(
            func_analytical=self.U_nl,  # Analytical solution function
            func_numerical=lambda r: np.interp(r, r_values, normalized_R_values),  # Interpolated FEM solution
            anum='quantum_numbers: ', lb=0, rb=0, ipoints=self.relative_search_points,
            overlay=self.overlay_plot, idx=self.experiment_index
        )

        # Exporting Initial Point
        self.save_initial_point_u0(self.initial_value_u0)


class FEMSolver:
    def __init__(self, search_grid, num_elements, potential_func, mass, energy, angular_momentum, E_nl_func):
        self.r_min = np.min(search_grid)
        self.r_max = np.max(search_grid)
        self.num_elements = num_elements
        self.potential_func = potential_func  # Potential function V(r)
        self.mass = mass  # Reduced mass for two-body system
        self.energy = energy  # Energy eigenvalue
        self.angular_momentum = angular_momentum  # Angular momentum quantum number (l)
        self.E_nl_func = E_nl_func  # Function to calculate energy eigenvalue E_nl
        self.fundamental_values = fundamental_values  # Object for constants
        self.mesh = search_grid

        # Calculate and store the singular point r_s once during initialization
        self.r_s = (self.fundamental_values.Z * self.fundamental_values.e_charge ** 2) / (
                4 * np.pi * self.fundamental_values.eps_0 * self.energy)

    def create_mesh(self):
        """Create a uniform mesh over the radial domain."""
        return np.linspace(self.r_min, self.r_max, self.num_elements + 1)

    def modified_basis_functions(self, r_left, r_right):
        """Define modified basis functions that account for singularities at r=0 and r=r_s."""
        epsilon = 1e-6  # Small cutoff value to avoid singularity at r = 0

        r_eff_left = max(r_left, epsilon)  # Avoid r = 0 exactly for left boundary

        # Handle the basis functions based on whether the singularity lies in this element
        if r_eff_left < self.r_s:
            phi_1 = (r_right - r_eff_left) / (r_right - r_left)
            phi_2 = (r_eff_left - r_left) / (r_right - r_left)
        else:
            phi_1 = (r_right - self.r_s) / (r_right - r_left)
            phi_2 = (self.r_s - r_left) / (r_right - r_left)

        return phi_1, phi_2

    def assemble_stiffness_matrix(self):
        """Assemble the stiffness matrix using FEM."""
        num_nodes = len(self.mesh)  # Use the length of self.mesh directly
        K = np.zeros((num_nodes, num_nodes))  # Stiffness matrix
        F = np.zeros(num_nodes)  # Force vector

        for elem in range(self.num_elements):
            # Check that we're not exceeding the length of the mesh
            if elem >= num_nodes - 1:
                print(f"Warning: Element {elem} exceeds mesh boundary.")
                break  # Break the loop if we go beyond mesh

            r_left = self.mesh[elem]
            r_right = self.mesh[elem + 1]
            h = r_right - r_left

            # Call `element_stiffness` without passing `r_s` as an argument
            k_local, f_local = self.element_stiffness(r_left, r_right)

            # Print element bounds and contributions for debugging
            print(f"\nAssembling element {elem}: r_left = {r_left}, r_right = {r_right}")
            print(f"  Local stiffness matrix k_local:\n{k_local}")
            print(f"  Local force vector f_local: {f_local}")

            # Assembly into the global stiffness matrix
            K[elem:elem + 2, elem:elem + 2] += k_local
            F[elem:elem + 2] += f_local

        print("\nGlobal Stiffness Matrix K:\n", K)
        print("\nGlobal Force Vector F:\n", F)

        return K, F

    def element_stiffness(self, r_left, r_right):
        """Compute the local stiffness matrix for an element, using the stored singular point `r_s`."""
        h = r_right - r_left
        # Use `self.r_s` directly here
        phi_1, phi_2 = self.modified_basis_functions(r_left, r_right)

        # Local stiffness matrix
        k_local = np.zeros((2, 2))
        f_local = np.zeros(2)  # No explicit source term

        for i in range(2):
            for j in range(2):
                r_mid = (r_left + r_right) / 2  # Midpoint of the element
                V_r = self.potential_func(r_mid)  # Potential at midpoint
                E_nl_value = self.E_nl_func()  # Calculate energy eigenvalue E_nl()

                # Print the intermediate values
                # print(f"Element [{i},{j}] - Midpoint: {r_mid}")
                # print(f"  Coulomb potential V_r at midpoint: {V_r}")
                # print(f"  Energy eigenvalue E_nl: {E_nl_value}")
                # print(f"  Step size h: {h}")
                # print(f"  Basis function values (phi_1, phi_2): {phi_1}, {phi_2}")

                # Derivative terms (central difference approximation)
                d_phi_i = (-1) ** i / h
                d_phi_j = (-1) ** j / h

                # Modify for rescaled function u(r) = r * R(r)
                k_local[i, j] = (
                        (d_phi_i * d_phi_j)  # Second derivative part
                        + (2 / r_mid + 1 / (E_nl_value - V_r) * (V_r / r_mid)) * d_phi_j  # First derivative part
                        + (self.angular_momentum * (
                        self.angular_momentum + 1) / r_mid ** 2) * phi_1  # Angular momentum part
                        - ((np.power(E_nl_value - V_r, 2) - np.power(self.mass, 2)) / (
                        4 * np.power(self.fundamental_values.hbarc, 2))) * phi_1  # Relativistic correction term
                )

                # Print the stiffness matrix contributions
                # print(f"  k_local[{i},{j}] = {k_local[i, j]}\n")

        return k_local, f_local

    def solve_fem(self):
        """Solve the FEM system for the unknown radial function."""
        K, F = self.assemble_stiffness_matrix()

        # Print out matrix and vector for debugging
        print("Stiffness Matrix K:", K)
        print("Force Vector F:", F)

        # Apply boundary conditions (R(0) = 0 and R(r_max) = 0)
        # Adjust both the first and last rows for boundary conditions
        K[0, :] = 0
        K[:, 0] = 0
        K[0, 0] = 1
        F[0] = 0

        K[-1, :] = 0
        K[:, -1] = 0
        K[-1, -1] = 1
        F[-1] = 0

        # Solve the linear system
        R = solve(K, F)

        # Print out solution vector R
        print("Solution Vector R:", R)

        # Return the mesh (r values) and radial function (R values)
        return self.mesh, R
