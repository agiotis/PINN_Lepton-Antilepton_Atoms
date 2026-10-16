# main.py

# Set up the solver with hyperparameters: (1) is the number of hidden units/neurons of the first hidden layer of
# 2-layer NN, (2) reflects the number of equidistant points that comprise the grid from r=0 to r=b (for some b where
# the wave function is practically vanishing), (3) fuc is a regularizing constant for each distinct term of the error
# function, (4) optimization method is either 'BFGS' or 'trust-constr' for finding minimizers u, v, w within
# NeuralNetwork computational approximation of u(r) for given equation
# import argparse
# from quantum_solver.quantum_system import Particle, Electron
# from quantum_solver.main_solver import MainSolver
# from quantum_solver.quantum_utils import Schrodinger, Dirac
import argparse
from itertools import product
import numpy as np
import re
from quantum_solver.quantum_system import Particle, Electron, Positronium
from quantum_solver.main_solver import MainSolver
from quantum_solver.quantum_utils import Schrodinger, Dirac, TwoBodyDiracCoulomb, KleinGordon, Dynamic

from quantum_solver.fundamental_values import FundamentalValues
from quantum_solver import dirac_canonical

FV = FundamentalValues()


class StoreExplicitTolerance(argparse.Action):
    """Record an explicit tolerance without changing the existing CLI default."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace._minimization_tol_explicit = True


def generate_breit_terms_combinations():
    """Return the 32 five-entry string tuples in the existing CLI order."""
    return list(product(('0', '1'), repeat=5))


def normalize_dirac_coefficient_mode(value):
    return dirac_canonical.normalize_coefficient_mode(value)


def validate_dirac_coefficient_mode(arguments):
    mode = normalize_dirac_coefficient_mode(getattr(arguments, "dirac_coefficient_mode", None))
    is_dirac = (arguments.diff_equation == "Dirac" or
                (arguments.diff_equation == "Dynamic" and getattr(arguments, "parent_equation", None) == "Dirac"))
    if is_dirac and mode is None:
        raise ValueError("--dirac-coefficient-mode {learnable,fixed} is required for Dirac runs")
    return mode if is_dirac else None


# Generate the range with a step of 0.02
def generate_choices(start, end, step):
    return [round(x, 2) for x in np.arange(start, end + step, step)]


# Define range for k_optimization_constant
k_choices = generate_choices(0.1, 0.3, 0.02)


def main(arguments):
    # Print the initial values of arguments
    global dynamic_system
    print(f"Initial mass_relation: {arguments.mass_relation}")
    print(f"Default diff_equation: {arguments.diff_equation}")
    # Default to Klein-Gordon if mass_relation is m1=m2 and no diff_equation is specified
    # if arguments.mass_relation == "m1=m2" and arguments.diff_equation == "Schrodinger":
    #     arguments.diff_equation = "KleinGordon"
    if arguments.mass_relation == "dynamic" and arguments.diff_equation == "Schrodinger":
        arguments.diff_equation = "Dynamic"
        arguments.parent_equation = "Schrodinger"
    elif arguments.mass_relation == "dynamic" and arguments.diff_equation == "Dirac":
        arguments.diff_equation = "Dynamic"
        arguments.parent_equation = "Dirac"

    # Print the values after modification
    print(f"Actual (modified) diff_equation reflecting mass relation: {arguments.diff_equation}")
    dirac_coefficient_mode = validate_dirac_coefficient_mode(arguments)
    manual_u0 = None

    if arguments.manual_u0 is not None:
        # Αφαιρούμε αγκύλες αν υπάρχουν
        clean = args.manual_u0.strip()
        clean = re.sub(r"[\[\]]", "", clean)

        # Αντικαθιστούμε κόμματα με κενά (για να παίζουν και οι 2 μορφές)
        clean = clean.replace(",", " ")

        # Parse με numpy
        manual_u0 = np.fromstring(clean, sep=" ")

        if manual_u0.size == 0:
            raise ValueError("Failed to parse --manual_u0. Check format.")
    hyperparameters = {
        'diff_equation': arguments.diff_equation,
        'arguments.parent_equation': arguments.parent_equation,
        'hamiltonian_correction_terms': arguments.hamiltonian_correction_terms,
        'num_of_hidden_units': arguments.num_of_hidden_units,
        'num_of_equidistant_points': arguments.num_of_equidistant_points,
        'fuc': arguments.fuc,
        'optimization_method': arguments.optimization_method,
        'minimization_tol_lvl': arguments.minimization_tol_lvl,
        'minimization_tol_explicit': getattr(arguments, '_minimization_tol_explicit', True),
        'primary_quantum_number': None,
        'azimuthal_quantum_number': None,
        'total_angular_momentum': None,
        'k_operator': None,
        'coupling_constant': arguments.coupling_constant,
        'two_body_system': arguments.two_body_system,
        'k_optimization_constant': arguments.k_optimization_constant,
        # 'integral_normalization_constant': arguments.integral_normalization_constant,
        'u0_init_method': arguments.u0_init_method,
        'activation_function': arguments.activation_function,
        'user_search_grid': arguments.search_grid,
        'overlay_plot': arguments.overlay_plot,
        'input_file': arguments.input_file,
        'verbose': arguments.verbose,
        'experiment_index': arguments.experiment_index,
        'mass_relation': arguments.mass_relation,
        'subsystem': arguments.subsystem,
        'numerical_method': arguments.numerical_method,
        'manual_u0': manual_u0,
        'dirac_coefficient_mode': dirac_coefficient_mode
    }
    # Instantiate quantum_utils.Schrodinger object to insert primary/azimuthal quantum values for Schrödinger equation
    if arguments.diff_equation == 'Schrodinger':
        schrodinger_instance = Schrodinger(hyperparameters)
        print("Values for n, l are: ", schrodinger_instance.n, schrodinger_instance.l)
        print("coupling constant (fine structure constant): ", hyperparameters['coupling_constant'])
        hyperparameters['primary_quantum_number'] = schrodinger_instance.n
        hyperparameters['azimuthal_quantum_number'] = schrodinger_instance.l
        print("System Index: ", schrodinger_instance.system_index)
        # create particle instance to inherent properties and methods required for Schrödinger equation solution
        particle = Particle(hyperparameters, schrodinger_instance.system_index)

        solver = MainSolver(hyperparameters)
        # Solve the system
        solver.solve_system(particle)

    elif arguments.diff_equation == 'Dirac':
        # Instantiate quantum_utils.Dirac to insert primary/azimuthal quantum values along with total spin and angular
        # momentum
        dirac_instance = Dirac(hyperparameters)
        # print(dirac_instance.n, dirac_instance.l, dirac_instance.j, dirac_instance.k)
        hyperparameters['primary_quantum_number'] = dirac_instance.n
        hyperparameters['azimuthal_quantum_number'] = dirac_instance.l
        hyperparameters['total_angular_momentum'] = dirac_instance.j
        hyperparameters['k_operator'] = dirac_instance.k
        print("Values for n, l, j, k, are: ", dirac_instance.n, dirac_instance.l, dirac_instance.j, dirac_instance.k)
        print("coupling constant (fine structure constant): ", hyperparameters['coupling_constant'])
        # create electron instance to inherent properties and methods required for Dirac equation solution
        electron = Electron(hyperparameters, dirac_instance.system_index)
        print("System Index: ", dirac_instance.system_index)

        solver = MainSolver(hyperparameters)
        # Solve the system
        solver.solve_system(electron)

    elif arguments.diff_equation == 'TwoBodyDiracCoulomb':
        tb_dirac_coulomb_instance = TwoBodyDiracCoulomb(hyperparameters)
        hyperparameters['primary_quantum_number'] = tb_dirac_coulomb_instance.n
        hyperparameters['azimuthal_quantum_number'] = tb_dirac_coulomb_instance.l
        electron = Electron(hyperparameters, tb_dirac_coulomb_instance.system_index)
        solver = MainSolver(hyperparameters)
        solver.solve_system(electron)

    elif arguments.diff_equation == 'KleinGordon':
        klein_gordon_instance = KleinGordon(hyperparameters)
        hyperparameters['primary_quantum_number'] = klein_gordon_instance.n
        hyperparameters['azimuthal_quantum_number'] = klein_gordon_instance.l
        print("coupling constant (fine structure constant): ", hyperparameters['coupling_constant'])
        positronium = Positronium(hyperparameters, klein_gordon_instance.system_index)
        solver = MainSolver(hyperparameters)
        solver.solve_system(positronium)
    elif arguments.diff_equation == 'Dynamic':
        dynamic_instance = Dynamic(hyperparameters)
        hyperparameters['primary_quantum_number'] = dynamic_instance.n
        hyperparameters['azimuthal_quantum_number'] = dynamic_instance.l
        print("coupling constant (fine structure constant): ", hyperparameters['coupling_constant'])
        if dynamic_instance.system_index >= 5:
            dynamic_system = Positronium(hyperparameters, dynamic_instance.system_index)
            arguments.parent_equation = "KleinGordon"
        elif dynamic_instance.system_index <= 4 and arguments.parent_equation == "Schrodinger":
            dynamic_system = Particle(hyperparameters, dynamic_instance.system_index)
        elif dynamic_instance.system_index <= 4 and arguments.diff_equation == "Dirac":
            dynamic_system = Electron(hyperparameters, dynamic_instance.system_index)

        solver = MainSolver(hyperparameters, arguments.parent_equation)
        solver.solve_system(dynamic_system)
    else:
        parser.error(f"Invalid differential equation: {arguments.diff_equation}. "
                     f"Type main.py --help for valid options.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve Schrodinger, Dirac, Two-body Dirac-Coulomb or equivalent Klein-Gordon equation.")
    parser.add_argument("--diff_equation",
                        choices=["Schrodinger", "Dirac", "TwoBodyDiracCoulomb", "KleinGordon", "Dynamic"],
                        default="Schrodinger")
    # parser.add_argument("--diff_equation", choices=["Schrodinger", "Dirac"], default="Schrodinger")

    # Update the choices parameter with the formatted combinations
    # Use the function to get formatted breit combinations
    breit_combinations = generate_breit_terms_combinations()
    # Update the choices parameter with the formatted combinations as strings
    # Convert tuples to strings
    breit_combinations_str = [' '.join(map(str, combo)) for combo in breit_combinations]
    parser.add_argument("--parent_equation",
                        choices=["Schrodinger", "Dirac", "TwoBodyDiracCoulomb", "KleinGordon", "Dynamic"],
                        default="Schrodinger")
    parser.add_argument("--hamiltonian_correction_terms", choices=breit_combinations_str,
                        default="0 0 0 0 0",
                        help="Choices for hamiltonian_correction_terms (i.e., Breit-Darwin) terms that enter the "
                             "Hamiltonian H_0: 'custom' allows you to provide"
                             "your own values from all possible combinations (31 in total). Default "
                             "choice is no Breit terms at all, only H_0 in Hamiltonian operator")
    parser.add_argument("--num_of_hidden_units", type=int, default=10)
    parser.add_argument("--num_of_equidistant_points", type=int, default=150)
    parser.add_argument("--fuc", type=float, default=1)
    parser.add_argument("--coupling_constant", type=float, default=FV.fsc, help="User can provide coupling constant "
                                                                                "during input. If not provided then "
                                                                                "self.fsc is set beforehand.")
    parser.add_argument("--two_body_system", choices=["non-relativistic", "relativistic",
                                                      "combined_hamiltonian", "one-body"],
                        default="combined_hamiltonian",
                        help="If used then we solve for the actual two-body Klein-Gordon "
                             "equation with M_0 = 2 * mass in the relativistic correction "
                             "term, not its approximation for the non-relativistic one-body"
                             " system. Also we account for a combined approach of equation "
                             "3.1 from Scot et al. 1992 paper")
    parser.add_argument("--k_optimization_constant", type=float, default=0.16, choices=k_choices, help="Optimization "
                                                                                                       "constant kappa")
    # parser.add_argument("--integral_normalization_constant", action='store_true', default=False, help="normalization "
    #                                                                                                   "constant N "
    #                                                                                                   "that divides "
    #                                                                                                   "the error "
    #                                                                                                   "function to "
    #                                                                                                   "avoid trivial "
    #                                                                                                   "solutions U("
    #                                                                                                   "r)=0, "
    #                                                                                                   "during "
    #                                                                                                   "optimization "
    #                                                                                                   "of the K-G "
    #                                                                                                   "equation")
    parser.add_argument("--optimization_method", choices=["BFGS", "trust-constr", "L-BFGS-B"], default="trust-constr")
    parser.set_defaults(_minimization_tol_explicit=False)
    parser.add_argument("--minimization_tol_lvl", type=float, default=1e-8, action=StoreExplicitTolerance,
                        help="Tolerance level for the optimization method. "
                             "The specific interpretation of this tolerance may vary "
                             "depending on the optimization method and problem characteristics. "
                             "For example, when using trust-constr with a sigmoid activation function, "
                             "the termination condition may satisfy g_tol, since sigmoid can suffer from vanishing "
                             "gradient during optimization."
                             "On the other hand, with ReLU activation, the termination condition may satisfy x_tol.")
    parser.add_argument("--u0_init_method", choices=["random", "xavier", "fixed", "bounded", "manual", "physics_conditioned"], default="random")
    parser.add_argument(
        "--manual_u0",
        type=str,
        default=None,
        help="Manual initial u0 vector, e.g. '[0.1 0.2 0.3]' or '0.1,0.2,0.3'"
    )
    parser.add_argument("--activation_function", choices=["sigmoid", "relu", "tanh", "smplus"], default="relu")
    parser.add_argument("--search_grid", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--overlay_plot", action='store_true', default=False, help="Enable overlayed plot")
    parser.add_argument("--input_file", action='store_true', default=False, help="Read values for quantum numbers and "
                                                                                 "quantum system index from file")
    parser.add_argument("--verbose", action='store_true', default=False, help="Print plots and optimization progress "
                                                                              "bar if used")
    parser.add_argument("--experiment_index", type=int, default=2, help="Ablation experiment index to be passed to "
                                                                        "Plotting module")
    parser.add_argument("--mass_relation", choices=["m1>>m2", "m1=m2", "dynamic"], required=True, help="Axis 1: "
                                                                                                       "m1>>m2 solves "
                                                                                                       "for substems "
                                                                                                       "1 to 4, "
                                                                                                       "meaning "
                                                                                                       "Muonium, "
                                                                                                       "e-Tauonium, "
                                                                                                       "mu-Tauonium. "
                                                                                                       "NewSystemDiffMass "
                                                                                                       "Each system "
                                                                                                       "is solved as "
                                                                                                       "a one-body "
                                                                                                       "system using "
                                                                                                       "(1) standard "
                                                                                                       "Schrodinger "
                                                                                                       "equation, "
                                                                                                       "(2) extended "
                                                                                                       "Schrodinger "
                                                                                                       "with "
                                                                                                       "hamiltonian "
                                                                                                       "correction "
                                                                                                       "terms and (3) "
                                                                                                       "Dirac "
                                                                                                       "equation. "
                                                                                                       "Axis 2: m1=m2 "
                                                                                                       "solves for "
                                                                                                       "subsystems 5 "
                                                                                                       "to 8, i.e., "
                                                                                                       "Positronium, "
                                                                                                       "True Muonium, "
                                                                                                       "True "
                                                                                                       "Tauonium and "
                                                                                                       "NewSystemEqMass."
                                                                                                       "Each system "
                                                                                                       "is solved by "
                                                                                                       "a coupled "
                                                                                                       "ODE, namely, "
                                                                                                       "Klein-Gordon "
                                                                                                       "equation.")
    parser.add_argument("--subsystem",
                        choices=["Muonium", "e-Tauonium", "μ-Tauonium", "NewSystemDiffMass", "Positronium",
                                 "True_Muonium", "True_Tauonium",
                                 "NewSystemEqMass"], metavar="SUBSYSTEM", default="Other",
                        help="this is optional to select the desired quantum "
                             "system according to the selected mass_relation "
                             "parameter")
    parser.add_argument("--dirac-coefficient-mode", dest="dirac_coefficient_mode", default=None,
                        help="Explicit Dirac coefficient mode: learnable or fixed. Required for Dirac runs.")
    parser.add_argument("--numerical_method", choices=["NeuralNetworks", "FEM"], default="NeuralNetworks",
                        help="This option refers only to the two-body Dirac equation that is solved using 2nd order "
                             "KleinGordon equation. Our proposed default approach relies on Neural Networks for "
                             "the numerical approximation, while the FEM parameter corresponds to the Finite "
                             "Elements Method of Scot et al. 1992")

    args = parser.parse_args()
    if args.hamiltonian_correction_terms == "custom":
        # Get the custom Breit-Darwin terms from the user
        args.hamiltonian_correction_terms = input("Enter custom hamiltonian correction terms: ")

    main(args)
