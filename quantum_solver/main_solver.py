class MainSolver:
    def __init__(self, hyperparameters, parent_equation=None):
        self.parent_equation = parent_equation
        self.hyperparameters = hyperparameters

    def solve_system(self, quantum_system):
        # Use hyperparameters and call appropriate methods to solve the selected equation
        if self.hyperparameters['diff_equation'] == 'Schrodinger':
            quantum_system.solve_schrodinger_equation(self.hyperparameters)
        elif self.hyperparameters['diff_equation'] == 'Dirac':
            quantum_system.solve_dirac_equation(self.hyperparameters, lower_limit_1=1e-5, upper_limit_1=1e5, lower_limit_2=1e-7, upper_limit_2=1e7)
        elif self.hyperparameters['diff_equation'] == 'KleinGordon' or (
                self.hyperparameters['mass_relation'] == "m1=m2"
                or self.hyperparameters['mass_relation'] == "dynamic"):
            if self.parent_equation == "Schrodinger":
                quantum_system.solve_schrodinger_equation(self.hyperparameters)
            elif self.parent_equation == "Dirac":
                quantum_system.solve_dirac_equation(self.hyperparameters, lower_limit_1=1e-5, upper_limit_1=1e5, lower_limit_2=1e-7, upper_limit_2=1e7)
            else:
                quantum_system.solve_KG_equation(self.hyperparameters)
                # now we test for reference our own implementation of the Finite Elements Method of Scot et al. 1992
                self.hyperparameters['numerical_method'] = 'FEM'
                quantum_system.solve_KG_FEM(self.hyperparameters)
