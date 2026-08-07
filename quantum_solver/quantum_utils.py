import os
import csv


class QuantumNumber:
    def __init__(self, n=None, l=None, j=None, k=None, fsc=None, mass1=None, mass2=None, sindex=None):
        self.n = n
        self.l = l
        self.j = j
        self.k = k
        self.fsc = fsc
        self.mass1 = mass1
        self.mass2 = mass2
        self.system_index = sindex
        self.input_file = None


class Schrodinger(QuantumNumber):
    def __init__(self, hyperparameters):
        super().__init__(n=None, l=None, j=None, k=None, fsc=None, sindex=None)
        self.input_file = hyperparameters['input_file']
        self.mass_relation = hyperparameters['mass_relation']
        self.subsystem = hyperparameters['subsystem']
        self._user_input()

    def _user_input(self):
        print("Non-Relativistic Description")
        if not os.path.isfile("./uinput/inpsch.csv") or self.input_file is False:
            if self.mass_relation == "m1>>m2":
                if self.subsystem == "Muonium":
                    self.system_index = 1
                elif self.subsystem == "e-Tauonium":
                    self.system_index = 2
                elif self.subsystem == "μ-Tauonium":
                    self.system_index = 3
                elif self.subsystem == "NewSystemDiffMass":
                    self.system_index = 4
                else:
                    raise ValueError("Invalid subsystem for m1>>m2")
            elif self.mass_relation == "m1=m2":
                if self.subsystem == "Positronium":
                    self.system_index = 5
                elif self.subsystem == "True Muonium":
                    self.system_index = 6
                elif self.subsystem == "True Tauonium":
                    self.system_index = 7  # Assuming you add a new index for True Tauonium
                elif self.subsystem == "NewSystemEqMass":
                    self.system_index = 8
                else:
                    raise ValueError("Invalid subsystem for m1=m2")
            elif self.mass_relation == "dynamic":
                self.system_index = int(
                    input("Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, "
                          "5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit) :"))
                if self.system_index == 0:
                    raise ValueError("Please select a valid System: 1, 2, 3, 4, 5, 6, 7, or 8")
            else:
                raise ValueError("Invalid mass relation")

            # Loop until valid n and l are provided
            while True:
                ninp, linp = input("Enter the values of n and l: ").split()
                self.n = int(ninp)
                self.l = int(linp)

                if self.n > self.l:
                    break
                else:
                    print(
                        'Error: principal quantum number "n" has to be greater than orbital angular momentum "l". '
                        'Please enter the values again.')

            # if self.fsc is None:
            #     self.fsc = float(input("enter coupling constant: "))
        else:
            try:
                with open("./uinput/inpsch.csv") as file:
                    reader = csv.reader(file)
                    rows = [row for row in reader]
                # [sindex, ninp, linp, fsc_inp] = rows
                [sindex, ninp, linp] = rows
                self.system_index = int(sindex[0])
                self.n = int(ninp[0])
                self.l = int(linp[0])
                # self.fsc = float(fsc_inp[0])
            except FileNotFoundError:
                raise ValueError("No inpsch.csv file exists!")


class Dirac(QuantumNumber):
    def __init__(self, hyperparameters):
        super().__init__(n=None, l=None, j=None, k=None, fsc=None, sindex=None)
        self.input_file = hyperparameters['input_file']
        self.mass_relation = hyperparameters['mass_relation']
        self.subsystem = hyperparameters['subsystem']
        self._user_input()

    def _user_input(self):
        print("Relativistic Description")
        input_file_path = os.path.join(os.getcwd(), 'uinput', 'inpdc.csv')

        # Check for file existence and input_file flag
        if not os.path.isfile(input_file_path) or self.input_file is False:
            print(f"File not found or input_file set to False: {input_file_path}")

            # Handle mass relation cases
            if self.mass_relation == "m1>>m2":
                if self.subsystem == "Muonium":
                    self.system_index = 1
                elif self.subsystem == "e-Tauonium":
                    self.system_index = 2
                elif self.subsystem == "μ-Tauonium":
                    self.system_index = 3
                elif self.subsystem == "NewSystemDiffMass":
                    self.system_index = 4
                else:
                    raise ValueError("Invalid subsystem for m1>>m2")
            elif self.mass_relation == "m1=m2":
                if self.subsystem == "Positronium":
                    self.system_index = 5
                elif self.subsystem == "True Muonium":
                    self.system_index = 6
                elif self.subsystem == "True Tauonium":
                    self.system_index = 7
                elif self.subsystem == "NewSystemEqMass":
                    self.system_index = 8
                else:
                    raise ValueError("Invalid subsystem for m1=m2")
            elif self.mass_relation == "dynamic":
                self.system_index = int(
                    input("Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, "
                          "5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit) :"))
                if self.system_index == 0:
                    raise ValueError("Please select a valid System: 1, 2, 3, 4, 5, 6, 7, or 8")
            else:
                raise ValueError("Invalid mass relation")

            # Loop until valid n and l are provided
            while True:
                ninp, linp = input("Enter the values of n and l: ").split()
                self.n = int(ninp)
                self.l = int(linp)

                if self.n > self.l:
                    break
                else:
                    print(
                        'Error: principal quantum number "n" has to be greater than orbital angular momentum "l". '
                        'Please enter the values again.')

            # Calculate total angular momentum j and k
            if self.l == 0:
                self.j = 1 / 2
                self.k = -(self.j + 1 / 2)
            else:
                spin = input("Enter spin (up or down):")
                if spin == "up":
                    self.j = self.l - 1 / 2
                    self.k = self.j + 1 / 2
                else:
                    self.j = self.l + 1 / 2
                    self.k = -(self.j + 1 / 2)
        else:
            try:
                print(f"Attempting to read from: {input_file_path}")
                if os.path.isfile(input_file_path) and self.input_file is True:
                    with open(input_file_path, newline='') as file:
                        # Read each value from separate lines
                        reader = csv.reader(file)
                        values = [float(row[0]) for row in reader]

                        if len(values) < 5:
                            raise ValueError(f"Expected 5 values, but got {len(values)}.")

                        # Assign values to class attributes
                        self.system_index = int(values[0])
                        self.n = int(values[1])
                        self.l = int(values[2])
                        self.j = values[3]
                        self.k = values[4]

                        print(f"System index: {self.system_index}, n: {self.n}, l: {self.l}, j: {self.j}, k: {self.k}")
                else:
                    raise FileNotFoundError(f"File {input_file_path} not found.")
            except FileNotFoundError as e:
                print(str(e))
                raise ValueError("No inpdc.csv file exists!")
            except ValueError as e:
                print(f"Error reading file: {str(e)}")


class TwoBodyDiracCoulomb:
    pass


class Dynamic(QuantumNumber):
    def __init__(self, hyperparameters):
        super().__init__(n=None, l=None, j=None, k=None, fsc=None, sindex=None)
        self.input_file = hyperparameters['input_file']
        self.mass_relation = hyperparameters['mass_relation']
        self.subsystem = hyperparameters['subsystem']
        self._user_input()

    def _user_input(self):
        print("Dynamic System - Relativistic Description")
        print("System dynamically adapts to the corresponding mass relation selection, among the following choices:")
        if not os.path.isfile("./uinput/inpkg.csv") or self.input_file is False:
            if self.mass_relation == "m1>>m2":
                if self.subsystem == "Muonium":
                    self.system_index = 1
                elif self.subsystem == "e-Tauonium":
                    self.system_index = 2
                elif self.subsystem == "μ-Tauonium":
                    self.system_index = 3
                elif self.subsystem == "NewSystemDiffMass":
                    self.system_index = 4
                else:
                    raise ValueError("Invalid subsystem for m1>>m2")
            elif self.mass_relation == "m1=m2":
                if self.subsystem == "Positronium":
                    self.system_index = 5
                elif self.subsystem == "True Muonium":
                    self.system_index = 6
                elif self.subsystem == "True Tauonium":
                    self.system_index = 7
                elif self.subsystem == "NewSystemEqMass":
                    self.system_index = 8
                else:
                    raise ValueError("Invalid subsystem for m1=m2")
            elif self.mass_relation == "dynamic":
                self.system_index = int(
                    input("Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, "
                          "5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit) :"))
                if self.system_index == 0:
                    raise ValueError("Please select a valid System: 1, 2, 3, 4, 5, 6, 7, or 8")
            else:
                raise ValueError("Invalid mass relation")

            # Loop until valid n and l are provided
            while True:
                ninp, linp = input("Enter the values of n and l: ").split()
                self.n = int(ninp)
                self.l = int(linp)

                if self.n > self.l:
                    break
                else:
                    print(
                        'Error: principal quantum number "n" has to be greater than orbital angular momentum "l". '
                        'Please enter the values again.')

            # if self.fsc is None:
            #     self.fsc = float(input("enter coupling constant: "))
        else:
            try:
                with open("./uinput/inpkg.csv") as file:
                    reader = csv.reader(file)
                    rows = [row for row in reader]
                # [sindex, ninp, linp, fsc_inp] = rows
                [sindex, ninp, linp] = rows
                self.system_index = int(sindex[0])
                self.n = int(ninp[0])
                self.l = int(linp[0])
                # if self.fsc is None:
                #     self.fsc = float(fsc_inp[0])
            except FileNotFoundError:
                raise ValueError("No inpkg.csv file exists!")


class KleinGordon(QuantumNumber):
    def __init__(self, hyperparameters):
        super().__init__(n=None, l=None, j=None, k=None, fsc=None, sindex=None)
        self.input_file = hyperparameters['input_file']
        self.mass_relation = hyperparameters['mass_relation']
        self.subsystem = hyperparameters['subsystem']
        self._user_input()

    def _user_input(self):
        print("Two-body Relativistic Description")
        print("Klein-Gordon 2nd order ODE")
        if not os.path.isfile("./uinput/inpkg.csv") or self.input_file is False:
            if self.mass_relation == "m1>>m2":
                if self.subsystem == "Muonium":
                    self.system_index = 1
                elif self.subsystem == "e-Tauonium":
                    self.system_index = 2
                elif self.subsystem == "μ-Tauonium":
                    self.system_index = 3
                elif self.subsystem == "NewSystemDiffMass":
                    self.system_index = 4
                else:
                    raise ValueError("Invalid subsystem for m1>>m2")
            elif self.mass_relation == "m1=m2":
                if self.subsystem == "Positronium":
                    self.system_index = 5
                elif self.subsystem == "True Muonium":
                    self.system_index = 6
                elif self.subsystem == "True Tauonium":
                    self.system_index = 7
                elif self.subsystem == "NewSystemEqMass":
                    self.system_index = 8
                else:
                    raise ValueError("Invalid subsystem for m1=m2")
            elif self.mass_relation == "dynamic":
                self.system_index = int(
                    input("Choose system: (1: Muonium, 2: e-Tauonium, 3: μ-Tauonium, 4: NewSystemDiffMass, "
                          "5: Positronium, 6: True Muonium, 7: True Tauonium, 8: NewSystemEqMass, 0: Exit) :"))
                if self.system_index == 0:
                    raise ValueError("Please select a valid System: 1, 2, 3, 4, 5, 6, 7, or 8")
            else:
                raise ValueError("Invalid mass relation")

            # Loop until valid n and l are provided
            while True:
                ninp, linp = input("Enter the values of n and l: ").split()
                self.n = int(ninp)
                self.l = int(linp)

                if self.n > self.l:
                    break
                else:
                    print(
                        'Error: principal quantum number "n" has to be greater than orbital angular momentum "l". '
                        'Please enter the values again.')
            # if self.fsc is None:
            #     self.fsc = float(input("enter coupling constant: "))
        else:
            try:
                with open("./uinput/inpkg.csv") as file:
                    reader = csv.reader(file)
                    rows = [row for row in reader]
                # [sindex, ninp, linp, fsc_inp] = rows
                [sindex, ninp, linp] = rows
                self.system_index = int(sindex[0])
                self.n = int(ninp[0])
                self.l = int(linp[0])
                # if self.fsc is None:
                #     self.fsc = float(fsc_inp[0])
            except FileNotFoundError:
                raise ValueError("No inpkg.csv file exists!")
