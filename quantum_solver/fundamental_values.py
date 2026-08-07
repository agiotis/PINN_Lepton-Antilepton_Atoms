# quantum_solver/fundamental_values.py
import numpy as np


class FundamentalValues:
    def __init__(self):
        # Constants
        # alpha is the fine structure constant of equation e^2/(4*π*ε_0*h_bar *c) = 1/137
        # alpha = 1.0 / 137.035982  = self.fsc

        # self.Z = 68  # atomic number of pionic atom (Klein-Gordon case) used for analytic solution
        self.Z = 1  # newly defined atomic number that enters two-body Klein-Gordon equation
        self.A = 137  # atomic number ...

        self.hbarc = 197.3269804  # reduced planck constant multiplied by c squared
        self.fsc = 0.0072973525693  # fine structure constant 1 / 137.035999084 # we need to explicitly define this !!!!
        # fsc was tested as "coupling_constant = 0.07" and it seems to be working for equal masses of at most 100 MeV
        # = 0.1 GeV
        self.ifsc = 137.035999084  # inverse fine structure constant

        # initial value of mc2 = 139.5688 - From G. Gkartzonikas code !
        # updated mass_of_pion = 139.57039  (Klein-Gordon case) - agiotis finding: recent work
        # M. Trassinelli et al. Measurement of the charged pion mass using X-ray spectroscopy of exotic atoms,
        # Phys. Lett. B 759, 583 (2016),doi:10.1016/j.physletb.2016.06.025, 1605.03300.
        self.mc2 = 139.5688
        # normally we need the recuded mass which is # m_r = m/(1 + m/M) ~= m
        #
        # ˜

        # Subatomic particles masses (CODATA 2018)
        self.m_e = 0.51099895000  # electron mass in MeV  # we need to check user-defined parameters
        self.m_mu = 105.6583755  # muon mass in MeV
        self.m_p = 938.27208816  # proton mass in MeV
        self.m_tau = 1776.86  # tau mass in MeV

        self.a0ih = self.hbarc / (self.m_e * self.fsc)  # Bohr radius of the hydrogen
        self.eps_0 = 55.26349406  # $e^2 * eV^{-1} * μm^{-1}$
        self.e_charge = 1.4399764  # elementary charge squared in MeV * fm
        # we need to set self.e_charge within range 1,40-1,50 with step 0.01

    def twobody(self, sind, A=None, Z=None, N=None) -> list:
        """Two body purely leptonic systems"""
        if sind == 1:  # Muonium
            mass1 = self.m_mu
            mass2 = self.m_e
            partl = "Muonium"  # only when m1<<m2
        elif sind == 2:  # e_Tauonium
            mass1 = self.m_tau
            mass2 = self.m_e
            partl = "e_Tauonium"  # only when m1<<m2
        elif sind == 3:  # τ_Tauonium
            mass1 = self.m_tau
            mass2 = self.m_mu
            partl = "τ_Tauonium"  # only when m1<<m2
        elif sind == 5:  # Positronium
            mass1 = self.m_e
            mass2 = self.m_e
            partl = "Positronium"  # only when m1=m2 # m_r = m/(1 + m/M) ~= m
        elif sind == 6:  # True_Muonium (new system)
            mass1 = self.m_mu
            mass2 = self.m_mu
            partl = "True_Muonium"  # only when m1=m2
        elif sind == 7:  # True Tauonium
            mass1 = self.m_tau
            mass2 = self.m_tau
            partl = "True_Tauonium"  # only when m1=m2
        elif sind == 4:  # NewSystemDiffMass
            mass1, mass2 = input("Enter the different masses m1, m2 of the particles: ").split()
            mass1 = float(mass1)
            mass2 = float(mass2)
            partl = "NewSystemDiffMass"  # only when m1<m2
        elif sind == 8:  # NewSystemEqfMass
            mass1 = float(input("Enter one of the equal masses m1 or m2 of the particles: "))
            mass2 = mass1
            partl = "NewSystemEqMass"  # only when m1=m2
        else:
            exit()

        if sind >= 1 and sind <= 4:
            # reduced mass calculation for the above selected system --> rmass
            rmass = (mass1 * mass2) / (mass1 + mass2)
            if sind == 3:  # specifically here we select the correct mass of m_mu as the smaller particle
                alpha0 = self.m_mu / rmass * self.a0ih
            elif sind == 4:  # case of new system with m1<m2
                alpha0 = np.min([mass1, mass2]) / rmass * self.a0ih
            else:
                alpha0 = self.m_e / rmass * self.a0ih
        else:
            # reduced mass calculation for the above selected system --> rmass
            # The following code calculates the reduced mass or the pion neglecting binding energy. ASK PROF KOSMAS!!
            # if A is None or Z is None or N is None:
            #     raise ValueError("A, Z, and N must be provided for the Positronium, True_Muonium & True_Taunium cases")
            # m_p = self.m_p  # proton mass
            # m_n = self.mc2  # neutron mass (using pion mass for this context)
            # M = N * m_n + Z * m_p  # Total mass of the nucleus, neglecting binding energy
            # rmass = mass1 / (1 + mass1 / M)

            ##  CHECK 24-07-2024  if alhpha0 is correct !!!
            # rmass = np.min([mass1, mass2])
            rmass = (mass1 * mass2) / (mass1 + mass2)
            # rmass = mass1  # this case seems to be running for the relativistic two body integral energy calculation
            # if we use rmass = mass1/2 then we get a negative value for term4 and thus NaN value for sqrt()..
            # rmass = mass1/2
            alpha0 = self.m_e / rmass * self.a0ih  # this is needs testing in Klein-Gordon (it's ok for the Positronium)
            # alpha0 will be computed with a new equation for other particles.

            # in above case self.a0i includes the self.fsc fixed to 1/137 !
        # print("reduced mass =", rmass)
        # print("mass1 =", mass1)
        return [mass1, mass2, rmass, alpha0, partl]
