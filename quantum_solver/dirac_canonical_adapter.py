"""Isolated adapter from runtime-like arrays to the canonical shadow helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dirac_canonical as dc


@dataclass(frozen=True)
class DiracStateMetadata:
    n: int
    l: int
    j: float
    kappa: float
    reference_energy_mev: float
    fine_structure_constant: float = dc.FINE_STRUCTURE_CONSTANT

    def validate(self):
        if self.n < 1 or self.l < 0 or self.kappa == 0:
            raise ValueError("invalid Dirac quantum-state metadata")
        computed = dc.dirac_energy_eigenvalue(
            self.n, self.kappa, alpha=self.fine_structure_constant
        )
        if not np.isclose(computed, self.reference_energy_mev, rtol=0.0, atol=5e-15):
            raise ValueError("Dirac reference eigenenergy mismatch")


@dataclass(frozen=True)
class GroundStateMetadata(DiracStateMetadata):
    n: int = 1
    l: int = 0
    j: float = 0.5
    kappa: float = -1.0
    reference_energy_mev: float = dc.GROUND_REFERENCE_ENERGY_MEV
    fine_structure_constant: float = dc.FINE_STRUCTURE_CONSTANT

    def validate(self):
        if (self.n, self.l, self.j, self.kappa) != (1, 0, 0.5, -1.0):
            raise ValueError("GroundStateMetadata accepts only n=1,l=0,j=0.5,kappa=-1")
        super().validate()


class ShadowDiracAdapter:
    def __init__(self, hidden_units: int, beta: float, coefficient_mode: str, activation: str = "tanh",
                 metadata: DiracStateMetadata | None = None):
        self.hidden_units = hidden_units
        self.beta = beta
        self.coefficient_mode = dc.normalize_coefficient_mode(coefficient_mode)
        self.activation = activation
        self.metadata = metadata or GroundStateMetadata()
        self.metadata.validate()
        self.layout = dc.branch_layout(hidden_units, self.coefficient_mode)

    def evaluate(self, radial, vector):
        return dc.evaluate_spinor(radial, vector, self.hidden_units, self.beta, self.coefficient_mode, self.activation)

    def objective(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        return dc.primary_objective(radial, spinor, self.metadata.kappa, dc.ELECTRON_REST_ENERGY_MEV,
                                    self.metadata.reference_energy_mev)

    def fallback_objective(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        return dc.normalized_residual_objective(radial, spinor, self.metadata.kappa, dc.ELECTRON_REST_ENERGY_MEV,
                                                self.metadata.reference_energy_mev)


    def norm(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        return dc.joint_spinor_norm(spinor["F"], spinor["G"], radial)

    def energy_terms(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        potential = dc.pure_coulomb_potential(radial)
        return dc.canonical_energy_terms(radial, spinor["F"], spinor["G"], potential,
                                         dc.ELECTRON_REST_ENERGY_MEV)

    def normalized(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        f_norm, g_norm = dc.normalize_spinor(spinor["F"], spinor["G"], radial)
        return f_norm, g_norm

    def energy(self, radial, vector):
        spinor = self.evaluate(radial, vector)
        potential = dc.pure_coulomb_potential(radial)
        return dc.canonical_integral_energy(radial, spinor["F"], spinor["G"], potential,
                                            dc.ELECTRON_REST_ENERGY_MEV)

    def export_data(self, radial, vector, normalized: bool = True):
        spinor = self.evaluate(radial, vector)
        if normalized:
            f_value, g_value = dc.normalize_spinor(spinor["F"], spinor["G"], radial)
        else:
            f_value, g_value = spinor["F"], spinor["G"]
        return dc.export_table(radial, f_value, g_value)
