"""Side-effect-free canonical reduced-radial Dirac helpers for Stage 1D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


LEARNABLE_COEFFICIENTS = "LEARNABLE_COEFFICIENTS"
FIXED_UNIT_COEFFICIENTS = "FIXED_UNIT_COEFFICIENTS"
VALID_MODES = {LEARNABLE_COEFFICIENTS, FIXED_UNIT_COEFFICIENTS}
EXTERNAL_LEARNABLE_MODE = "learnable"
EXTERNAL_FIXED_MODE = "fixed"
EXTERNAL_TO_INTERNAL_MODE = {
    EXTERNAL_LEARNABLE_MODE: LEARNABLE_COEFFICIENTS,
    EXTERNAL_FIXED_MODE: FIXED_UNIT_COEFFICIENTS,
}
INTERNAL_TO_EXTERNAL_MODE = {value: key for key, value in EXTERNAL_TO_INTERNAL_MODE.items()}


def normalize_coefficient_mode(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in VALID_MODES:
        return raw
    normalized = raw.lower()
    if normalized in EXTERNAL_TO_INTERNAL_MODE:
        return EXTERNAL_TO_INTERNAL_MODE[normalized]
    raise ValueError("dirac_coefficient_mode must be 'learnable' or 'fixed'")


def external_coefficient_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    internal = normalize_coefficient_mode(mode)
    return INTERNAL_TO_EXTERNAL_MODE[internal]

HBARC_MEV_FM = 197.3269804
FINE_STRUCTURE_CONSTANT = 0.0072973525693
E_SQUARED_MEV_FM = 1.4399764
ELECTRON_REST_ENERGY_MEV = 0.51099895000
GROUND_REFERENCE_ENERGY_MEV = 0.5109853441257417
DENOMINATOR_RELATIVE_FACTOR = 100.0 * np.finfo(float).eps
MIN_ABSOLUTE_SCALE = np.sqrt(np.finfo(float).tiny)
SHAPE_RELATIVE_FACTOR = 100.0 * np.finfo(float).eps


@dataclass(frozen=True)
class BranchLayout:
    mode: str
    hidden_units: int
    expected_length: int
    f_network: slice
    g_network: slice
    c_f_index: int | None
    c_g_index: int | None


@dataclass(frozen=True)
class UnpackedVector:
    mode: str
    hidden_units: int
    f_network: np.ndarray
    g_network: np.ndarray
    c_f: float
    c_g: float


def validate_finite(name: str, *values) -> None:
    for value in values:
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")


def _radial_grid(r) -> np.ndarray:
    radial = np.asarray(r, dtype=float)
    if radial.ndim != 1 or radial.size < 2:
        raise ValueError("radial grid must be a one-dimensional array with at least two points")
    validate_finite("radial grid", radial)
    if np.any(np.diff(radial) <= 0):
        raise ValueError("radial grid must be strictly increasing")
    return radial


def _trapz(values: np.ndarray, radial: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, radial))
    return float(np.trapz(values, radial))


def branch_layout(hidden_units: int, mode: str) -> BranchLayout:
    if not isinstance(hidden_units, int) or hidden_units <= 0:
        raise ValueError("hidden_units must be a positive integer")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown coefficient mode: {mode!r}")
    h = hidden_units
    if mode == LEARNABLE_COEFFICIENTS:
        return BranchLayout(mode, h, 6 * h + 2, slice(0, 3 * h), slice(3 * h + 1, 6 * h + 1), 3 * h,
                            6 * h + 1)
    return BranchLayout(mode, h, 6 * h, slice(0, 3 * h), slice(3 * h, 6 * h), None, None)


def unpack_vector(vector, hidden_units: int, mode: str) -> UnpackedVector:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError("parameter vector must be one-dimensional")
    validate_finite("parameter vector", values)
    layout = branch_layout(hidden_units, mode)
    if values.size != layout.expected_length:
        raise ValueError(
            f"{mode} with h={hidden_units} requires exactly {layout.expected_length} values; got {values.size}"
        )
    if mode == LEARNABLE_COEFFICIENTS:
        c_f = float(values[layout.c_f_index])
        c_g = float(values[layout.c_g_index])
    else:
        c_f = 1.0
        c_g = 1.0
    return UnpackedVector(mode, hidden_units, values[layout.f_network].copy(), values[layout.g_network].copy(), c_f,
                          c_g)


def _activation_and_derivative(z: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray]:
    if activation == "tanh":
        value = np.tanh(z)
        return value, 1.0 - value * value
    if activation == "sigmoid":
        value = np.empty_like(z)
        positive = z >= 0
        value[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        value[~positive] = exp_z / (1.0 + exp_z)
        return value, value * (1.0 - value)
    if activation == "relu":
        return np.maximum(0.0, z), (z > 0.0).astype(float)
    if activation == "smplus":
        value = np.logaddexp(0.0, z)
        sigmoid, _ = _activation_and_derivative(z, "sigmoid")
        return value, sigmoid
    raise ValueError(f"unsupported activation: {activation!r}")


def neural_value_and_derivative(r, network_parameters, hidden_units: int, activation: str = "tanh"):
    radial = np.asarray(r, dtype=float)
    params = np.asarray(network_parameters, dtype=float)
    if params.ndim != 1 or params.size != 3 * hidden_units:
        raise ValueError(f"network requires exactly {3 * hidden_units} parameters")
    validate_finite("neural input", radial, params)
    a = params[:hidden_units]
    b = params[hidden_units:2 * hidden_units]
    d = params[2 * hidden_units:3 * hidden_units]
    z = radial.reshape(-1, 1) * b.reshape(1, -1) + d.reshape(1, -1)
    activation_value, activation_derivative = _activation_and_derivative(z, activation)
    neural = np.sum(a.reshape(1, -1) * activation_value, axis=1)
    derivative = np.sum(a.reshape(1, -1) * b.reshape(1, -1) * activation_derivative, axis=1)
    return neural.reshape(radial.shape), derivative.reshape(radial.shape)


def reduced_component(r, network_parameters, hidden_units: int, beta: float, coefficient: float,
                      activation: str = "tanh"):
    radial = np.asarray(r, dtype=float)
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(coefficient):
        raise ValueError("component coefficient must be finite")
    neural, neural_derivative = neural_value_and_derivative(radial, network_parameters, hidden_units, activation)
    exponential = np.exp(-beta * radial)
    value = coefficient * radial * exponential * neural
    derivative = coefficient * exponential * (radial * neural_derivative + (1.0 - beta * radial) * neural)
    validate_finite("reduced component", value, derivative)
    return value, derivative


def evaluate_spinor(r, vector, hidden_units: int, beta: float, mode: str, activation: str = "tanh") -> Dict[str, np.ndarray]:
    unpacked = unpack_vector(vector, hidden_units, mode)
    f_value, f_derivative = reduced_component(r, unpacked.f_network, hidden_units, beta, unpacked.c_f, activation)
    g_value, g_derivative = reduced_component(r, unpacked.g_network, hidden_units, beta, unpacked.c_g, activation)
    return {
        "F": f_value,
        "G": g_value,
        "dF": f_derivative,
        "dG": g_derivative,
        "c_f": unpacked.c_f,
        "c_g": unpacked.c_g,
    }


def pure_coulomb_potential(r, e_squared_mev_fm: float = E_SQUARED_MEV_FM) -> np.ndarray:
    radial = np.asarray(r, dtype=float)
    validate_finite("Coulomb input", radial, e_squared_mev_fm)
    if e_squared_mev_fm <= 0:
        raise ValueError("e_squared_mev_fm must be positive")
    if np.any(radial <= 0):
        raise ValueError("Coulomb potential requires r>0")
    return -float(e_squared_mev_fm) / radial


def coupled_residuals(r, f_value, g_value, f_derivative, g_derivative, kappa: float, rest_energy_mev: float,
                      eigenenergy_mev: float, hbarc_mev_fm: float = HBARC_MEV_FM,
                      potential_mev=None) -> Tuple[np.ndarray, np.ndarray]:
    radial = np.asarray(r, dtype=float)
    if np.any(radial <= 0):
        raise ValueError("residual evaluation requires r>0")
    f_value = np.asarray(f_value, dtype=float)
    g_value = np.asarray(g_value, dtype=float)
    f_derivative = np.asarray(f_derivative, dtype=float)
    g_derivative = np.asarray(g_derivative, dtype=float)
    if potential_mev is None:
        potential_mev = pure_coulomb_potential(radial)
    potential_mev = np.asarray(potential_mev, dtype=float)
    validate_finite("residual inputs", radial, f_value, g_value, f_derivative, g_derivative, potential_mev,
                    rest_energy_mev, eigenenergy_mev, hbarc_mev_fm, kappa)
    if hbarc_mev_fm <= 0:
        raise ValueError("hbarc must be positive")
    residual_f = (f_derivative + kappa * f_value / radial
                  - (rest_energy_mev + eigenenergy_mev - potential_mev) * g_value / hbarc_mev_fm)
    residual_g = (g_derivative - kappa * g_value / radial
                  - (rest_energy_mev - eigenenergy_mev + potential_mev) * f_value / hbarc_mev_fm)
    validate_finite("residual outputs", residual_f, residual_g)
    return residual_f, residual_g


def _scaled_spinor(f_value, g_value, radial):
    radial = _radial_grid(radial)
    f_value = np.asarray(f_value, dtype=float)
    g_value = np.asarray(g_value, dtype=float)
    if f_value.shape != radial.shape or g_value.shape != radial.shape:
        raise ValueError("F and G must match the radial grid")
    validate_finite("spinor", f_value, g_value)
    scale = float(max(np.max(np.abs(f_value)), np.max(np.abs(g_value))))
    if scale == 0.0:
        raise ValueError("zero spinor norm")
    if scale < MIN_ABSOLUTE_SCALE:
        raise ValueError("near-zero spinor norm")
    scaled_f = f_value / scale
    scaled_g = g_value / scale
    scaled_norm = _trapz(scaled_f * scaled_f + scaled_g * scaled_g, radial)
    span = float(radial[-1] - radial[0])
    if not np.isfinite(scaled_norm) or scaled_norm <= SHAPE_RELATIVE_FACTOR * span:
        raise ValueError("zero or numerically unresolved spinor norm")
    return radial, scale, scaled_f, scaled_g, scaled_norm


def joint_spinor_norm(f_value, g_value, r) -> float:
    _, scale, _, _, scaled_norm = _scaled_spinor(f_value, g_value, r)
    log_norm = 2.0 * np.log(scale) + np.log(scaled_norm)
    if log_norm <= np.log(np.finfo(float).tiny):
        raise ValueError("near-zero spinor norm")
    if log_norm >= np.log(np.finfo(float).max):
        raise ValueError("spinor norm overflows binary64")
    norm = float(np.exp(log_norm))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid spinor norm")
    return norm


def normalize_spinor(f_value, g_value, r):
    _, _, scaled_f, scaled_g, scaled_norm = _scaled_spinor(f_value, g_value, r)
    divisor = np.sqrt(scaled_norm)
    normalized_f = scaled_f / divisor
    normalized_g = scaled_g / divisor
    validate_finite("normalized spinor", normalized_f, normalized_g)
    return normalized_f, normalized_g


def canonical_energy_terms(r, f_value, g_value, potential_mev, rest_energy_mev: float):
    radial, _, scaled_f, scaled_g, _ = _scaled_spinor(f_value, g_value, r)
    potential = np.asarray(potential_mev, dtype=float)
    if potential.shape != radial.shape:
        raise ValueError("potential must match the radial grid")
    validate_finite("energy inputs", potential, rest_energy_mev)
    difference = scaled_f * scaled_f - scaled_g * scaled_g
    sum_squares = scaled_f * scaled_f + scaled_g * scaled_g
    denominator = _trapz(difference, radial)
    spinor_sum = _trapz(sum_squares, radial)
    potential_term = _trapz(potential * difference, radial)
    numerator = potential_term + rest_energy_mev * spinor_sum
    validate_finite("energy terms", denominator, spinor_sum, potential_term, numerator)
    if abs(denominator) <= DENOMINATOR_RELATIVE_FACTOR * spinor_sum:
        raise ValueError("canonical energy denominator is numerically singular")
    return {
        "potential_term": potential_term,
        "spinor_sum": spinor_sum,
        "numerator": numerator,
        "denominator": denominator,
    }


def canonical_energy_numerator(r, f_value, g_value, potential_mev, rest_energy_mev: float) -> float:
    return float(canonical_energy_terms(r, f_value, g_value, potential_mev, rest_energy_mev)["numerator"])


def canonical_energy_denominator(r, f_value, g_value, potential_mev, rest_energy_mev: float) -> float:
    return float(canonical_energy_terms(r, f_value, g_value, potential_mev, rest_energy_mev)["denominator"])


def canonical_integral_energy(r, f_value, g_value, potential_mev, rest_energy_mev: float) -> float:
    terms = canonical_energy_terms(r, f_value, g_value, potential_mev, rest_energy_mev)
    energy = terms["numerator"] / terms["denominator"]
    if not np.isfinite(energy):
        raise ValueError("canonical integral energy is non-finite")
    return float(energy)


def primary_objective(r, spinor, kappa: float, rest_energy_mev: float, eigenenergy_mev: float,
                      hbarc_mev_fm: float = HBARC_MEV_FM):
    radial = _radial_grid(r)
    residual_f, residual_g = coupled_residuals(radial, spinor["F"], spinor["G"], spinor["dF"], spinor["dG"],
                                               kappa, rest_energy_mev, eigenenergy_mev, hbarc_mev_fm)
    ell_c = hbarc_mev_fm / rest_energy_mev
    residual_term = ell_c * ell_c * _trapz(residual_f * residual_f + residual_g * residual_g, radial)
    common_scale = float(max(np.max(np.abs(spinor["F"])), np.max(np.abs(spinor["G"]))))
    if common_scale == 0.0:
        norm = 0.0
        norm_penalty = 1.0
        zero_penalized = True
    else:
        norm = joint_spinor_norm(spinor["F"], spinor["G"], radial)
        norm_penalty = (norm - 1.0) ** 2
        zero_penalized = False
    total = residual_term + norm_penalty
    validate_finite("primary objective", residual_term, norm_penalty, total)
    return {
        "ell_c_fm": float(ell_c),
        "residual_term": float(residual_term),
        "norm": float(norm),
        "norm_penalty": float(norm_penalty),
        "total_objective": float(total),
        "zero_penalized": zero_penalized,
        "finite": True,
    }


def normalized_residual_objective(r, spinor, kappa: float, rest_energy_mev: float, eigenenergy_mev: float,
                                  hbarc_mev_fm: float = HBARC_MEV_FM):
    radial = _radial_grid(r)
    normalized_f, normalized_g = normalize_spinor(spinor["F"], spinor["G"], radial)
    norm = joint_spinor_norm(spinor["F"], spinor["G"], radial)
    scale = np.sqrt(norm)
    normalized_df = spinor["dF"] / scale
    normalized_dg = spinor["dG"] / scale
    residual_f, residual_g = coupled_residuals(radial, normalized_f, normalized_g, normalized_df, normalized_dg,
                                               kappa, rest_energy_mev, eigenenergy_mev, hbarc_mev_fm)
    ell_c = hbarc_mev_fm / rest_energy_mev
    residual_term = ell_c * ell_c * _trapz(residual_f * residual_f + residual_g * residual_g, radial)
    validate_finite("normalized residual objective", residual_term)
    return float(residual_term)


def dirac_energy_eigenvalue(n: int, kappa: float, mass_mev: float = ELECTRON_REST_ENERGY_MEV,
                            alpha: float = FINE_STRUCTURE_CONSTANT) -> float:
    denominator = n - abs(kappa) + np.sqrt(kappa * kappa - alpha * alpha)
    return float(mass_mev / np.sqrt(1.0 + (alpha / denominator) ** 2))


def export_table(r, f_value, g_value):
    radial = np.asarray(r, dtype=float)
    f_value = np.asarray(f_value, dtype=float)
    g_value = np.asarray(g_value, dtype=float)
    if radial.shape != f_value.shape or radial.shape != g_value.shape:
        raise ValueError("export arrays must share one shape")
    validate_finite("export arrays", radial, f_value, g_value)
    return ("r_fm", "F_reduced", "G_reduced"), np.column_stack((radial, f_value, g_value))
