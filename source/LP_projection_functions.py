import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as R
from scipy.optimize import root_scalar
from scipy.special import jv, kn, jn_zeros


def left_side(l, u):
    return u * (jv(l + 1, u) / jv(l, u))


def right_side(l, V, u):
    w = np.sqrt(V**2 - u**2)
    return w * (kn(l + 1, w) / kn(l, w))


def diff(l, V, u):
    return left_side(l, u) - right_side(l, V, u)


def get_guided_modes(l, m, V, radius, R, PHI, dA, verbose=False):
    """
    Compute a single guided LP(l,m) mode for a step-index fiber and return its transverse
    E field component, which can be used as either E_x or E_y

    Parameters
    ----------
    l : int
            Azimuthal order of the LP mode (nonnegative integer).
    m : int
            Radial mode index (positive integer, m >= 1).
    V : float
            Normalized frequency parameter of the fibre (k0 * a * sqrt(n_core^2 - n_clad^2)).
    radius : float
            Core radius (same length units as values in R).
    R : array_like
            Radial coordinate grid (same shape as PHI). Elements are radii at which the field
            is evaluated. Values greater than `radius` are interpreted as cladding points.
    PHI : array_like
            Azimuthal coordinate grid (radians), same shape as R, used to compute the
            cos(l*phi) and sin(l*phi) angular dependence.
    verbose : bool, optional
            If True, prints intermediate root-finding results. Default is False.

    Returns
    -------
    dict or None
            If the mode is guided (V > cutoff) and the root-finding succeeds, returns a
            dictionary with the following keys:
                - "l": int, same as input l
                - "m": int, same as input m
                - "u": float, solved transverse core eigenvalue u
                - "cos": ndarray, E(r,phi) * cos(l*phi) (same shape as R/PHI)
                - "sin": ndarray, E(r,phi) * sin(l*phi) (same shape as R/PHI)
            If V is below the mode cutoff, returns None.
    """

    # Calculate the cutoff of the mode
    if l == 0 and m == 1:
        cutoff = 1e-6
    elif l == 0:
        cutoff = jn_zeros(1, m - 1)[-1]
    else:
        cutoff = jn_zeros(l - 1, m)[-1]

    if V <= cutoff:
        return None

    bracket_min = cutoff + 1e-6
    bracket_max = min(jn_zeros(l, m)[-1] - 1e-6, V - 1e-6)
    bracket = (bracket_min, bracket_max)

    result = root_scalar(f=lambda x: diff(l, V, x), bracket=bracket, method="bisect")

    if verbose:
        print("\n", f"Mode LP{l}{m}\n", result)

    if not result.converged:
        print(
            f"Warning: Mode LP{l}{m} is allowed by the V parameter (V={V}), but root finding did not converge."
        )
        return None

    u_lp = result.root
    w_lp = np.sqrt(V**2 - u_lp**2)

    B = jv(l, u_lp) / kn(l, w_lp)

    E_lp_core = jv(l, u_lp / radius * R)
    E_lp_cladding = B * kn(l, w_lp / radius * R)

    # --- Adjust for their respective domains ---
    E_lp_core[R > radius] = 0
    E_lp_cladding[R <= radius] = 0

    E_lp_tot = E_lp_core + E_lp_cladding

    P_norm = np.sum(np.abs(E_lp_tot * np.exp(+1j * l * PHI)) ** 2) * dA

    result = {
        "l": l,
        "m": m,
        "u": u_lp,
        "p_phi": E_lp_tot * np.exp(+1j * l * PHI) / np.sqrt(P_norm),
        "m_phi": E_lp_tot * np.exp(-1j * l * PHI) / np.sqrt(P_norm) if l != 0 else 0 + 0j,
    }

    return result


def get_LP_modes_projection_coefficients(E_input, mode, dA):
    """
    Compute projection coefficients of a two-component electric field onto an LP mode.

    Parameters
    ----------
    E_input : sequence of numpy.ndarray
        Two-component sampled complex electric field given as (E_x, E_y).
    mode : dict
    dA : float
        Area element associated with each grid sample (integration weight).

    Returns
    -------
    dict
        A dictionary with the following entries:
          - "l", "m", "u": mode identifiers copied from the input mode dict.
          - "P_mode" (float): modal power (norm) computed as sum(|E_mode_cos|^2) * dA.
          - "x_cos", "y_cos", "x_sin", "y_sin": projection coefficients (complex or real)
            corresponding to the overlaps of the input field components with the
            cosine and sine mode fields, normalized by P_mode. Coefficients whose
            absolute value is within a numerical tolerance (1e-10) are returned as 0.

    Notes
    -----
    - Overlap integrals are evaluated as sum(conj(E_mode) * E_input_component) * dA.
      Conjugation is applied to support future use of complex mode bases; for real
     -valued LP modes the conjugation has no effect.
    - P_mode is computed using the cosine-mode field only (assumes the cos/sin pair
      are normalized consistently).
    - A small absolute-tolerance (1e-10) is used to treat numerically negligible
      coefficients as zero to avoid spurious tiny complex residues.
    """

    E_input_x = E_input[0]
    E_input_y = E_input[1]

    E_mode_p_phi = mode.get("p_phi")
    E_mode_m_phi = mode.get("m_phi")

    # Overlap integrals:
    # The denominator is not needed since P=1 for every mode
    A_x_p_phi = np.sum(np.conj(E_mode_p_phi) * E_input_x) * dA
    A_y_p_phi = np.sum(np.conj(E_mode_p_phi) * E_input_y) * dA
    A_x_m_phi = np.sum(np.conj(E_mode_m_phi) * E_input_x) * dA
    A_y_m_phi = np.sum(np.conj(E_mode_m_phi) * E_input_y) * dA

    tol = 1e-10

    result = {
        "l": mode.get("l"),
        "m": mode.get("m"),
        "u": mode.get("u"),
        "x_p_phi": A_x_p_phi if not np.isclose(A_x_p_phi, 0, tol) else 0+0j,
        "y_p_phi": A_y_p_phi if not np.isclose(A_y_p_phi, 0, tol) else 0+0j,
        "x_m_phi": A_x_m_phi if not np.isclose(A_x_m_phi, 0, tol) else 0+0j,
        "y_m_phi": A_y_m_phi if not np.isclose(A_y_m_phi, 0, tol) else 0+0j,
    }

    return result


def get_complete_guided_field(guided_modes, df_coeff, X, Y):

    E_guided_x = np.zeros_like(X, dtype=complex)
    E_guided_y = np.zeros_like(Y, dtype=complex)
    sum_squared_coeff = 0

    for mode in guided_modes:
        if mode is None:
            continue

        l = mode["l"]
        m = mode["m"]

        coeff = df_coeff.loc[l, m]

        E_guided_x_lm = (
            coeff["x_p_phi"] * mode["p_phi"] + coeff["x_m_phi"] * mode["m_phi"]
        )
        E_guided_y_lm = (
            coeff["y_p_phi"] * mode["p_phi"] + coeff["y_m_phi"] * mode["m_phi"]
        )

        E_guided_x += E_guided_x_lm
        E_guided_y += E_guided_y_lm

        sum_squared_coeff += (
            np.abs(coeff["x_p_phi"]) ** 2
            + np.abs(coeff["x_m_phi"]) ** 2
            + np.abs(coeff["y_p_phi"]) ** 2
            + np.abs(coeff["y_m_phi"]) ** 2
        )

    return E_guided_x, E_guided_y, sum_squared_coeff


def _gaussian_electric_field_alligned(
    X,
    Y,
    z,
    dA,
    w0_x=0.5,
    w0_y=None,
    wavelength=1.0,
    polarization_angle=0.0,
    amplitude=1.0,
):
    """
    Return complex transverse electric field components (E_x, E_y) of an
    elliptical Gaussian beam in the plane at distance z from the waist at z=0.

    Parameters
    - z: propagation distance (same units as waist and wavelength)
    - w0_x, w0_y: waist radii (w0) along x and y (1/e^2 intensity). If
      w0_y is None it's set equal to w0_x.
    - wavelength: vacuum wavelength
    - x0, y0: beam center offset in the transverse plane
    - polarization_angle: angle (rad) of linear polarization measured from x
    - amplitude: peak amplitude at waist (z=0)

    Uses globals X, Y for the transverse grid.
    Returns complex arrays (E_x, E_y).
    """
    if w0_y is None:
        w0_y = w0_x

    k = 2 * np.pi / wavelength

    # Rayleigh ranges for each axis
    zR_x = np.pi * w0_x**2 / wavelength
    zR_y = np.pi * w0_y**2 / wavelength

    # spot sizes at z
    wx = w0_x * np.sqrt(1.0 + (z / zR_x) ** 2)
    wy = w0_y * np.sqrt(1.0 + (z / zR_y) ** 2)

    # curvature radii (avoid division by zero at z=0)
    # Rx = np.where(np.abs(z) < 1e-12, np.inf, z * (1.0 + (zR_x / z) ** 2))
    # Ry = np.where(np.abs(z) < 1e-12, np.inf, z * (1.0 + (zR_y / z) ** 2))
    z_safe = np.where(np.abs(z) < 1e-12, np.inf, z)
    Rx = z_safe * (1.0 + (zR_x / z_safe) ** 2)
    Ry = z_safe * (1.0 + (zR_y / z_safe) ** 2)

    # Gouy phases
    gouy = np.arctan(z / zR_x) + np.arctan(z / zR_y)

    # amplitude normalization so power is comparable at different z
    amp_norm = amplitude * np.sqrt((w0_x * w0_y) / (wx * wy))

    # real-space Gaussian envelope
    envelope = np.exp(-(X**2) / (wx**2) - (Y**2) / (wy**2))

    # curvature phase terms (handle infinite curvature as zero quadratic phase)
    phase_quadratic = np.zeros_like(X, dtype=float)
    phase_quadratic += np.where(np.isfinite(Rx), -k * (X**2) / (2.0 * Rx), 0.0)
    phase_quadratic += np.where(np.isfinite(Ry), -k * (Y**2) / (2.0 * Ry), 0.0)

    # longitudinal phase and total phase
    phase_longitudinal = k * z
    total_phase = phase_longitudinal + phase_quadratic - gouy

    field = amp_norm * envelope * np.exp(1j * total_phase)

    Ex = field * np.cos(polarization_angle)
    Ey = field * np.sin(polarization_angle)

    # power is normalized to 1
    P = np.sum((np.abs(Ex) ** 2 + np.abs(Ey) ** 2)) * dA
    Ex /= np.sqrt(P)
    Ey /= np.sqrt(P)

    return Ex, Ey


def tilted_gaussian_electric_field(
    X_lab,
    Y_lab,
    Z_lab,
    x_waist=0.0,
    y_waist=0.0,
    z_waist=0.0,
    euler_alpha=0.0,
    euler_beta=0.0,
    euler_gamma=0.0,
    **kwargs,
):
    R_matrix = R.from_euler("zxy", [euler_alpha, euler_beta, euler_gamma]).as_matrix().T

    X_rel = X_lab - x_waist
    Y_rel = Y_lab - y_waist
    Z_rel = Z_lab - z_waist

    X_beam = R_matrix[0, 0] * X_rel + R_matrix[0, 1] * Y_rel + R_matrix[0, 2] * Z_rel
    Y_beam = R_matrix[1, 0] * X_rel + R_matrix[1, 1] * Y_rel + R_matrix[1, 2] * Z_rel
    Z_beam = R_matrix[2, 0] * X_rel + R_matrix[2, 1] * Y_rel + R_matrix[2, 2] * Z_rel

    return _gaussian_electric_field_alligned(X_beam, Y_beam, Z_beam, **kwargs)


def get_tilted_beam_from_incidence(
    X_lab,
    Y_lab,
    z_plane,
    x_incidence,
    y_incidence,
    dist_to_waist,
    euler_alpha,
    euler_beta,
    euler_gamma,
    **kwargs,
):

    Z_lab = z_plane

    P_inc = np.array([x_incidence, y_incidence, z_plane])

    R_matrix = R.from_euler("zxy", [euler_alpha, euler_beta, euler_gamma]).as_matrix()

    # Propagation vector seen from the Lab f.o.r.
    k_vec = R_matrix @ np.array([0, 0, 1])

    # Waist position in the Lab f.o.r.
    P_waist = P_inc - dist_to_waist * k_vec

    return tilted_gaussian_electric_field(
        X_lab=X_lab,
        Y_lab=Y_lab,
        Z_lab=Z_lab,
        x_waist=P_waist[0],
        y_waist=P_waist[1],
        z_waist=P_waist[2],
        euler_alpha=euler_alpha,
        euler_beta=euler_beta,
        euler_gamma=euler_gamma,
        **kwargs,
    )
