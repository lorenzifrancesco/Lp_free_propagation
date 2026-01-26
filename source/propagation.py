import numpy as np
import pandas as pd
from scipy.special import jv, kn
from scipy.integrate import simpson, trapezoid


def fiber_propagation(df_coeff, n1, a, lam, z_fiber):
    """
    Computes the fiber propagation by applying phase factors to the coefficients.

    Parameters:
        df_coeff (pd.DataFrame): DataFrame containing mode coefficients and 'u' values.
        n1 (float): Refractive index of the core.
        a (float): Core radius of the fiber.
        lam (float): Wavelength of the light.
        z_fiber (float): Propagation distance along the fiber.

    Returns:
        pd.DataFrame: Updated DataFrame with modified coefficients.
    """
    df = df_coeff.copy()
    k0 = 2 * np.pi / lam

    df["beta_lm"] = (np.sqrt((n1 * k0) ** 2 - (df["u"] / a) ** 2)).astype(np.complex128)
    df["phase_fact"] = np.exp(-1j * df["beta_lm"] * z_fiber)

    columns = ["x_p_phi", "y_p_phi", "x_m_phi", "y_m_phi"]
    df.loc[:, columns] = df.loc[:, columns].multiply(df["phase_fact"], axis=0)

    return df.drop(columns=["beta_lm", "phase_fact"])


def verify_resolution(
    dx,
    dz,
    lambda_0,
    NA,
    oversampling_factor_x,
    oversampling_factor_z,
):

    nyquist_dx = lambda_0 / (2 * NA * oversampling_factor_x)

    if dx > nyquist_dx:
        raise ValueError(
            f"Resolution dx={dx:.2e} exceeds Nyquist limit with required oversampling {nyquist_dx:.2e}.\n"
            f"Required dx <= lambda_0 / (2 * NA * oversampling_factor))"
        )

    nyquist_dz = lambda_0 / (oversampling_factor_z * NA**2)

    if dz > nyquist_dz:
        raise ValueError(
            f"Resolution dz={dz:.2e} exceeds Nyquist limit with required oversampling {nyquist_dz:.2e}.\n"
            f"Required dz <= lambda_0 / (oversampling_factor * NA^2)"
        )


def free_propagation_asm_hankel(
    guided_modes,
    df_coeff,
    z,
    NA,
    Rz_factor,
    dx,
    k_max_factor,
    fiber_V,
    R_origin,
    min_point_per_period=10,
    radius=1.0,
    lambda_0=1.0,
    return_z_gradient=False,
):
    """
    Propagate guided fiber modes using analytical Hankel transforms and ASM.
    Computes the propagated electric field components (Ex, Ey) at distance z from
    the fiber end using the Angular Spectrum Method (ASM) with analytical Fourier/Hankel
    transforms for the spatial mode profiles.

        Args:
            guided_modes: List of mode dictionaries containing 'l', 'm', 'u' parameters
            df_coeff: DataFrame with polarization coefficients indexed by (l, m)
            z: Propagation distance
            NA: Numerical aperture of the fiber
            Rz_factor: Scaling factor for transverse grid size
            min_dx: Minimum resolution required for the x and y axis
            fiber_V: V-number of the fiber
            R_origin: Minimum allowed grid radius
            min_point_per_period: Minimum sampling points per Bessel oscillation (default: 10)
            radius: Fiber core radius (default: 1.0)
            lambda_0: Wavelength (default: 1.0)
        Returns:
            tuple: (E_final_x, E_final_y, R_z) - Complex field arrays and grid radius
    """

    def analytical_hankel_core(l, u, a, k):
        """
        Analytical Hankel Transform of the Core field (J_l) from 0 to a.
        """
        h = u / a

        # Denominator: u^2 - k^2
        # Handle singularity at k = u/a with a small epsilon
        denom = h**2 - k**2
        denom[np.abs(denom) < 1e-12] = 1e-12

        # Formula: (a / (h^2-k^2)) * [ u*J_{l+1}(ha)*J_l(ka) - k*J_l(ha)*J_{l+1}(ka) ]
        term = (a / denom) * (
            h * jv(l + 1, u) * jv(l, k * a) - k * jv(l, u) * jv(l + 1, k * a)
        )
        return term

    def analytical_hankel_cladding(l, w, a, k):
        """
        Analytical Hankel Transform of the Cladding field (K_l) from a to infinity.

        """
        q = w / a

        # Denominator: q^2 + k^2
        denom = q**2 + k**2

        # Formula: (a / (w^2+k^2)) * [ w/a * J_l(ka) * K_{l+1}(w) - k * J_{l+1}(ka) * K_l(w) ]
        term = (a / denom) * (
            q * jv(l, k * a) * kn(l + 1, w) - k * jv(l + 1, k * a) * kn(l, w)
        )
        return term

    def get_normalization_factor(l, u, w, a):
        """
        Computes the normalization constant N such that Integral(|E|^2 dA) = 1
        for the spatial mode profile R(r).
        """
        # Core Integral: Int(J_l^2(ur/a) r dr) from 0 to a
        int_core = (a**2 / 2) * (jv(l, u) ** 2 - jv(l - 1, u) * jv(l + 1, u))

        # Cladding Integral: Int(K_l^2(wr/a) r dr) from a to inf
        int_clad = (a**2 / 2) * (kn(l - 1, w) * kn(l + 1, w) - kn(l, w) ** 2)

        # Continuity factor B at interface: J_l(u) / K_l(w)
        B = jv(l, u) / kn(l, w)

        # Total Power = 2*pi * (Core_Int + B^2 * Clad_Int)
        total_norm_sq = 2 * np.pi * (int_core + B**2 * int_clad)

        return np.sqrt(total_norm_sq)

    # Coordinates in position space
    R_z = (NA * z + radius) * Rz_factor
    R_z = max(R_origin, R_z)

    # --- GRID CALCULATION ---

    Nx = int(np.ceil((2 * R_z) / dx))

    x = np.linspace(-R_z, R_z, Nx)
    y = np.linspace(-R_z, R_z, Nx)
    R = np.hypot(x[None, :], y[:, None])
    PHI = np.arctan2(y[:, None], x[None, :])

    # Coordinate in k_space
    # Since we are not using fft we can use as many point as we want
    k0 = 2 * np.pi / lambda_0
    k_max = max(k0 * k_max_factor, 10 / radius)

    # * Calculating N_k to have {min_point_per_period} point per period for k_max
    # Asymptotic J_l(ax) ~ cos(ax + φ) -> λ=2π/a
    # Max period when calculating Hankel transform
    #   R_max = R_z * √2  (accounting for the corners)
    #   λ_min = 2π/R_max
    #   Δx =  λ_min/min_point_per_period = 2π/R_max/min_point_per_period
    #   N_k = (k_max / Δx)

    N_k = int(np.ceil(k_max / (2 * np.pi / R_z / np.sqrt(2) / min_point_per_period)))
    k_grid = np.linspace(1e-5, k_max, N_k)

    # --- 3. Pre-cooked propagator (1D) ---
    # H(k) = exp(i * z * sqrt(k0^2 - k^2))
    kz_sq = (k0**2 - k_grid**2).astype(complex)
    kz = np.sqrt(kz_sq)
    propagator = np.exp(1j * kz * z)

    if return_z_gradient:
        dz_factor = 1j * kz

    # --- 4. Accumulate Field and Derivative---
    E_final_x = np.zeros_like(R, dtype=complex)
    E_final_y = np.zeros_like(R, dtype=complex)

    dEx_dz = np.zeros_like(R, dtype=complex) if return_z_gradient else None
    dEy_dz = np.zeros_like(R, dtype=complex) if return_z_gradient else None

    # Pre-compute a 1D radial axis for interpolation (speed optimization)
    r_1d = np.linspace(0, R_z * np.sqrt(2), Nx)

    for mode in guided_modes:
        if mode is None:
            continue

        l = mode["l"]
        m = mode["m"]
        u = mode["u"]
        w = np.sqrt(fiber_V**2 - u**2)

        coeffs = df_coeff.loc[l, m]
        if isinstance(coeffs, pd.DataFrame):
            coeffs = coeffs.iloc[0]

        B = jv(l, u) / kn(l, w)
        norm_factor = get_normalization_factor(l, u, w, radius)

        # * Analytically computed Hankel transform in the core and in the clad
        H_core = analytical_hankel_core(l, u, radius, k_grid)
        H_clad = analytical_hankel_cladding(l, w, radius, k_grid)

        # Total Hankel transform F_k
        H_k = H_core + B * H_clad

        H_k /= norm_factor

        # * Application of the propagator
        H_k_prop = H_k * propagator

        # * Numerical inverse Hankel function
        # f(r, z) = Integral [ F(k) * J_l(kr) * k dk ]

        # Compute the integrand and integrate through simpson
        bessel_term = jv(l, k_grid[None, :] * r_1d[:, None])
        integrand = H_k_prop[None, :] * bessel_term * k_grid[None, :]
        f_r_prop = simpson(integrand, x=k_grid, axis=1)

        # Interpolation
        field_envelope = np.interp(R, r_1d, f_r_prop)

        # Integrate the gradient if required
        if return_z_gradient:
            integrand_dz = integrand * dz_factor[None, :]
            f_r_prop_dz = simpson(integrand_dz, x=k_grid, axis=1)
            field_envelope_dz = np.interp(R, r_1d, f_r_prop_dz)

        # --- Reconstruct Angular Dependence & Polarization ---
        ang_p = np.exp(1j * l * PHI)
        ang_m = np.exp(-1j * l * PHI)

        # Add contributions to X and Y fields
        E_final_x += field_envelope * (
            coeffs["x_p_phi"] * ang_p + coeffs["x_m_phi"] * ang_m
        )
        E_final_y += field_envelope * (
            coeffs["y_p_phi"] * ang_p + coeffs["y_m_phi"] * ang_m
        )

        if return_z_gradient:
            dEx_dz += field_envelope_dz * (
                coeffs["x_p_phi"] * ang_p + coeffs["x_m_phi"] * ang_m
            )
            dEy_dz += field_envelope_dz * (
                coeffs["y_p_phi"] * ang_p + coeffs["y_m_phi"] * ang_m
            )

    if return_z_gradient:
        return E_final_x, E_final_y, dEx_dz, dEy_dz, R_z

    return E_final_x, E_final_y, R_z
