import os
import sys
import numpy as np

# Allow running as a script from repo root or tests/ without packaging.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.LP_projection_functions import _gaussian_electric_field_alligned


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def _make_grid(w0, wavelength, z_prop, domain_factor=6.0, points_per_waist=12, min_n=256):
    z_rayleigh = np.pi * w0**2 / wavelength
    w_z = w0 * np.sqrt(1.0 + (z_prop / z_rayleigh) ** 2)

    # Keep the computation window wide enough to capture the tails.
    half_width = domain_factor * w_z
    dx_target = w0 / points_per_waist
    n_target = int(np.ceil((2.0 * half_width) / dx_target))
    n = max(min_n, _next_pow2(n_target))

    dx = (2.0 * half_width) / n
    coords = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(coords, coords)
    return X, Y, dx, half_width, n, w_z


def _asm_propagate(field, dx, wavelength, z_prop):
    n = field.shape[0]
    k0 = 2.0 * np.pi / wavelength

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    ky = kx
    KX, KY = np.meshgrid(kx, ky)

    kz = np.sqrt((k0**2 - KX**2 - KY**2).astype(complex))
    propagator = np.exp(1j * kz * z_prop)

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    propagated = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * propagator)))

    return propagated


def test_gaussian_beam_propagation():
    wavelength = 1.0
    # Analytical Gaussian beam is paraxial; keep w0 >> wavelength and modest z.
    w0 = 50.0
    z_prop = 100.0
    polarization_angle = 0.35

    X, Y, dx, half_width, n, w_z = _make_grid(w0, wavelength, z_prop)
    dA = dx**2

    Ex0, Ey0 = _gaussian_electric_field_alligned(
        X,
        Y,
        z=0.0,
        dA=dA,
        w0_x=w0,
        w0_y=w0,
        wavelength=wavelength,
        polarization_angle=polarization_angle,
    )

    Ex_num = _asm_propagate(Ex0, dx, wavelength, z_prop)
    Ey_num = _asm_propagate(Ey0, dx, wavelength, z_prop)

    Ex_ana, Ey_ana = _gaussian_electric_field_alligned(
        X,
        Y,
        z=z_prop,
        dA=dA,
        w0_x=w0,
        w0_y=w0,
        wavelength=wavelength,
        polarization_angle=polarization_angle,
    )

    # Align global phase before comparison.
    overlap = np.vdot(Ex_ana, Ex_num) + np.vdot(Ey_ana, Ey_num)
    phase = np.angle(overlap) if np.abs(overlap) > 0 else 0.0
    phase_factor = np.exp(-1j * phase)
    Ex_num *= phase_factor
    Ey_num *= phase_factor

    max_diff = max(
        float(np.max(np.abs(Ex_num - Ex_ana))),
        float(np.max(np.abs(Ey_num - Ey_ana))),
    )
    max_amp = max(
        float(np.max(np.abs(Ex_ana))),
        float(np.max(np.abs(Ey_ana))),
    )

    edge_max = max(
        float(np.max(np.abs(Ex_ana[0, :]))),
        float(np.max(np.abs(Ex_ana[-1, :]))),
        float(np.max(np.abs(Ex_ana[:, 0]))),
        float(np.max(np.abs(Ex_ana[:, -1]))),
    )

    print(
        f"N={n}, dx={dx:.3e}, half_width={half_width:.3e}, w(z)={w_z:.3e}, "
        f"edge_ratio={edge_max / max_amp:.3e}"
    )
    print(f"max_diff={max_diff:.3e}, rel_diff={max_diff / max_amp:.3e}")

    rel_tol = 2e-2
    abs_tol = 1e-3
    allowed = max(abs_tol, rel_tol * max_amp)
    assert max_diff <= allowed, f"Max diff {max_diff:.3e} exceeds tolerance {allowed:.3e}"


if __name__ == "__main__":
    test_gaussian_beam_propagation()
