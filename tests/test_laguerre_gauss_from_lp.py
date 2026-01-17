import os
import sys
import numpy as np

# Allow running as a script from repo root or tests/ without packaging.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.LP_projection_functions import get_guided_modes


def _build_grid(radius, axis_size=4.0, grid_size=256):
    axis_ext = axis_size * radius
    x = np.linspace(-axis_ext, axis_ext, grid_size)
    y = np.linspace(-axis_ext, axis_ext, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    dA = (2.0 * axis_ext / grid_size) ** 2
    return R, PHI, dA


def test_laguerre_gauss_from_two_lp_modes():
    # LP11 has two degenerate orientations (cos and sin); combine with +i to get a helical LG.
    l = 1
    m = 1
    V = 4.0  # Above LP11 cutoff (about 2.405)
    radius = 1.0

    R, PHI, dA = _build_grid(radius=radius)
    mode = get_guided_modes(l, m, V, radius, R, PHI, dA)
    assert mode is not None, "LP11 mode should be guided for V > 2.405."

    p_phi = mode["p_phi"]
    m_phi = mode["m_phi"]

    lp_cos = (p_phi + m_phi) / np.sqrt(2.0)
    lp_sin = (p_phi - m_phi) / (1j * np.sqrt(2.0))

    lg_from_lp = (lp_cos + 1j * lp_sin) / np.sqrt(2.0)

    # Align global phase before comparison.
    overlap = np.vdot(p_phi, lg_from_lp)
    phase = np.angle(overlap) if np.abs(overlap) > 0 else 0.0
    lg_from_lp *= np.exp(-1j * phase)

    max_diff = float(np.max(np.abs(lg_from_lp - p_phi)))
    max_amp = float(np.max(np.abs(p_phi)))

    print(f"max_diff={max_diff:.3e}, rel_diff={max_diff / max_amp:.3e}")

    rel_tol = 1e-8
    assert max_diff <= rel_tol * max_amp


if __name__ == "__main__":
    test_laguerre_gauss_from_two_lp_modes()
