import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger

from log_init import init_logging
from io_utils import load_config, AppConfig
from source.LP_projection_functions import (
    get_guided_modes,
    get_LP_modes_projection_coefficients,
    get_tilted_beam_from_incidence,
)
from source.propagation import fiber_propagation


def generate_guided_field(
    config: AppConfig,
    save_outputs: bool = True,
    coeff_path: str = "propagate_field_coeff.csv",
    modes_path: str = "guided_modes.npy",
) -> Tuple[List[dict], pd.DataFrame, pd.DataFrame]:
    fiber = config.fiber
    beam = config.beam
    domain = config.domain
    grid = config.grid.generator
    generator = config.generator
    custom_coeffs = config.custom_coeffs

    fiber_v = fiber.v_number
    modes_to_test = generator.modes_to_test
    fiber_n1 = fiber.n1
    fiber_length = fiber.length
    dist_from_fiber = generator.dist_from_fiber
    use_custom_coeffs = generator.use_custom_coeffs

    radius = fiber.radius_units
    if use_custom_coeffs:
        modes_to_test = [(item.l, item.m) for item in custom_coeffs]
        if not modes_to_test:
            raise ValueError("USE_CUSTOM_COEFFS is True, but custom_coeffs is empty.")

    logger.info("Starting guided field generation")
    logger.debug(
        "Params: FIBER_V={}, MODES_TO_TEST={}, FIBER_N1={}, FIBER_LENGTH={}, "
        "DIST_FROM_FIBER={}, USE_CUSTOM_COEFFS={}",
        fiber_v,
        modes_to_test,
        fiber_n1,
        fiber_length,
        dist_from_fiber,
        use_custom_coeffs,
    )
    logger.debug(
        "Beam params: LAMBDA={}, DIST_TO_WAIST={}, W0_X={}, W0_Y={}, X0={}, Y0={}, "
        "ROLL_ANGLE={}, PITCH_ANGLE={}, YAW_ANGLE={}, POLARIZATION_ANGLE={}",
        beam.wavelength,
        beam.dist_to_waist,
        beam.w0_x,
        beam.w0_y,
        beam.x0,
        beam.y0,
        beam.roll_angle,
        beam.pitch_angle,
        beam.yaw_angle,
        beam.polarization_angle,
    )
    logger.debug("Grid params: AXIS_SIZE={}, GRID_SIZE={}", domain.axis_size, grid.grid_size)

    na = beam.wavelength * fiber_v / (2 * np.pi * radius)
    logger.debug("Computed NA={}", na)

    axis_ext = domain.axis_size * radius
    x = np.linspace(-axis_ext, axis_ext, grid.grid_size)
    y = np.linspace(-axis_ext, axis_ext, grid.grid_size)
    X, Y = np.meshgrid(x, y)
    logger.debug("Grid built with axis_ext={}, shape={}", axis_ext, X.shape)

    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    dA = (axis_ext * 2 / grid.grid_size) ** 2
    logger.debug("Computed dA={}", dA)

    E_input = None
    if not use_custom_coeffs:
        E_input = get_tilted_beam_from_incidence(
            X,
            Y,
            z_plane=0,
            x_incidence=beam.x0,
            y_incidence=beam.y0,
            dist_to_waist=beam.dist_to_waist,
            euler_alpha=beam.roll_angle,
            euler_beta=beam.pitch_angle,
            euler_gamma=beam.yaw_angle,
            dA=dA,
            w0_x=beam.w0_x,
            w0_y=beam.w0_y,
            wavelength=beam.wavelength,
            polarization_angle=beam.polarization_angle,
        )
        logger.debug(
            "Input field computed with shapes: Ex={}, Ey={}",
            E_input[0].shape,
            E_input[1].shape,
        )
    else:
        logger.info("Using custom LP mode coefficients; skipping input field projection.")

    guided_modes = []
    coefficients = []
    for l, m in modes_to_test:
        logger.debug("Computing guided mode LP{}{}", l, m)
        mode = get_guided_modes(l, m, fiber_v, radius, R, PHI, dA)

        if mode is None:
            logger.debug("Mode LP{}{} not guided or not converged", l, m)
            continue

        guided_modes.append(mode)
        logger.debug("Mode LP{}{} found with u={}", l, m, mode.get("u"))

        if use_custom_coeffs:
            spec = next((item for item in custom_coeffs if item.l == l and item.m == m), None)
            if spec is None:
                logger.debug("No custom coefficient provided for LP{}{}", l, m)
                continue
            coefficients_res = {
                "l": l,
                "m": m,
                "u": mode.get("u"),
                "x_p_phi": spec.x_p_phi,
                "y_p_phi": spec.y_p_phi,
                "x_m_phi": spec.x_m_phi,
                "y_m_phi": spec.y_m_phi,
            }
        else:
            coefficients_res = get_LP_modes_projection_coefficients(E_input, mode, dA)

        coefficients.append(coefficients_res)

    if not coefficients:
        raise ValueError("No valid coefficients were generated; check mode selection and inputs.")

    df_coeff = pd.DataFrame(coefficients)
    df_coeff.set_index(["l", "m"], inplace=True)
    logger.info("Computed projection coefficients for {} modes", len(df_coeff))

    logger.info("Squared modulus of coefficients:\n{}", "*" * 70)
    logger.info(
        "\n{}",
        (df_coeff.iloc[:, 1:]).to_string(
            float_format=lambda x: f"{x:.2f}", justify="center", col_space=10
        ),
    )
    logger.info("{}", "*" * 70)

    df_coeff_fib_prop = fiber_propagation(
        df_coeff,
        n1=fiber_n1,
        a=radius,
        lam=beam.wavelength,
        z_fiber=fiber_length,
    )
    logger.info("Fiber propagation completed")

    if save_outputs:
        df_coeff_fib_prop.to_csv(coeff_path, index="True")
        np.save(modes_path, guided_modes)
        logger.info("Saved outputs to {} and {}", coeff_path, modes_path)

    return guided_modes, df_coeff, df_coeff_fib_prop


def main(config_path: Optional[str] = None) -> None:
    init_logging()
    path = config_path or (sys.argv[1] if len(sys.argv) > 1 else "input/config.toml")
    config = load_config(path)
    generate_guided_field(config)


if __name__ == "__main__":
    main()
