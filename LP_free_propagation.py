import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from log_init import init_logging
from io_utils import load_config, AppConfig
from source.propagation import free_propagation_asm_hankel, verify_resolution


def _format_table(title, headers, rows):
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * width for width in widths)
    lines = [title, fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)


def build_heatmaps(line_data, x_plot, z_plot, axis_unit, out_dir):
    xz_rows = []
    yz_rows = []

    for z_val in z_plot:
        x_line, line_x, line_y = line_data[z_val]
        xz_rows.append(np.interp(x_plot, x_line, line_x, left=0.0, right=0.0))
        yz_rows.append(np.interp(x_plot, x_line, line_y, left=0.0, right=0.0))

    xz_map = np.asarray(xz_rows)
    yz_map = np.asarray(yz_rows)

    extent = [x_plot[0], x_plot[-1], z_plot[0], z_plot[-1]]

    plt.figure()
    plt.imshow(xz_map, extent=extent, origin="lower", aspect="auto")
    plt.xlabel(rf"$\mathnormal{{x}}$ [$\mathnormal{{{axis_unit}}}$]")
    plt.ylabel(rf"$\mathnormal{{z}}$ [$\mathnormal{{{axis_unit}}}$]")
    plt.title(r"$\mathnormal{Intensity}$ projection ($\mathnormal{x}$-$\mathnormal{z}$)")
    plt.colorbar(label=r"$\mathnormal{Intensity}$")
    plt.tight_layout()
    plt.savefig(out_dir / "intensity_xz.pdf")
    plt.close()

    plt.figure()
    plt.imshow(yz_map, extent=extent, origin="lower", aspect="auto")
    plt.xlabel(rf"$\mathnormal{{y}}$ [$\mathnormal{{{axis_unit}}}$]")
    plt.ylabel(rf"$\mathnormal{{z}}$ [$\mathnormal{{{axis_unit}}}$]")
    plt.title(r"$\mathnormal{Intensity}$ projection ($\mathnormal{y}$-$\mathnormal{z}$)")
    plt.colorbar(label=r"$\mathnormal{Intensity}$")
    plt.tight_layout()
    plt.savefig(out_dir / "intensity_yz.pdf")
    plt.close()


def run_free_propagation(
    config: AppConfig,
    guided_modes: Optional[object] = None,
    df_coeff_fib_prop: Optional[pd.DataFrame] = None,
    modes_path: str = "guided_modes.npy",
    coeff_path: str = "propagate_field_coeff.csv",
    output_dir: str = "output",
    media_dir: str = "media",
    save_h5: bool = True,
    h5_path: Optional[str] = None,
) -> None:
    fiber = config.fiber
    beam = config.beam
    domain = config.domain
    grid = config.grid.propagation
    runtime = config.runtime
    propagation = config.propagation
    visualization = config.visualization

    supervision_mode = runtime.supervision_mode
    n_threads = runtime.n_threads
    recompute = runtime.recompute

    fiber_v = fiber.v_number
    dist_from_fiber = propagation.dist_from_fiber_array()
    rz_factor = grid.rz_factor
    lambda_0 = beam.wavelength
    fiber_radius_m = fiber.radius_m
    axis_size = domain.axis_size
    dx_propagated_field = grid.dx
    oversampling_x = grid.oversampling_x
    oversampling_z = grid.oversampling_z
    k_max_factor = grid.k_max_factor
    min_point_per_period = grid.min_point_per_period
    r_origin_factor = domain.r_origin_factor

    cmap = plt.get_cmap(visualization.cmap, visualization.cmap_levels)

    logger.info("Starting free-space propagation")
    logger.debug(
        "Params: SUPERVISION_MODE={}, N_THREADS={}, RECOMPUTE={}, FIBER_V={}, "
        "DIST_FROM_FIBER(min,max,count)=({}, {}, {}), RZ_FACTOR={}, LAMBDA={}",
        supervision_mode,
        n_threads,
        recompute,
        fiber_v,
        float(dist_from_fiber.min()),
        float(dist_from_fiber.max()),
        len(dist_from_fiber),
        rz_factor,
        lambda_0,
    )
    logger.debug(
        "Grid params: AXIS_SIZE={}, DX_PROPAGATED_FIELD={}, OVERSAMPLING_X={}, OVERSAMPLING_Z={}",
        axis_size,
        dx_propagated_field,
        oversampling_x,
        oversampling_z,
    )

    output_folder = Path(output_dir)
    output_folder.mkdir(exist_ok=True)
    logger.debug("Output folder set to {}", output_folder)

    radius = fiber.radius_units
    na = lambda_0 * fiber_v / (2 * np.pi * radius)
    axis_ext = axis_size * radius

    if guided_modes is None:
        guided_modes = np.load(modes_path, allow_pickle=True)
        logger.info("Loaded guided modes: {}", len(guided_modes))

    if df_coeff_fib_prop is None:
        df_coeff_fib_prop = pd.read_csv(
            coeff_path,
            dtype={0: float, 1: float, 2: float},
            converters={3: complex, 4: complex, 5: complex, 6: complex},
        )
        df_coeff_fib_prop.set_index(["l", "m"], inplace=True)
        logger.info("Loaded propagated coefficients: {}", df_coeff_fib_prop.shape)

    z_min = float(dist_from_fiber.min())
    z_max = float(dist_from_fiber.max())
    r_origin = r_origin_factor * axis_ext

    def calc_r_z(z_units):
        return max(r_origin, (na * z_units + radius) * rz_factor)

    r_z_min = calc_r_z(z_min)
    r_z_max = calc_r_z(z_max)
    r_z_values = np.array([calc_r_z(float(z_val)) for z_val in dist_from_fiber], dtype=float)

    if fiber_radius_m is None:
        unit = "r_F"
        scale = 1.0
        domain_title = "Domain (r_F units)"
    else:
        unit = "m"
        scale = fiber_radius_m
        domain_title = "Domain (SI units)"

    domain_rows = [
        ("z", f"{z_min * scale:.3e}", f"{z_max * scale:.3e}", unit),
        ("x", f"{-r_z_max * scale:.3e}", f"{r_z_max * scale:.3e}", unit),
        ("y", f"{-r_z_max * scale:.3e}", f"{r_z_max * scale:.3e}", unit),
        ("R_z", f"{r_z_min * scale:.3e}", f"{r_z_max * scale:.3e}", unit),
        ("dx", f"{dx_propagated_field * scale:.3e}", f"{dx_propagated_field * scale:.3e}", unit),
        ("lambda", f"{lambda_0 * scale:.3e}", f"{lambda_0 * scale:.3e}", unit),
    ]

    logger.info(
        "\n{}",
        _format_table(domain_title, ["Quantity", "Min", "Max", "Unit"], domain_rows),
    )

    k0 = 2 * np.pi / lambda_0
    k_max = max(k0 * k_max_factor, 10 / radius)
    nx_min = int(np.ceil((2 * r_z_min) / dx_propagated_field))
    nx_max = int(np.ceil((2 * r_z_max) / dx_propagated_field))
    nk_min = int(
        np.ceil(k_max / (2 * np.pi / r_z_min / np.sqrt(2) / min_point_per_period))
    )
    nk_max = int(
        np.ceil(k_max / (2 * np.pi / r_z_max / np.sqrt(2) / min_point_per_period))
    )

    grid_rows = [
        ("Nx (per axis)", f"{nx_min}", f"{nx_max}", "points"),
        ("Nx^2 (grid)", f"{nx_min * nx_min}", f"{nx_max * nx_max}", "points"),
        ("N_k", f"{nk_min}", f"{nk_max}", "samples"),
    ]
    logger.info(
        "\n{}",
        _format_table("Grid sizes (expected min/max)", ["Quantity", "Min", "Max", "Unit"], grid_rows),
    )

    h5_file = None
    fields_group = None
    if save_h5:
        try:
            import h5py
        except ImportError:
            logger.error("h5py is required to save HDF5 output.")
            raise

        h5_path = Path(h5_path) if h5_path is not None else output_folder / "field_data.h5"
        h5_path.parent.mkdir(exist_ok=True)
        h5_file = h5py.File(h5_path, "w")
        h5_file.attrs["axis_unit"] = unit
        h5_file.attrs["axis_scale"] = scale
        h5_file.attrs["fiber_radius_units"] = radius
        if fiber_radius_m is not None:
            h5_file.attrs["fiber_radius_m"] = fiber_radius_m
        h5_file.attrs["lambda_0"] = lambda_0
        h5_file.attrs["na"] = na
        h5_file.attrs["axis_size_factor"] = axis_size
        h5_file.attrs["r_origin_factor"] = r_origin_factor
        h5_file.attrs["rz_factor"] = rz_factor
        h5_file.attrs["dx"] = dx_propagated_field * scale
        h5_file.attrs["k_max_factor"] = k_max_factor
        h5_file.attrs["min_point_per_period"] = min_point_per_period

        domain_group = h5_file.create_group("domain")
        domain_group.create_dataset("z", data=dist_from_fiber * scale)
        domain_group.create_dataset("r_z", data=r_z_values * scale)
        domain_group.attrs["unit"] = unit
        domain_group.attrs["z_min"] = z_min * scale
        domain_group.attrs["z_max"] = z_max * scale
        domain_group.attrs["r_z_min"] = r_z_min * scale
        domain_group.attrs["r_z_max"] = r_z_max * scale
        domain_group.attrs["nx_min"] = nx_min
        domain_group.attrs["nx_max"] = nx_max
        domain_group.attrs["nk_min"] = nk_min
        domain_group.attrs["nk_max"] = nk_max

        init_group = h5_file.create_group("initial_data")
        init_group.attrs["radius_units"] = radius
        if fiber_radius_m is not None:
            init_group.attrs["radius_m"] = fiber_radius_m

        if guided_modes is not None:
            modes_rows = [
                (int(mode["l"]), int(mode["m"]), float(mode.get("u", np.nan)))
                for mode in guided_modes
                if mode is not None
            ]
            if modes_rows:
                modes_dtype = np.dtype([("l", "i4"), ("m", "i4"), ("u", "f8")])
                init_group.create_dataset("modes", data=np.array(modes_rows, dtype=modes_dtype))

        if df_coeff_fib_prop is not None and not df_coeff_fib_prop.empty:
            coeff_group = init_group.create_group("coefficients")
            coeff_index = df_coeff_fib_prop.index
            coeff_group.create_dataset("l", data=coeff_index.get_level_values(0).to_numpy())
            coeff_group.create_dataset("m", data=coeff_index.get_level_values(1).to_numpy())
            for column in ["u", "x_p_phi", "y_p_phi", "x_m_phi", "y_m_phi"]:
                coeff_group.create_dataset(column, data=df_coeff_fib_prop[column].to_numpy())

        fields_group = h5_file.create_group("fields")

    def _write_h5_field(z_index, z_val, intensity, prop_axis_ext, x_axis, e_x, e_y):
        if fields_group is None:
            return
        group = fields_group.create_group(f"z_{z_index:05d}")
        group.attrs["z"] = z_val * scale
        group.attrs["axis_ext"] = prop_axis_ext * scale
        group.create_dataset("x", data=x_axis * scale)
        group.create_dataset("y", data=x_axis * scale)
        if e_x is not None:
            group.create_dataset("E_x", data=e_x, compression="gzip", compression_opts=4)
        if e_y is not None:
            group.create_dataset("E_y", data=e_y, compression="gzip", compression_opts=4)
        group.create_dataset("intensity", data=intensity, compression="gzip", compression_opts=4)

    def process_propagation(z_dist):
        file_path = output_folder / f"intensity_gradient_z{z_dist}.npz"
        if not recompute and file_path.exists():
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    if "intensity" in data:
                        if save_h5 and ("E_x" not in data or "E_y" not in data):
                            logger.debug(
                                "Cached file missing field data at z={}, recomputing",
                                z_dist,
                            )
                        else:
                            intensity = data["intensity"]
                            prop_axis_ext = float(data["axis_ext"])
                            logger.debug("Loaded cached propagation at z={}", z_dist)
                            nx = intensity.shape[0]
                            x_axis = np.linspace(-prop_axis_ext, prop_axis_ext, nx)
                            mid = nx // 2
                            line_x = intensity[mid, :]
                            line_y = intensity[:, mid]
                            e_x = data["E_x"] if save_h5 else None
                            e_y = data["E_y"] if save_h5 else None
                            return intensity, prop_axis_ext, x_axis, line_x, line_y, e_x, e_y
                    logger.debug("Cached file missing intensity at z={}, recomputing", z_dist)
            except Exception as exc:
                logger.debug("Failed to load cached data at z={} ({}); recomputing", z_dist, exc)

        logger.debug("Propagating field at z={}", z_dist)
        E_propagated_x, E_propagated_y, dEx_dz, dEy_dz, prop_axis_ext = free_propagation_asm_hankel(
            guided_modes,
            df_coeff_fib_prop,
            z_dist,
            na,
            rz_factor,
            dx_propagated_field,
            k_max_factor,
            fiber_v,
            r_origin_factor * axis_ext,
            min_point_per_period=min_point_per_period,
            radius=radius,
            lambda_0=lambda_0,
            return_z_gradient=True,
        )
        logger.debug(
            "Field computed at z={} with shape={} and axis_ext={}",
            z_dist,
            E_propagated_x.shape,
            prop_axis_ext,
        )

        intensity = np.abs(E_propagated_x) ** 2 + np.abs(E_propagated_y) ** 2
        nx = E_propagated_x.shape[0]
        dx_local = (2 * prop_axis_ext) / (nx - 1)
        grad_I_y, grad_I_x = np.gradient(intensity, dx_local)
        grad_I_z = 2 * np.real(
            E_propagated_x * np.conj(dEx_dz) + E_propagated_y * np.conj(dEy_dz)
        )

        if not supervision_mode:
            save_payload = {
                "grad_I": (grad_I_x, grad_I_y, grad_I_z),
                "intensity": intensity,
                "dist_from_fiber": z_dist,
                "axis_ext": prop_axis_ext,
                "fiber_radius": radius,
                "lambda_0": lambda_0,
            }
            if save_h5:
                save_payload["E_x"] = E_propagated_x
                save_payload["E_y"] = E_propagated_y
            np.savez(file_path, **save_payload)
            logger.debug("Saved output for z={}", z_dist)

        x_axis = np.linspace(-prop_axis_ext, prop_axis_ext, nx)
        mid = nx // 2
        line_x = intensity[mid, :]
        line_y = intensity[:, mid]

        return intensity, prop_axis_ext, x_axis, line_x, line_y, E_propagated_x, E_propagated_y

    try:
        if supervision_mode:
            logger.info("Running in supervision mode")
            logger.info("*" * 50)
            logger.info(
                "Characteristic length on the x,y directions: Lx = {:.3f}\t currently using: δx = {:.3f}",
                lambda_0 / na,
                dx_propagated_field,
            )
            logger.info(
                "Characteristic length on the z direction   : Lz = {:.3f}\t currently using: δz = {:.3f}",
                2 * lambda_0 / (na**2),
                dist_from_fiber[1] - dist_from_fiber[0],
            )
            logger.info("*" * 50)

            line_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for z_index, z_dist in enumerate(tqdm(dist_from_fiber, desc="Propagating field")):
                intensity, prop_axis_ext, x_axis, line_x, line_y, e_x, e_y = process_propagation(
                    z_dist
                )
                line_data[z_dist] = (x_axis, line_x, line_y)
                _write_h5_field(z_index, z_dist, intensity, prop_axis_ext, x_axis, e_x, e_y)

                if z_dist == dist_from_fiber[0]:
                    plt.figure(figsize=(10, 8))
                    im = plt.imshow(
                        intensity,
                        extent=[-prop_axis_ext, prop_axis_ext, -prop_axis_ext, prop_axis_ext],
                        cmap=cmap,
                        origin="lower",
                    )
                    plt.colorbar(im, label="Intensity")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    title = plt.title(f"Propagated Field Intensity at z={z_dist}")
                    plt.tight_layout()
                    plt.ion()
                    plt.show()
                else:
                    im.set_data(intensity)
                    im.set_extent([-prop_axis_ext, prop_axis_ext, -prop_axis_ext, prop_axis_ext])
                    title.set_text(f"Propagated Field Intensity at z={z_dist}")
                    im.autoscale()
                    plt.draw()
                    plt.pause(0.05)

            media_folder = Path(media_dir)
            media_folder.mkdir(exist_ok=True)

            z_sorted = sorted(line_data.keys())
            if fiber_radius_m is None:
                axis_unit = "r_F"
                scale = 1.0
            else:
                axis_unit = "m"
                scale = fiber_radius_m

            x_plot = np.linspace(-r_z_max * scale, r_z_max * scale, nx_max)
            z_plot = np.array(z_sorted, dtype=float) * scale
            scaled_line_data = {
                z * scale: (x * scale, lx, ly) for z, (x, lx, ly) in line_data.items()
            }
            build_heatmaps(scaled_line_data, x_plot, z_plot, axis_unit, media_folder)
            logger.info(
                "Saved heatmaps: {} and {}",
                media_folder / "intensity_xz.pdf",
                media_folder / "intensity_yz.pdf",
            )
        else:
            logger.info("Running in computation mode")

            try:
                dz = dist_from_fiber[1] - dist_from_fiber[0]
                verify_resolution(
                    dx_propagated_field,
                    dz,
                    lambda_0,
                    na,
                    oversampling_x,
                    oversampling_z,
                )
                logger.info("Resolution check passed: dx={}, dz={}", dx_propagated_field, dz)
            except Exception as exc:
                logger.error("Resolution check failed: {}", exc)
                sys.exit(1)

            logger.info("Launching propagation with {} worker threads", n_threads)
            line_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = {
                    executor.submit(process_propagation, z_dist): (idx, z_dist)
                    for idx, z_dist in enumerate(dist_from_fiber)
                }

                for future in tqdm(
                    as_completed(futures), total=len(dist_from_fiber), desc="Propagating field"
                ):
                    z_index, z_val = futures[future]
                    intensity, prop_axis_ext, x_axis, line_x, line_y, e_x, e_y = future.result()
                    line_data[z_val] = (x_axis, line_x, line_y)
                    _write_h5_field(z_index, z_val, intensity, prop_axis_ext, x_axis, e_x, e_y)

            media_folder = Path(media_dir)
            media_folder.mkdir(exist_ok=True)

            z_sorted = sorted(line_data.keys())
            if fiber_radius_m is None:
                axis_unit = "r_F"
                scale = 1.0
            else:
                axis_unit = "m"
                scale = fiber_radius_m

            x_plot = np.linspace(-r_z_max * scale, r_z_max * scale, nx_max)
            z_plot = np.array(z_sorted, dtype=float) * scale
            scaled_line_data = {
                z * scale: (x * scale, lx, ly) for z, (x, lx, ly) in line_data.items()
            }
            build_heatmaps(scaled_line_data, x_plot, z_plot, axis_unit, media_folder)
            logger.info(
                "Saved heatmaps: {} and {}",
                media_folder / "intensity_xz.pdf",
                media_folder / "intensity_yz.pdf",
            )
    finally:
        if h5_file is not None:
            h5_file.close()
            logger.info("Saved HDF5 field data to {}", h5_path)


def main(config_path: Optional[str] = None) -> None:
    init_logging()
    path = config_path or (sys.argv[1] if len(sys.argv) > 1 else "input/config.toml")
    config = load_config(path)
    run_free_propagation(config)


if __name__ == "__main__":
    main()
