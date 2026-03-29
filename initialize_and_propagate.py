import sys
from typing import Optional

from loguru import logger

from log_init import init_logging
from io_utils import load_config
from LP_guided_field_generator import generate_guided_field
from LP_free_propagation import run_free_propagation


def main(config_path: Optional[str] = None) -> None:
    init_logging()
    path = config_path or (sys.argv[1] if len(sys.argv) > 1 else "input/config_balanced.toml")
    config = load_config(path)

    logger.info("Running generator + propagation pipeline")
    guided_modes, _, df_coeff_fib_prop = generate_guided_field(config)
    run_free_propagation(config, guided_modes=guided_modes, df_coeff_fib_prop=df_coeff_fib_prop)


if __name__ == "__main__":
    main()

