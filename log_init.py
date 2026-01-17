import os
import sys
from pathlib import Path

from loguru import logger
import contextlib


def init_logging():
    # Figure out which script is being run directly
    script_name = Path(sys.argv[0]).stem or "interactive"
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build filenames dynamically
    log_file = log_dir / f"{script_name}.log"
    log_file_deep = log_dir / f"{script_name}_deep.log"

    # Ensure files exist to avoid rotation rename errors
    with contextlib.suppress(Exception):
        log_file.touch(exist_ok=True)
        log_file_deep.touch(exist_ok=True)

    # Remove default handler and add ours
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True, enqueue=True)
    # Disable rotation to avoid rename races across processes; ensure enqueue for mp safety
    logger.add(log_file, level="DEBUG", rotation=None, colorize=False, enqueue=True)
    logger.add(log_file_deep, level="TRACE", rotation=None, colorize=False, enqueue=True)

    logger.debug(f"Logger initialized for script: {script_name}")

    # Apply user matplotlib style globally if present
    style_path = Path.home() / ".config" / "matplotlib" / "matplotlibrc"
    if style_path.exists():
        with contextlib.suppress(Exception):
            import matplotlib

            matplotlib.rc_file(style_path)
            logger.debug(f"Loaded matplotlib rc from {style_path}")
