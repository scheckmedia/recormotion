import logging
from pathlib import Path

from recormotion.config import Configuration


def setup_logger():
    """Helper function to setup file and stream logging for the applicationk"""
    cfg = Configuration().config.logging
    level = logging.getLevelName(cfg.level.upper())

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    stdout_handler.setFormatter(formatter)

    handlers = [stdout_handler]

    if cfg.file is not None:
        Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(cfg.file, "w")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        handlers=handlers,
    )
