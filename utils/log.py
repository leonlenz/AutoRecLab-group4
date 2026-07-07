import logging
from pathlib import Path
import sys

from rich.console import Console
from rich.logging import RichHandler

_ROOT_LOGGER = logging.getLogger("isgsa")
# Set logger to debug and then filter more using handlers below
_ROOT_LOGGER.setLevel(logging.DEBUG)

try:
    utf8_stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
except Exception:
    utf8_stdout = sys.stdout

try:
    utf8_stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", closefd=False)
except Exception:
    utf8_stderr = sys.stderr

console_handler = RichHandler(console=Console(file=utf8_stderr))
console_handler.setLevel(logging.INFO)
_ROOT_LOGGER.addHandler(console_handler)


def set_log_level(level: str):
    level = level.upper()
    if level in logging._nameToLevel:
        console_handler.setLevel(level)
        _ROOT_LOGGER.debug("LOG LEVEL IS SET TO DEBUG")
    else:
        raise ValueError(f"Unknown log level: {level}")


def attach_file_handler(file_log_dir: Path, level=logging.DEBUG):
    # Remove any previously attached file handlers so each run logs to its own
    # debug.log instead of accumulating handlers across multiple runs.
    for handler in list(_ROOT_LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            _ROOT_LOGGER.removeHandler(handler)
            handler.close()

    file_log_dir.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(file_log_dir / "debug.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="[%Y/%m/%d %H:%M:%S]",
        )
    )

    _ROOT_LOGGER.addHandler(file_handler)
