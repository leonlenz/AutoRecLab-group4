import hashlib
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("dataloader")

_DATA_DIR = Path(os.environ.get("OMNIREC_DATA_PATH", Path.home() / ".omnirec/data"))


def get_data_dir() -> Path:
    _DATA_DIR.mkdir(exist_ok=True, parents=True)
    return _DATA_DIR


def calculate_checksum(file_pth: Path, chunk_size=1024 * 1024) -> str:
    hash = hashlib.sha256()
    with open(file_pth, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash.update(chunk)
        return hash.hexdigest()


def verify_checksum(file_pth: Path, checksum: str | None) -> bool:
    if not checksum:
        logger.warning("No checksum provided, skipping checksum verification...")
        return True
    else:
        logger.info("Verifying checksum...")
        res = calculate_checksum(file_pth) == checksum
        if res:
            logger.info("Checksum verified successfully!")
        else:
            logger.warning("Checksum verification failed!")

        return res


def is_valid_url(url) -> bool:
    parsed = urlparse(url)
    return all([parsed.scheme in ("http", "https"), parsed.netloc])
