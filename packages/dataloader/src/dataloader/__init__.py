import logging
from pathlib import Path
from time import time

import pandas as pd

from dataloader.loaders import registry
from dataloader.util import get_data_dir

force_canonicalize = False
force_download = False


logging.basicConfig()
logger = logging.getLogger("dataloader")


def load_dataset(data_set_name: str):
    canon_dir = get_data_dir() / "canon"
    canon_dir.mkdir(parents=True, exist_ok=True)
    canon_pth = (canon_dir / data_set_name).with_suffix(".csv")
    if canon_pth.exists() and not (force_canonicalize or force_download):
        logger.info(
            "Canonicalized data set already exists, skipping download and canonicalization."
        )
        return pd.read_csv(canon_pth)

    df = registry._run_loader(data_set_name, force_download)
    df = _canonicalize(df, canon_pth)
    return df


# TODO: Expose drop dup and norm id params to public API somehow
def _canonicalize(
    df: pd.DataFrame,
    canon_pth: Path,
    drop_duplicates=True,
    normalize_identifiers=True,
    normalize_timestamps=True,
) -> pd.DataFrame:
    start_time = time()
    logger.info("Canonicalizing raw data...")

    if drop_duplicates:
        df = _drop_duplicates(df)
    if normalize_identifiers:
        df = _normalize_identifiers(df)
    if normalize_timestamps:
        df = _normalize_timestamps(df)

    stop_time = time()
    logger.info(f"Canonicalized raw data in {(stop_time - start_time):.4f}s.")
    logger.info(f"Saving to {canon_pth}...")
    df.to_csv(canon_pth, index=False)

    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping duplicate interactions...")
    logger.info(f"Number of interactions before: {len(df)}")
    df.drop_duplicates(subset=["user", "item"], keep="last", inplace=True)
    logger.info(f"Number of interactions after: {len(df)}")
    return df


def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Normalizing identifiers...")
    for col in ["user", "item"]:
        unique_ids = {key: value for value, key in enumerate(df[col].unique())}
        df[col] = df[col].map(unique_ids)
    logger.info("Done.")
    return df


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        logger.info("Normalizing timestamps...")
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            ts = (
                pd.to_datetime(ts, unit="s", errors="coerce", utc=True).astype("int64")
                // 10**9
            )
        else:
            ts = pd.to_datetime(ts, errors="coerce", utc=True).astype("int64") // 10**9
        df["timestamp"] = ts
        logger.info("Done.")

    return df
