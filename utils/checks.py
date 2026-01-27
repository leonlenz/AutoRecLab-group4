import shutil
import sys

def require_executable(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found: {name}")