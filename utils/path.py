from pathlib import Path


def mkdir(*args, **kwargs) -> Path:
    pth = Path(*args, *kwargs)
    pth.mkdir(parents=True, exist_ok=True)
    return pth
