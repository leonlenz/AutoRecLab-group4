"""
Workspace context utilities.

Provides the LLM planner with awareness of what custom data files the user
placed in the workspace directory. Since the workspace can be in different
locations (Docker: /app/out/workspace, local: ./out/workspace),
this module dynamically discovers the correct paths.
"""
import os
from pathlib import Path
from typing import Optional

from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("workspace_context")


def find_project_root() -> Path:
    """
    Find the project root directory by looking for sentinel files
    (pyproject.toml, setup.py, .git).

    Returns:
        Path: The absolute path to the project root.
    """
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        sentinels = [
            parent / "pyproject.toml",
            parent / "setup.py",
            parent / "setup.cfg",
            parent / ".git",
            parent / "main.py",
        ]
        for sentinel in sentinels:
            if sentinel.exists():
                logger.debug(f"Found project root at {parent}")
                return parent

    logger.warning(f"Could not determine project root, falling back to {cwd}")
    return cwd


def find_workspace_dir() -> Optional[Path]:
    """
    Find the current workspace directory by checking common locations.

    The workspace is where code gets executed (Interpreter writes runfile.py here).
    It can be:
      - ./out/workspace (when running locally via `uv run main.py`)
      - /app/out/workspace (when running inside Docker)
      - A custom path set via ARL_out_dir env var

    Returns:
        Optional[Path]: Path to the workspace directory, or None if not found.
    """
    env_out_dir = os.environ.get("ARL_out_dir")
    if env_out_dir:
        candidate = Path(env_out_dir).resolve() / "workspace"
        if candidate.exists():
            return candidate

    candidates = [
        Path.cwd() / "out" / "workspace",
        Path.cwd().resolve() / "out" / "workspace",
        Path("/app/out/workspace"),
    ]

    if Path("/app").exists():
        candidates.append(Path("/app/out/workspace"))

    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Found workspace at {candidate}")
            return candidate

    logger.debug("No workspace directory found yet (will be created during execution)")
    return None


def scan_workspace_for_data_files(workspace_dir: Optional[Path] = None) -> str:
    """
    Scan ONLY the workspace directory for user-placed data files.

    Standard datasets should be loaded via the dataloader package.
    Only custom/user-provided files (CSV, Parquet, JSON, etc.) placed
    directly in the workspace are listed here.
    Internal files (.pkl, .py, .log) are ignored automatically.

    Args:
        workspace_dir: The workspace directory to scan. If None, auto-detect.

    Returns:
        str: A formatted block describing available files, or empty string if none found.
    """
    if workspace_dir is None:
        workspace_dir = find_workspace_dir()

    if workspace_dir is None or not workspace_dir.exists():
        return ""

    relevant_extensions = {
        ".csv": "CSV data file",
        ".tsv": "TSV data file",
        ".parquet": "Parquet data file",
        ".json": "JSON data file",
        ".jsonl": "JSONL data file",
    }

    found_files: list[tuple[Path, str]] = []

    # Only scan the workspace directory itself (non-recursive)
    # to avoid picking up runfile.py, working/, etc.
    for item in workspace_dir.iterdir():
        if item.is_file() and item.suffix.lower() in relevant_extensions:
            found_files.append((item, relevant_extensions.get(item.suffix.lower(), "Data file")))

    if not found_files:
        return ""

    lines = ["## Custom data files in workspace", ""]
    for fpath, desc in sorted(found_files, key=lambda x: x[0].name):
        size_str = _format_size(fpath.stat().st_size)
        lines.append(f"- `{fpath.name}` — {desc} ({size_str})")
    lines.append("")
    lines.append("To use a custom file in your code, reference it by its filename")
    lines.append("(the code runs inside the workspace directory):")
    lines.append("```python")
    lines.append("df = pd.read_csv('filename.csv')")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_workspace_context_block(
    project_root: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
) -> str:
    """
    Get a formatted context block about the workspace that can be injected
    into LLM prompts.

    This tells the planner:
      - Where the project root and workspace are
      - How to use the dataloader package for standard datasets
      - What custom data files the user placed in the workspace

    Args:
        project_root: Override for project root (auto-detected if None).
        workspace_dir: Override for workspace dir (auto-detected if None).

    Returns:
        str: A context block to inject into prompts.
    """
    if project_root is None:
        project_root = find_project_root()
    if workspace_dir is None:
        workspace_dir = find_workspace_dir()

    parts = [
        "## Workspace & File Context",
        "",
        f"- **Project root:** `{project_root}`",
        f"- **Code execution workspace:** `{workspace_dir if workspace_dir else 'Not yet created (will be ./out/workspace)'}`",
        "",
        "### How to access data files",
        "",
        "When your code runs, the working directory is the code execution workspace listed above.",
        "",
        "You have two options to load datasets:",
        "",
        "1. **Use the `dataloader` package** (recommended for standard datasets):",
        "   ```python",
        "   from dataloader.loaders.registry import _run_loader",
        "   df = _run_loader('MovieLens100K')  # Downloads & caches automatically",
        "   ```",
        "   Available datasets include: MovieLens100K, MovieLens1M, MovieLens10M, MovieLens20M,",
        "   MovieLens25M, MovieLensLatest, MovieLensLatestSmall, MovieLens1BSynthetic,",
        "   Amazon2014*, Amazon2018*, Amazon2023*, Yelp2023, Gowalla, BeerAdvocate, etc.",
        "",
        "2. **Use custom files placed in the workspace** (listed below):",
        "",
    ]

    # Only scan the workspace for user-placed custom files
    custom_files_block = scan_workspace_for_data_files(workspace_dir)
    if custom_files_block:
        parts.append(custom_files_block)
    else:
        parts.append("   *(No custom data files found in workspace.)*")
        parts.append("")

    return "\n".join(parts)
