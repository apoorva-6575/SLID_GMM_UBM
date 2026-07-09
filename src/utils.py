"""General project utilities."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path):
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
