"""AutoForge - AI-powered multi-agent development platform."""

from pathlib import Path

__version__ = "2.7.22"

# Package data directory (constitution/, templates/ live here)
DATA_DIR: Path = Path(__file__).parent / "data"
