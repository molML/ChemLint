
import sys
from pathlib import Path

# Ensure the project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../molml_mcp
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
from pathlib import Path


# You can set your custom data directory by running (change path to desired location): export MOLML_MCP_DATA_DIR="~/user/molml_mcp_data"
# If not set, defaults to ~/.molml_mcp/

def get_data_root() -> Path:
    # Allow user to override via environment variable
    custom = os.getenv("MOLML_MCP_DATA_DIR")
    if custom:
        root = Path(custom).expanduser()
    else:
        # Default: ~/.molml_mcp/
        root = Path.home() / ".molml_mcp"

    root.mkdir(parents=True, exist_ok=True)
    return root

DATA_ROOT = get_data_root()
LOG_PATH = DATA_ROOT / "history.log"