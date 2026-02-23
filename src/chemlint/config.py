import os
from pathlib import Path


# You can set your custom data directory by running (change path to desired location): export CHEMLINT_DATA_DIR="~/user/chemlint_data"
# If not set, defaults to ~/.chemlint/

def get_data_root() -> Path:
    # Allow user to override via environment variable
    custom = os.getenv("MOLML_MCP_DATA_DIR")
    if custom:
        root = Path(custom).expanduser()
    else:
        # Default: ~/.chemlint/
        root = Path.home() / ".chemlint"

    root.mkdir(parents=True, exist_ok=True)
    return root

DATA_ROOT = get_data_root()
LOG_PATH = DATA_ROOT / "history.log"