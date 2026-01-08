"""Tests for server.py initialization."""
import sys
import subprocess
from pathlib import Path
import pytest


@pytest.mark.slow
def test_server_imports_and_initializes():
    """Test that server.py can be imported and initialized without errors."""
    # Run server.py directly as subprocess using the virtual environment
    server_path = Path(__file__).parent.parent / "src" / "molml_mcp" / "server.py"
    
    # Try to find the virtual environment python
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable
    
    # Run the server script to check if it initializes without errors
    result = subprocess.run(
        [python_cmd, str(server_path)],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    # The script should exit cleanly (exit code 0)
    assert result.returncode == 0, f"Server failed to run: {result.stderr}"
