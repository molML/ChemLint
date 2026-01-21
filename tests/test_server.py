"""Tests for server.py initialization."""
import sys
import subprocess
from pathlib import Path
import pytest


@pytest.mark.slow
@pytest.mark.server
def test_server_imports_and_initializes():
    """Test that server.py can be imported and initialized without errors."""
    # Run server.py with __main__ block which initializes and exits cleanly
    server_path = Path(__file__).parent.parent / "src" / "molml_mcp" / "server.py"
    
    # Try to find the virtual environment python
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable
    
    # Run the server script directly - it has a __main__ block that exits after init
    result = subprocess.run(
        [python_cmd, str(server_path)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Check that initialization succeeded
    assert result.returncode == 0, f"Server initialization failed: {result.stderr}"
    assert "✓" in result.stdout or "initialized successfully" in result.stdout, f"Server did not complete initialization: {result.stdout}"
