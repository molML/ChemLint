import pytest
from pathlib import Path
import tempfile
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from chemlint.infrastructure.resources import create_project_manifest

@pytest.fixture(scope="session")
def session_workdir():
    d = Path(tempfile.mkdtemp())
    create_project_manifest(d, "test")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# import sys
# from pathlib import Path

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# from chemlint.infrastructure.resources import _store_resource, _load_resource

# # Test manifest path
# TEST_MANIFEST = Path(__file__).parent / 'data' / 'test_manifest.json'
