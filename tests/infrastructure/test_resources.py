import pytest
from pathlib import Path


def test_a(session_workdir, request):
    d = session_workdir / request.node.name
    d.mkdir(exist_ok=True)
    (d / "out.txt").write_text("A")

    assert (d / "out.txt").read_text() == "A"


def test_store_load_resource(session_workdir):
    from molml_mcp.infrastructure.resources import _store_resource, _load_resource

    data = {"a": 1, "b": 2}
    filename = _store_resource(data, session_workdir / "test_manifest.json", "store_resource_test_data", "Test data storage", "json")
    loaded_data = _load_resource(session_workdir / "test_manifest.json", filename)

    assert data == loaded_data
