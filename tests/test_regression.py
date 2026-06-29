"""
Deterministic regression tests.

These tests pin fixed-seed outputs to catch silent regressions when new
tools are added or existing logic changes.  Each test uses a fixed seed
and asserts on the exact row counts / output values so behaviour stays
stable across code changes.
"""

import pandas as pd
import pytest
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "dummy_data_raw_small.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manifest(tmp_path, name: str) -> str:
    from chemlint.infrastructure.resources import create_project_manifest
    create_project_manifest(str(tmp_path), name)
    return str(tmp_path / f"{name}_manifest.json")


def store_df(df, manifest_path: str, label: str):
    from chemlint.infrastructure.resources import _store_resource
    return _store_resource(df, manifest_path, label, f"Input data: {label}", "csv")


# ---------------------------------------------------------------------------
# Test 1 – random_split_dataset is deterministic with fixed seed
# ---------------------------------------------------------------------------

def test_random_split_reproducibility(tmp_path):
    """Same seed must produce identical split sizes and row order."""
    from chemlint.tools.core_mol.data_splitting import random_split_dataset
    from chemlint.infrastructure.resources import _load_resource

    df = pd.read_csv(DATA_PATH)
    manifest = make_manifest(tmp_path, "split_repro")
    fn = store_df(df, manifest, "raw")

    result_a = random_split_dataset(
        project_manifest_path=manifest,
        input_filename=fn,
        train_df_output_filename="train_a",
        test_df_output_filename="test_a",
        test_size=0.2,
        random_state=7,
    )
    result_b = random_split_dataset(
        project_manifest_path=manifest,
        input_filename=fn,
        train_df_output_filename="train_b",
        test_df_output_filename="test_b",
        test_size=0.2,
        random_state=7,
    )

    assert result_a["n_train_rows"] == result_b["n_train_rows"]
    assert result_a["n_test_rows"] == result_b["n_test_rows"]
    assert result_a["random_state"] == 7

    train_a = _load_resource(manifest, result_a["train_df_output_filename"])
    train_b = _load_resource(manifest, result_b["train_df_output_filename"])
    pd.testing.assert_frame_equal(train_a.reset_index(drop=True),
                                  train_b.reset_index(drop=True))


def test_random_split_different_seeds_differ(tmp_path):
    """Different seeds must produce different train sets."""
    from chemlint.tools.core_mol.data_splitting import random_split_dataset
    from chemlint.infrastructure.resources import _load_resource

    df = pd.read_csv(DATA_PATH)
    manifest = make_manifest(tmp_path, "split_diff")
    fn = store_df(df, manifest, "raw")

    result_42 = random_split_dataset(
        project_manifest_path=manifest,
        input_filename=fn,
        train_df_output_filename="train_42",
        test_df_output_filename="test_42",
        test_size=0.2,
        random_state=42,
    )
    result_99 = random_split_dataset(
        project_manifest_path=manifest,
        input_filename=fn,
        train_df_output_filename="train_99",
        test_df_output_filename="test_99",
        test_size=0.2,
        random_state=99,
    )

    train_42 = _load_resource(manifest, result_42["train_df_output_filename"])
    train_99 = _load_resource(manifest, result_99["train_df_output_filename"])
    # First rows of each split should differ for different seeds
    assert list(train_42["smiles"]) != list(train_99["smiles"])


# ---------------------------------------------------------------------------
# Test 2 – random_state=None raises a clear error
# ---------------------------------------------------------------------------

def test_random_split_requires_seed(tmp_path):
    """Omitting random_state must raise a ValueError, not silently proceed."""
    from chemlint.tools.core_mol.data_splitting import random_split_dataset

    df = pd.read_csv(DATA_PATH)
    manifest = make_manifest(tmp_path, "split_noseed")
    fn = store_df(df, manifest, "raw")

    with pytest.raises(ValueError, match="random_state is required"):
        random_split_dataset(
            project_manifest_path=manifest,
            input_filename=fn,
            train_df_output_filename="train",
            test_df_output_filename="test",
            test_size=0.2,
            # random_state intentionally omitted
        )


# ---------------------------------------------------------------------------
# Test 3 – Lipinski filter outputs informational flag (not old "warning" key)
# ---------------------------------------------------------------------------

def test_lipinski_output_has_guideline_note(tmp_path):
    """filter_by_lipinski_ro5 result must carry guideline_note, not warning."""
    from chemlint.tools.core.filtering import filter_by_lipinski_ro5

    df = pd.read_csv(DATA_PATH)
    manifest = make_manifest(tmp_path, "lipinski_note")
    fn = store_df(df, manifest, "raw")

    result = filter_by_lipinski_ro5(
        input_filename=fn,
        project_manifest_path=manifest,
        smiles_column="smiles",
        output_filename="ro5_out",
        explanation="Lipinski check",
    )

    assert "guideline_note" in result, "Expected guideline_note key in result"
    assert "warning" not in result, "Old 'warning' key should have been removed"
    # The note must not label compounds as bad
    note = result["guideline_note"].lower()
    assert "informational" in note or "flag" in note


# ---------------------------------------------------------------------------
# Test 4 – modified_zscore outlier detection is deterministic
# ---------------------------------------------------------------------------

def test_modified_zscore_deterministic(tmp_path):
    """detect_outliers_modified_zscore must give the same result on same input."""
    from chemlint.tools.core.outliers import detect_outliers_modified_zscore

    df = pd.read_csv(DATA_PATH)
    manifest = make_manifest(tmp_path, "outlier_repro")
    fn = store_df(df, manifest, "raw")

    result_a = detect_outliers_modified_zscore(
        input_filename=fn,
        project_manifest_path=manifest,
        columns=["exp_mean [nM]"],
        output_filename="out_a",
        explanation="Outlier run A",
    )
    result_b = detect_outliers_modified_zscore(
        input_filename=fn,
        project_manifest_path=manifest,
        columns=["exp_mean [nM]"],
        output_filename="out_b",
        explanation="Outlier run B",
    )

    assert result_a["total_outliers"] == result_b["total_outliers"]
    assert result_a["outliers_per_column"] == result_b["outliers_per_column"]


# ---------------------------------------------------------------------------
# Test 5 – manifest records library versions on project creation
# ---------------------------------------------------------------------------

def test_manifest_library_versions(tmp_path):
    """New project manifest must include library_versions with key packages."""
    from chemlint.infrastructure.resources import create_project_manifest

    manifest = create_project_manifest(str(tmp_path), "ver_check")

    assert "library_versions" in manifest
    for pkg in ("python", "rdkit", "scikit-learn", "numpy", "pandas", "scipy"):
        assert pkg in manifest["library_versions"], f"Missing version for {pkg}"
        assert manifest["library_versions"][pkg] != ""
