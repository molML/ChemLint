"""Tests for SMILES_encoding.py - SMILES tokenization and one-hot encoding."""

import pandas as pd
import numpy as np
import pytest
from chemlint.tools.featurization.SMILES_encoding import (
    _tokenize_smiles,
    discover_tokens_from_dataset,
    create_default_vocab_json,
    compile_vocab_from_tokens,
    load_vocab_from_json,
    inspect_vocab_json,
    smiles_to_indices,
    batch_smiles_to_one_hot,
    flag_smiles_vocab_fit,
)
from chemlint.infrastructure.resources import create_project_manifest, _store_resource


def test_tokenize_smiles():
    """Test SMILES tokenization handles multi-character tokens, bonds, and stereochemistry."""
    # Basic single-character tokens
    assert _tokenize_smiles("CCO") == ["C", "C", "O"]
    assert _tokenize_smiles("CNS") == ["C", "N", "S"]
    
    # 2-character elements (Cl, Br) - CRITICAL TEST
    assert _tokenize_smiles("CCl") == ["C", "Cl"]
    assert _tokenize_smiles("CBr") == ["C", "Br"]
    assert _tokenize_smiles("ClCCl") == ["Cl", "C", "Cl"]
    assert _tokenize_smiles("BrCBr") == ["Br", "C", "Br"]
    # Test Cl/Br don't split into C+l or B+r
    assert _tokenize_smiles("CCCl") == ["C", "C", "Cl"]  # Not ["C", "C", "C", "l"]
    assert _tokenize_smiles("CBrC") == ["C", "Br", "C"]  # Not ["C", "B", "r", "C"]
    
    # Multiple 2-character elements in same molecule
    assert _tokenize_smiles("ClCBr") == ["Cl", "C", "Br"]
    
    # Bond types - single, double, triple
    assert _tokenize_smiles("C-C") == ["C", "-", "C"]  # Single bond (explicit)
    assert _tokenize_smiles("C=C") == ["C", "=", "C"]  # Double bond
    assert _tokenize_smiles("C#C") == ["C", "#", "C"]  # Triple bond
    assert _tokenize_smiles("C=O") == ["C", "=", "O"]  # Carbonyl
    assert _tokenize_smiles("C#N") == ["C", "#", "N"]  # Nitrile
    
    # Aromatic bonds (lowercase)
    assert _tokenize_smiles("c1ccccc1") == ["c", "1", "c", "c", "c", "c", "c", "1"]
    assert _tokenize_smiles("n1cccc1") == ["n", "1", "c", "c", "c", "c", "1"]
    
    # Stereochemistry - @ and @@ (2-character!)
    assert _tokenize_smiles("C@H") == ["C", "@", "H"]  # Single @
    assert _tokenize_smiles("C@@H") == ["C", "@@", "H"]  # Double @@ (must be ONE token)
    # Verify @@ is not split into two @ tokens
    tokens_chiral = _tokenize_smiles("C@@H")
    assert "@@" in tokens_chiral
    assert tokens_chiral.count("@") == 0  # Should be 0 single @ tokens
    assert tokens_chiral == ["C", "@@", "H"]
    
    # Stereochemistry - cis/trans (/ and \)
    assert _tokenize_smiles("C/C=C/C") == ["C", "/", "C", "=", "C", "/", "C"]
    assert _tokenize_smiles("C\\C=C\\C") == ["C", "\\", "C", "=", "C", "\\", "C"]
    assert _tokenize_smiles("C/C=C\\C") == ["C", "/", "C", "=", "C", "\\", "C"]
    
    # Complex stereochemistry in brackets
    tokens = _tokenize_smiles("C[C@H](O)C")
    assert "[C@H]" in tokens
    tokens = _tokenize_smiles("C[C@@H](O)C")
    assert "[C@@H]" in tokens
    
    # Charges in brackets
    tokens = _tokenize_smiles("C[NH+]C")
    assert "[NH+]" in tokens
    tokens = _tokenize_smiles("C[O-]")
    assert "[O-]" in tokens
    tokens = _tokenize_smiles("[NH4+]")
    assert "[NH4+]" in tokens
    
    # Ring numbers - single digit
    assert _tokenize_smiles("C1CC1") == ["C", "1", "C", "C", "1"]
    assert _tokenize_smiles("C1CCC1") == ["C", "1", "C", "C", "C", "1"]
    
    # Extended ring numbers (2-character: %10, %11, etc.)
    tokens = _tokenize_smiles("C%10")
    assert "%10" in tokens
    tokens = _tokenize_smiles("C%11CC%11")
    assert "%11" in tokens
    assert tokens.count("%11") == 2
    
    # Branches
    assert _tokenize_smiles("CC(C)C") == ["C", "C", "(", "C", ")", "C"]
    assert _tokenize_smiles("C(C)(C)C") == ["C", "(", "C", ")", "(", "C", ")", "C"]
    
    # Disconnected structures
    assert _tokenize_smiles("C.C") == ["C", ".", "C"]
    assert _tokenize_smiles("CCO.CCN") == ["C", "C", "O", ".", "C", "C", "N"]
    
    # Complex real-world examples combining multiple features
    # Chlorobenzene (aromatic + Cl)
    tokens = _tokenize_smiles("c1ccc(Cl)cc1")
    assert "Cl" in tokens
    assert "c" in tokens
    
    # Brominated alkene with stereochemistry
    tokens = _tokenize_smiles("Br/C=C/Br")
    assert "Br" in tokens
    assert tokens.count("Br") == 2
    assert "/" in tokens
    assert "=" in tokens


def test_discover_tokens_from_dataset(session_workdir, request):
    """Test token discovery from dataset."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create test dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CCl", "CC(C)O"],
        "id": [1, 2, 3, 4]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Discover tokens
    result = discover_tokens_from_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        min_frequency=1
    )
    
    # Check return structure
    assert "n_smiles" in result
    assert "n_unique_tokens" in result
    assert "tokens" in result
    assert "top_tokens" in result
    
    # Check values
    assert result["n_smiles"] == 4
    assert result["n_unique_tokens"] > 0
    
    # Should have found common tokens
    token_list = [t["token"] for t in result["tokens"]]
    assert "C" in token_list
    assert "c" in token_list
    assert "O" in token_list
    assert "Cl" in token_list
    assert "l" not in token_list  # 'l' alone should not be a token


def test_discover_tokens_with_frequency_filter(session_workdir, request):
    """Test token discovery with minimum frequency filter."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Dataset where some tokens appear only once
    df = pd.DataFrame({
        "smiles": ["CCO", "CCC", "CCCC", "Cl"]  # C appears 9 times, O once, Cl once
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Filter out rare tokens
    result = discover_tokens_from_dataset(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        min_frequency=3
    )
    
    # Only C should remain (appears 9 times)
    token_list = [t["token"] for t in result["tokens"]]
    assert "C" in token_list
    assert result["n_unique_tokens"] >= 1  # At least C


def test_create_default_vocab_json(session_workdir, request):
    """Test creating default vocabulary."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create default vocab with special tokens
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="default_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Should return a filename
    assert isinstance(vocab_file, str)
    assert "default_vocab" in vocab_file
    
    # Load and check contents
    vocab, special_tokens = load_vocab_from_json(vocab_file, manifest_path)
    
    # Should have pad and unk tokens
    assert '<pad>' in vocab
    assert '<unk>' in vocab
    assert 'pad' in special_tokens
    assert 'unk' in special_tokens
    
    # Should have common chemical tokens
    assert 'C' in vocab
    assert 'N' in vocab
    assert 'O' in vocab
    assert 'Cl' in vocab
    assert 'l' not in vocab  # 'l' alone should not be a token
    
    # Vocab size should be reasonable (44 default + 2 special = 46)
    assert len(vocab) >= 40


def test_compile_vocab_from_tokens(session_workdir, request):
    """Test compiling custom vocabulary from token list."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create custom minimal vocab
    custom_tokens = ["C", "O", "N", "Cl", "(", ")", "=", "1", "2"]
    vocab_file = compile_vocab_from_tokens(
        tokens=custom_tokens,
        project_manifest_path=manifest_path,
        output_filename="custom_vocab",
        explanation="Minimal vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Load and verify
    vocab, special_tokens = load_vocab_from_json(vocab_file, manifest_path)
    
    # Should have all custom tokens
    for token in custom_tokens:
        assert token in vocab
    
    # Should have special tokens
    assert '<pad>' in vocab
    assert '<unk>' in vocab
    
    # Size should be custom tokens + special tokens
    assert len(vocab) == len(custom_tokens) + 2


def test_inspect_vocab_json(session_workdir, request):
    """Test vocabulary inspection."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="inspect_test_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Inspect it
    info = inspect_vocab_json(vocab_file, manifest_path)
    
    # Check return structure
    assert "vocab_size" in info
    assert "special_tokens" in info
    assert "tokens" in info
    assert "token_to_index" in info
    
    # Check values
    assert info["vocab_size"] > 0
    assert 'pad' in info["special_tokens"]
    assert 'unk' in info["special_tokens"]
    assert isinstance(info["tokens"], list)
    assert isinstance(info["token_to_index"], dict)


def test_smiles_to_indices_basic(session_workdir, request):
    """Test converting SMILES to indices."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="test_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Convert simple SMILES
    result = smiles_to_indices(
        smiles="CCO",
        vocab_filename=vocab_file,
        project_manifest_path=manifest_path,
        max_length=None
    )
    
    # Check return structure
    assert "indices" in result
    assert "tokens" in result
    assert "length" in result
    
    # Should have 3 tokens
    assert len(result["tokens"]) == 3
    assert result["tokens"] == ["C", "C", "O"]
    assert len(result["indices"]) == 3
    assert result["length"] == 3


def test_smiles_to_indices_with_padding(session_workdir, request):
    """Test SMILES to indices with padding."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="test_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Convert with padding
    result = smiles_to_indices(
        smiles="CCO",
        vocab_filename=vocab_file,
        project_manifest_path=manifest_path,
        max_length=10
    )
    
    # Should be padded to length 10
    assert len(result["indices"]) == 10
    assert result["length"] == 3
    assert result["padded"] is True
    assert result["truncated"] is False


def test_smiles_to_indices_with_truncation(session_workdir, request):
    """Test SMILES to indices with truncation."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="test_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Convert with truncation
    result = smiles_to_indices(
        smiles="CCCCCCCCCC",  # 10 carbons
        vocab_filename=vocab_file,
        project_manifest_path=manifest_path,
        max_length=5
    )
    
    # Should be truncated to length 5
    assert len(result["indices"]) == 5
    assert result["length"] == 10  # Original length
    assert result["padded"] is False
    assert result["truncated"] is True


def test_batch_smiles_to_one_hot(session_workdir, request):
    """Test batch one-hot encoding of SMILES."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab
    vocab_file = create_default_vocab_json(
        project_manifest_path=manifest_path,
        output_filename="test_vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Create dataset
    df = pd.DataFrame({
        "smiles": ["CCO", "CC", "CCC"]
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Batch encode
    result = batch_smiles_to_one_hot(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        vocab_filename=vocab_file,
        output_filename="batch_onehot",
        max_length=5
    )
    
    # Check return structure
    assert "output_filename" in result
    assert "shape" in result
    assert "n_smiles" in result
    
    # Check shape (3 SMILES, max_length=5, vocab_size)
    assert result["shape"][0] == 3  # n_smiles
    assert result["shape"][1] == 5  # max_length
    assert result["shape"][2] > 40  # vocab_size (44 + special tokens)
    assert result["n_smiles"] == 3


def test_flag_smiles_vocab_fit(session_workdir, request):
    """Test flagging SMILES based on vocab fit."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create small vocab (missing some tokens)
    small_vocab_file = compile_vocab_from_tokens(
        tokens=["C", "O"],  # Very limited
        project_manifest_path=manifest_path,
        output_filename="small_vocab",
        explanation="Tiny vocab",
        add_pad=True,
        add_unk=True
    )
    
    # Create dataset with some SMILES that won't fit
    df = pd.DataFrame({
        "smiles": ["CCO", "CCN", "CCC"]  # CCN has N which is not in vocab
    })
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Flag SMILES
    result = flag_smiles_vocab_fit(
        input_filename=input_file,
        project_manifest_path=manifest_path,
        smiles_column="smiles",
        vocab_filename=small_vocab_file,
        add_coverage_column=True,
        output_filename="flagged_molecules"
    )
    
    # Check return structure
    assert "n_smiles" in result
    assert "n_passed" in result
    assert "n_unknown_token" in result
    assert "missing_tokens" in result
    assert "output_filename" in result
    
    # Should have found missing token 'N'
    assert "N" in result["missing_tokens"]
    assert result["n_smiles"] == 3
    
    # Not all should pass (CCN has N)
    assert result["n_passed"] < 3


def test_discover_tokens_invalid_column(session_workdir, request):
    """Test error handling for invalid column name."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    df = pd.DataFrame({"smiles": ["CCO"]})
    input_file = _store_resource(df, manifest_path, "test_molecules", "test data", "csv")
    
    # Try with non-existent column
    with pytest.raises(ValueError, match="not found"):
        discover_tokens_from_dataset(
            input_filename=input_file,
            project_manifest_path=manifest_path,
            smiles_column="nonexistent",
            min_frequency=1
        )


def test_smiles_to_indices_unknown_token_without_unk(session_workdir, request):
    """Test error when unknown token encountered without <unk> in vocab."""
    test_dir = session_workdir / request.node.name
    test_dir.mkdir(exist_ok=True)
    create_project_manifest(str(test_dir), "test")
    manifest_path = str(test_dir / "test_manifest.json")
    
    # Create vocab without <unk> and with limited tokens
    small_vocab_file = compile_vocab_from_tokens(
        tokens=["C", "O"],
        project_manifest_path=manifest_path,
        output_filename="no_unk_vocab",
        explanation="Vocab without unk",
        add_pad=True,
        add_unk=False  # No <unk> token
    )
    
    # Try to encode SMILES with token not in vocab
    with pytest.raises(ValueError, match="not in vocabulary"):
        smiles_to_indices(
            smiles="CCN",  # N is not in vocab
            vocab_filename=small_vocab_file,
            project_manifest_path=manifest_path,
            max_length=None
        )
