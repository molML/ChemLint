# MCP-exposed functions gatherer
def get_all_smiles_encoding_tools():
    """
    Returns a list of MCP-exposed SMILES encoding functions for server registration.
    """
    return [
        discover_tokens_from_dataset,
        create_default_vocab_json,
        compile_vocab_from_tokens,
        load_vocab_from_json,
        inspect_vocab_json,
        smiles_to_indices,
        batch_smiles_to_one_hot,
        flag_smiles_vocab_fit,
    ]

import re
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from collections import Counter
from pathlib import Path
from molml_mcp.infrastructure.resources import _load_resource, _store_resource





# Default token list for drug-like molecules
DEFAULT_TOKENS = [
    'C', 'c',  # Carbon (aliphatic, aromatic)
    'N', 'n',  # Nitrogen
    'O', 'o',  # Oxygen
    'S', 's',  # Sulfur
    'P', 'p',  # Phosphorus
    'F',       # Fluorine
    'Cl',      # Chlorine
    'Br',      # Bromine
    'I',       # Iodine
    'H',       # Hydrogen (explicit)
    'B',       # Boron
    '(',       # Branch start
    ')',       # Branch end
    '[',       # Bracket start (for complex atoms)
    ']',       # Bracket end
    '=',       # Double bond
    '#',       # Triple bond
    '@',       # Chirality
    '@@',      # Chirality (opposite)
    '+',       # Positive charge
    '-',       # Negative charge / single bond
    '/',       # Stereochemistry up
    '\\',      # Stereochemistry down
    '.',       # Disconnection
    '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Ring numbers
    '%10', '%11', '%12', '%13', '%14', '%15',  # Extended ring numbers
]


# SMILES tokenization pattern - matches multi-character tokens first
SMILES_REGEX = re.compile(
    r'(\[[^\]]+\]|'      # Bracketed atoms (e.g., [nH], [C@@H], [O-])
    r'@@|'               # Chirality (2 chars, must come before @)
    r'%\d{2}|'           # Ring numbers >= 10 (e.g., %10, %11)
    r'Br|Cl|'            # 2-character elements
    r'[BCNOPSFIHbcnopsfi]|'  # Single-character elements
    r'[=#@+\-\\\/().\[\]]|'  # Bonds, structural symbols, brackets
    r'\d)'               # Single digits (ring numbers)
)


def _tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into tokens.
    
    Handles multi-character tokens (Cl, Br, @@, bracketed atoms).
    
    Args:
        smiles: SMILES string to tokenize
        
    Returns:
        List of tokens
        
    Example:
        >>> _tokenize_smiles('CCCl')
        ['C', 'C', 'Cl']
        >>> _tokenize_smiles('C@@H')
        ['C', '@@', 'H']
    """
    return SMILES_REGEX.findall(smiles)


def discover_tokens_from_dataset(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    min_frequency: int = 1,
    max_tokens: Optional[int] = None
) -> Dict:
    """
    Discover all tokens from SMILES strings in a dataset column.
    
    Tokenizes all SMILES using the regex tokenizer (properly handling multi-character
    tokens like Cl, Br, @@, and bracketed atoms) and counts their frequencies.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        smiles_column: Column name containing SMILES strings
        min_frequency: Minimum frequency for a token to be included in results
        max_tokens: Maximum number of tokens to return (sorted by frequency)
        
    Returns:
        Dictionary containing:
            - n_smiles: Number of SMILES strings processed
            - n_unique_tokens: Number of unique tokens found
            - n_total_tokens: Total number of tokens (sum of all frequencies)
            - tokens: List of dicts with 'token' and 'count' keys, sorted by frequency
            - top_tokens: Top 20 most frequent tokens (for quick inspection)
            - min_frequency: Minimum frequency filter applied
            
    Example:
        >>> result = discover_tokens_from_dataset(
        ...     "molecules_A1B2C3D4.csv",
        ...     "/path/to/manifest.json",
        ...     "smiles",
        ...     min_frequency=10
        ... )
        >>> print(f"Found {result['n_unique_tokens']} unique tokens")
        >>> print(f"Top token: {result['top_tokens'][0]}")
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Get SMILES strings (drop NaN values)
    smiles_list = df[smiles_column].dropna().tolist()
    n_smiles = len(smiles_list)
    
    if n_smiles == 0:
        raise ValueError(f"No valid SMILES found in column '{smiles_column}'")
    
    # Tokenize all SMILES and count frequencies
    token_counter = Counter()
    n_failed = 0
    
    for smiles in smiles_list:
        try:
            tokens = _tokenize_smiles(str(smiles))
            token_counter.update(tokens)
        except Exception:
            n_failed += 1
            continue
    
    # Filter by minimum frequency
    filtered_tokens = {
        token: count 
        for token, count in token_counter.items() 
        if count >= min_frequency
    }
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(
        filtered_tokens.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Apply max_tokens limit if specified
    if max_tokens is not None and len(sorted_tokens) > max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens]
    
    # Format tokens as list of dicts
    tokens_list = [
        {"token": token, "count": count}
        for token, count in sorted_tokens
    ]
    
    # Get top 20 for quick inspection
    top_tokens = [
        {"token": token, "count": count}
        for token, count in sorted_tokens[:20]
    ]
    
    # Calculate statistics
    n_unique_tokens = len(filtered_tokens)
    n_total_tokens = sum(filtered_tokens.values())
    
    return {
        "n_smiles": n_smiles,
        "n_failed": n_failed,
        "n_unique_tokens": n_unique_tokens,
        "n_total_tokens": n_total_tokens,
        "min_frequency": min_frequency,
        "max_tokens_limit": max_tokens,
        "tokens": tokens_list,
        "top_tokens": top_tokens,
        "summary": f"Discovered {n_unique_tokens} unique tokens from {n_smiles} SMILES strings (min frequency: {min_frequency})"
    }


def _build_vocab(
    tokens: Optional[List[str]] = None,
    add_pad: bool = False,
    add_start: bool = False,
    add_end: bool = False,
    add_unk: bool = False,
    pad_token: str = '<pad>',
    start_token: str = '<start>',
    end_token: str = '<end>',
    unk_token: str = '<unk>'
) -> Dict[str, int]:
    """
    Build a vocabulary dictionary from tokens.
    
    Special tokens are added at the beginning if specified:
    - <pad>: padding token (typically index 0)
    - <start>: start of sequence token
    - <end>: end of sequence token
    - <unk>: unknown token (for tokens not in vocab)
    
    Args:
        tokens: List of tokens to include. If None, uses DEFAULT_TOKENS
        add_pad: Add padding token
        add_start: Add start token
        add_end: Add end token
        add_unk: Add unknown token
        pad_token: String to use for padding token
        start_token: String to use for start token
        end_token: String to use for end token
        unk_token: String to use for unknown token
        
    Returns:
        Dictionary mapping tokens to integer indices
        
    Example:
        >>> vocab = _build_vocab(add_pad=True, add_unk=True)
        >>> len(vocab)
        46  # 44 default tokens + pad + unk
        >>> vocab['<pad>']
        0
        >>> vocab['C']
        2
    """
    if tokens is None:
        tokens = DEFAULT_TOKENS.copy()
    else:
        tokens = list(tokens)  # Make a copy to avoid modifying input
    
    # Add special tokens at the beginning
    special_tokens = []
    if add_pad:
        special_tokens.append(pad_token)
    if add_start:
        special_tokens.append(start_token)
    if add_end:
        special_tokens.append(end_token)
    if add_unk:
        special_tokens.append(unk_token)
    
    # Combine special tokens with regular tokens
    all_tokens = special_tokens + tokens
    
    # Create vocabulary dictionary
    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    
    return vocab


def smiles_to_indices(
    smiles: str,
    vocab_filename: str,
    project_manifest_path: str,
    max_length: Optional[int] = None
) -> Dict:
    """
    Convert SMILES string to list of integer indices using vocabulary from JSON.
    
    MCP-friendly version that loads vocab from JSON resource.
    
    Args:
        smiles: SMILES string to encode
        vocab_filename: Vocab JSON resource filename
        project_manifest_path: Path to project manifest.json
        max_length: Maximum length (will pad/truncate). If None, no padding/truncation
        
    Returns:
        Dictionary containing:
            - indices: List of integer indices
            - tokens: List of tokens (for reference)
            - length: Number of tokens before padding/truncation
            - padded: Whether padding was applied
            - truncated: Whether truncation was applied
        
    Example:
        >>> result = smiles_to_indices(
        ...     'CCl',
        ...     'my_vocab.json',
        ...     'manifest.json',
        ...     max_length=5
        ... )
        >>> print(result['indices'])  # [2, 2, 13, 0, 0]
    """
    # Load vocab
    vocab, special_tokens = load_vocab_from_json(vocab_filename, project_manifest_path)
    
    # Tokenize SMILES
    tokens = _tokenize_smiles(smiles)
    original_length = len(tokens)
    
    # Get special token strings
    pad_token = special_tokens.get('pad', '<pad>')
    unk_token = special_tokens.get('unk', '<unk>')
    
    # Get pad and unk indices
    pad_idx = vocab.get(pad_token, 0)
    unk_idx = vocab.get(unk_token, None)
    
    # Convert tokens to indices
    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab[token])
        elif unk_idx is not None:
            indices.append(unk_idx)
        else:
            raise ValueError(
                f"Token '{token}' not in vocabulary and no <unk> token defined. "
                f"Vocab: {vocab_filename}"
            )
    
    # Track padding/truncation
    padded = False
    truncated = False
    
    # Apply padding/truncation if max_length specified
    if max_length is not None:
        if len(indices) < max_length:
            # Pad
            indices = indices + [pad_idx] * (max_length - len(indices))
            padded = True
        elif len(indices) > max_length:
            # Truncate
            indices = indices[:max_length]
            truncated = True
    
    return {
        "indices": indices,
        "tokens": tokens,
        "length": original_length,
        "padded": padded,
        "truncated": truncated,
        "max_length": max_length
    }


def smiles_to_one_hot(
    smiles: str,
    vocab_filename: str,
    project_manifest_path: str,
    max_length: int,
    output_filename: Optional[str] = None,
    explanation: str = "One-hot encoded SMILES"
) -> Dict:
    """
    Convert SMILES string to one-hot encoded matrix using vocab from JSON.
    
    MCP-friendly version that loads vocab from JSON and optionally saves output.
    
    Args:
        smiles: SMILES string to encode
        vocab_filename: Vocab JSON resource filename
        project_manifest_path: Path to project manifest.json
        max_length: Maximum sequence length (required for one-hot encoding)
        output_filename: If provided, saves one-hot matrix as numpy resource
        explanation: Description for saved resource
        
    Returns:
        Dictionary containing:
            - shape: Tuple (max_length, vocab_size)
            - tokens: List of tokens
            - output_filename: Saved resource filename (if output_filename provided)
        
    Example:
        >>> result = smiles_to_one_hot(
        ...     'CC',
        ...     'my_vocab.json',
        ...     'manifest.json',
        ...     max_length=3,
        ...     output_filename='cc_onehot'
        ... )
        >>> print(result['shape'])  # (3, 45)
    """
    # Load vocab
    vocab, special_tokens = load_vocab_from_json(vocab_filename, project_manifest_path)
    
    # Get indices using internal helper
    indices_result = smiles_to_indices(smiles, vocab_filename, project_manifest_path, max_length)
    indices = indices_result['indices']
    tokens = indices_result['tokens']
    
    # Create one-hot matrix
    vocab_size = len(vocab)
    one_hot = np.zeros((max_length, vocab_size), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1.0
    
    result = {
        "shape": one_hot.shape,
        "tokens": tokens,
        "max_length": max_length,
        "vocab_size": vocab_size
    }
    
    # Save if output filename provided
    if output_filename:
        saved_filename = _store_resource(
            one_hot,
            project_manifest_path,
            output_filename,
            explanation,
            'joblib'
        )
        result['output_filename'] = saved_filename
    
    return result


def batch_smiles_to_one_hot(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    vocab_filename: str,
    output_filename: str,
    max_length: Optional[int] = None,
    explanation: str = "Batch one-hot encoded SMILES"
) -> Dict:
    """
    Convert a batch of SMILES from dataset to one-hot encoded 3D array.
    
    MCP-friendly version that loads SMILES from dataset and vocab from JSON.
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        smiles_column: Column name containing SMILES strings
        vocab_filename: Vocab JSON resource filename
        output_filename: Name for output numpy array resource
        max_length: Maximum sequence length. If None, uses longest SMILES in batch
        explanation: Description for saved resource
        
    Returns:
        Dictionary containing:
            - output_filename: Saved resource filename
            - shape: Tuple (n_smiles, max_length, vocab_size)
            - n_smiles: Number of SMILES encoded
            - max_length: Maximum length used
            - vocab_size: Size of vocabulary
        
    Example:
        >>> result = batch_smiles_to_one_hot(
        ...     'molecules.csv',
        ...     'manifest.json',
        ...     'SMILES',
        ...     'my_vocab.json',
        ...     'batch_onehot'
        ... )
        >>> print(result['shape'])  # (100, 50, 45)
    """
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get SMILES list
    smiles_list = df[smiles_column].dropna().tolist()
    n_smiles = len(smiles_list)
    
    if n_smiles == 0:
        raise ValueError(f"No valid SMILES in column '{smiles_column}'")
    
    # Load vocab
    vocab, special_tokens = load_vocab_from_json(vocab_filename, project_manifest_path)
    vocab_size = len(vocab)
    
    # Get special token strings
    pad_token = special_tokens.get('pad', '<pad>')
    unk_token = special_tokens.get('unk', '<unk>')
    pad_idx = vocab.get(pad_token, 0)
    unk_idx = vocab.get(unk_token, None)
    
    # Determine max_length if not specified
    if max_length is None:
        max_length = max(len(_tokenize_smiles(s)) for s in smiles_list)
    
    # Encode each SMILES
    batch = np.zeros((n_smiles, max_length, vocab_size), dtype=np.float32)
    
    for i, smiles in enumerate(smiles_list):
        tokens = _tokenize_smiles(smiles)
        
        # Convert to indices
        for j, token in enumerate(tokens[:max_length]):
            if token in vocab:
                idx = vocab[token]
            elif unk_idx is not None:
                idx = unk_idx
            else:
                raise ValueError(
                    f"Token '{token}' not in vocabulary and no <unk> token. "
                    f"Vocab: {vocab_filename}"
                )
            batch[i, j, idx] = 1.0
        
        # Pad if necessary
        if len(tokens) < max_length:
            for j in range(len(tokens), max_length):
                batch[i, j, pad_idx] = 1.0
    
    # Save batch
    saved_filename = _store_resource(
        batch,
        project_manifest_path,
        output_filename,
        explanation,
        'joblib'
    )
    
    return {
        "output_filename": saved_filename,
        "shape": batch.shape,
        "n_smiles": n_smiles,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "summary": (
            f"Encoded {n_smiles} SMILES to one-hot array of shape "
            f"({n_smiles}, {max_length}, {vocab_size})"
        )
    }


def _save_vocab_to_json(
    vocab: Dict[str, int],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "SMILES vocabulary for tokenization",
    special_tokens: Optional[Dict[str, str]] = None
) -> str:
    """
    Save vocabulary to JSON file as a project resource.
    
    Args:
        vocab: Vocabulary dictionary (token -> index)
        project_manifest_path: Path to project manifest.json
        output_filename: Name for output vocab JSON (without extension)
        explanation: Description of vocabulary
        special_tokens: Dict with keys 'pad', 'start', 'end', 'unk' mapping to token strings
        
    Returns:
        Full filename of saved vocab resource
        
    Example:
        >>> vocab = _build_vocab(add_pad=True, add_unk=True)
        >>> filename = _save_vocab_to_json(
        ...     vocab,
        ...     "manifest.json",
        ...     "my_vocab",
        ...     special_tokens={'pad': '<pad>', 'unk': '<unk>'}
        ... )
    """
    # Prepare vocab data with metadata
    vocab_data = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "special_tokens": special_tokens or {},
        "metadata": {
            "description": explanation,
            "token_count": len(vocab)
        }
    }
    
    # Store as JSON resource
    output_id = _store_resource(
        vocab_data,
        project_manifest_path,
        output_filename,
        explanation,
        'json'
    )
    
    return output_id


def load_vocab_from_json(
    input_filename: str,
    project_manifest_path: str
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Load vocabulary from JSON resource.
    
    Args:
        input_filename: Vocab JSON resource filename
        project_manifest_path: Path to project manifest.json
        
    Returns:
        Tuple of (vocab dict, special_tokens dict)
        
    Example:
        >>> vocab, special_tokens = load_vocab_from_json(
        ...     "my_vocab_A1B2C3D4.json",
        ...     "manifest.json"
        ... )
        >>> print(f"Loaded vocab with {len(vocab)} tokens")
    """
    # Load JSON resource
    vocab_data = _load_resource(project_manifest_path, input_filename)
    
    # Extract vocab and special tokens
    vocab = vocab_data.get("vocab", {})
    special_tokens = vocab_data.get("special_tokens", {})
    
    return vocab, special_tokens


def inspect_vocab_json(
    input_filename: str,
    project_manifest_path: str
) -> Dict:
    """
    Inspect a vocabulary JSON resource and return its contents and statistics.
    
    MCP-friendly function that shows what's in a vocab without loading it into memory.
    
    Args:
        input_filename: Vocab JSON resource filename
        project_manifest_path: Path to project manifest.json
        
    Returns:
        Dictionary containing:
            - vocab_size: Number of tokens in vocabulary
            - special_tokens: Dict of special tokens (pad, start, end, unk)
            - tokens: List of all tokens in vocab
            - token_to_index: Full vocab mapping (token -> index)
            - description: Metadata description
            
    Example:
        >>> info = inspect_vocab_json(
        ...     "my_vocab_A1B2C3D4.json",
        ...     "manifest.json"
        ... )
        >>> print(f"Vocab has {info['vocab_size']} tokens")
        >>> print(f"Special tokens: {info['special_tokens']}")
    """
    # Load JSON resource
    vocab_data = _load_resource(project_manifest_path, input_filename)
    
    # Extract components
    vocab = vocab_data.get("vocab", {})
    special_tokens = vocab_data.get("special_tokens", {})
    metadata = vocab_data.get("metadata", {})
    
    # Get sorted list of tokens
    tokens_sorted = sorted(vocab.items(), key=lambda x: x[1])
    tokens_list = [token for token, idx in tokens_sorted]
    
    return {
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "tokens": tokens_list,
        "token_to_index": vocab,
        "description": metadata.get("description", "No description"),
        "summary": (
            f"Vocabulary with {len(vocab)} tokens. "
            f"Special tokens: {list(special_tokens.keys())}. "
            f"{metadata.get('description', '')}"
        )
    }


def create_default_vocab_json(
    project_manifest_path: str,
    output_filename: str = "default_smiles_vocab",
    add_pad: bool = True,
    add_start: bool = False,
    add_end: bool = False,
    add_unk: bool = True
) -> str:
    """
    Create and save a default SMILES vocabulary JSON for drug-like molecules.
    
    Uses DEFAULT_TOKENS (44 tokens covering common organic chemistry elements,
    bonds, rings, charges, and stereochemistry).
    
    Args:
        project_manifest_path: Path to project manifest.json
        output_filename: Name for output vocab (default: "default_smiles_vocab")
        add_pad: Include <pad> token
        add_start: Include <start> token
        add_end: Include <end> token
        add_unk: Include <unk> token
        
    Returns:
        Full filename of saved vocab resource
        
    Example:
        >>> vocab_file = create_default_vocab_json(
        ...     "manifest.json",
        ...     add_pad=True,
        ...     add_unk=True
        ... )
        >>> print(f"Created default vocab: {vocab_file}")
    """
    # Build vocab from defaults
    vocab = _build_vocab(
        tokens=None,  # Uses DEFAULT_TOKENS
        add_pad=add_pad,
        add_start=add_start,
        add_end=add_end,
        add_unk=add_unk
    )
    
    # Track which special tokens were added
    special_tokens = {}
    if add_pad:
        special_tokens['pad'] = '<pad>'
    if add_start:
        special_tokens['start'] = '<start>'
    if add_end:
        special_tokens['end'] = '<end>'
    if add_unk:
        special_tokens['unk'] = '<unk>'
    
    # Save to JSON
    explanation = (
        f"Default SMILES vocabulary with {len(vocab)} tokens "
        f"(44 base tokens + {len(special_tokens)} special tokens). "
        f"Covers common organic chemistry: C, N, O, S, P, halogens, "
        f"bonds, rings, charges, stereochemistry."
    )
    
    output_id = _save_vocab_to_json(
        vocab,
        project_manifest_path,
        output_filename,
        explanation,
        special_tokens
    )
    
    return output_id


def compile_vocab_from_tokens(
    tokens: List[str],
    project_manifest_path: str,
    output_filename: str,
    explanation: str = "Custom SMILES vocabulary",
    add_pad: bool = False,
    add_start: bool = False,
    add_end: bool = False,
    add_unk: bool = False
) -> str:
    """
    Compile a vocabulary from user-specified tokens and save to JSON.
    
    Args:
        tokens: List of tokens to include in vocabulary
        project_manifest_path: Path to project manifest.json
        output_filename: Name for output vocab JSON
        explanation: Description of vocabulary
        add_pad: Include <pad> token
        add_start: Include <start> token
        add_end: Include <end> token
        add_unk: Include <unk> token
        
    Returns:
        Full filename of saved vocab resource
        
    Example:
        >>> custom_tokens = ['C', 'N', 'O', 'Cl', '(', ')', '=', '1', '2']
        >>> vocab_file = compile_vocab_from_tokens(
        ...     custom_tokens,
        ...     "manifest.json",
        ...     "small_vocab",
        ...     "Minimal vocab for simple molecules",
        ...     add_pad=True,
        ...     add_unk=True
        ... )
    """
    # Build vocab
    vocab = _build_vocab(
        tokens=tokens,
        add_pad=add_pad,
        add_start=add_start,
        add_end=add_end,
        add_unk=add_unk
    )
    
    # Track special tokens
    special_tokens = {}
    if add_pad:
        special_tokens['pad'] = '<pad>'
    if add_start:
        special_tokens['start'] = '<start>'
    if add_end:
        special_tokens['end'] = '<end>'
    if add_unk:
        special_tokens['unk'] = '<unk>'
    
    # Save to JSON
    output_id = _save_vocab_to_json(
        vocab,
        project_manifest_path,
        output_filename,
        explanation,
        special_tokens
    )
    
    return output_id


def _check_smiles_vocab_coverage(
    smiles: str,
    vocab_filename: str,
    project_manifest_path: str,
    return_unknown: bool = True
) -> Dict:
    """
    Check if a SMILES string can be fully tokenized with vocabulary from JSON.
    
    Internal helper function.
    
    Args:
        smiles: SMILES string to check
        vocab_filename: Vocab JSON resource filename
        project_manifest_path: Path to project manifest.json
        return_unknown: Include list of unknown tokens in output
        
    Returns:
        Dictionary containing:
            - can_tokenize: bool, True if all tokens are in vocab
            - n_tokens: Total number of tokens
            - n_known: Number of tokens in vocab
            - n_unknown: Number of tokens not in vocab
            - coverage: Percentage of tokens in vocab (0-100)
            - unknown_tokens: List of tokens not in vocab (if return_unknown=True)
            
    Example:
        >>> result = check_smiles_vocab_coverage(
        ...     'CCCl',
        ...     'my_vocab.json',
        ...     'manifest.json'
        ... )
        >>> if result['can_tokenize']:
        ...     print("All tokens found in vocab!")
        >>> else:
        ...     print(f"Missing tokens: {result['unknown_tokens']}")
    """
    # Load vocab
    vocab, _ = load_vocab_from_json(vocab_filename, project_manifest_path)
    
    # Tokenize SMILES
    tokens = _tokenize_smiles(smiles)
    n_tokens = len(tokens)
    
    if n_tokens == 0:
        return {
            "can_tokenize": True,
            "n_tokens": 0,
            "n_known": 0,
            "n_unknown": 0,
            "coverage": 100.0,
            "unknown_tokens": []
        }
    
    # Check which tokens are in vocab
    unknown_tokens = []
    n_known = 0
    
    for token in tokens:
        if token in vocab:
            n_known += 1
        else:
            if return_unknown and token not in unknown_tokens:
                unknown_tokens.append(token)
    
    n_unknown = n_tokens - n_known
    coverage = (n_known / n_tokens) * 100.0
    can_tokenize = (n_unknown == 0)
    
    result = {
        "can_tokenize": can_tokenize,
        "n_tokens": n_tokens,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "coverage": round(coverage, 2),
    }
    
    if return_unknown:
        result["unknown_tokens"] = unknown_tokens
    
    return result


def flag_smiles_vocab_fit(
    input_filename: str,
    project_manifest_path: str,
    smiles_column: str,
    vocab_filename: str,
    add_coverage_column: bool = True,
    output_filename: Optional[str] = None,
    explanation: str = "Dataset with vocab fit flags"
) -> Dict:
    """
    Flag SMILES in dataset based on whether they fit in the vocabulary.
    
    MCP-friendly version that loads vocab from JSON resource and optionally
    adds a status column to the dataset with values:
    - 'passed': All tokens in vocabulary
    - 'failed': Some tokens missing and no <unk> token in vocab
    - 'unknown token': Some tokens missing but vocab has <unk> token
    
    Args:
        input_filename: CSV dataset resource filename
        project_manifest_path: Path to project manifest.json
        smiles_column: Column name containing SMILES strings
        vocab_filename: Vocab JSON resource filename
        add_coverage_column: Add 'vocab_status' string column
        output_filename: Name for output dataset (required if add_coverage_column=True)
        explanation: Description for saved dataset
        
    Returns:
        Dictionary containing:
            - n_smiles: Number of SMILES checked
            - n_passed: Number of SMILES fully tokenizable
            - n_failed: Number with missing tokens (no <unk> in vocab)
            - n_unknown_token: Number with missing tokens (vocab has <unk>)
            - overall_coverage: Average coverage percentage
            - missing_tokens: Set of all unknown tokens found
            - missing_token_counts: Dict of unknown token frequencies
            - output_filename: Saved dataset filename (if column added)
            
    Example:
        >>> result = flag_smiles_vocab_fit(
        ...     "molecules.csv",
        ...     "manifest.json",
        ...     "SMILES",
        ...     "my_vocab.json",
        ...     add_coverage_column=True,
        ...     output_filename="molecules_flagged"
        ... )
        >>> # User can then filter based on status:
        >>> # df[df['vocab_status'] == 'passed']  # Only fully covered
        >>> # df[df['vocab_status'].isin(['passed', 'unknown token'])]  # Allow <unk>
    """
    # Load vocab
    vocab, special_tokens = load_vocab_from_json(vocab_filename, project_manifest_path)
    has_unk_token = 'unk' in special_tokens
    
    # Load dataset
    df = _load_resource(project_manifest_path, input_filename)
    
    # Validate column
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. "
            f"Available: {list(df.columns)}"
        )
    
    # Get SMILES list (keep indices for mapping back)
    smiles_indices = df[smiles_column].dropna().index.tolist()
    smiles_list = df.loc[smiles_indices, smiles_column].tolist()
    n_smiles = len(smiles_list)
    
    if n_smiles == 0:
        raise ValueError(f"No valid SMILES in column '{smiles_column}'")
    # Check coverage for each SMILES and build column data
    n_passed = 0
    n_failed = 0
    n_unknown_token = 0
    total_coverage = 0.0
    all_unknown_tokens = Counter()
    
    coverage_status = {}
    
    for idx, smiles in zip(smiles_indices, smiles_list):
        result = _check_smiles_vocab_coverage(smiles, vocab_filename, project_manifest_path, return_unknown=True)
        
        total_coverage += result['coverage']
        
        # Determine status
        if result['can_tokenize']:
            # All tokens in vocab
            status = 'passed'
            n_passed += 1
        elif has_unk_token:
            # Has unknown tokens but vocab has <unk> token
            status = 'unknown token'
            n_unknown_token += 1
        else:
            # Has unknown tokens and no <unk> in vocab
            status = 'failed'
            n_failed += 1
        
        coverage_status[idx] = status
        
        # Count unknown tokens
        for token in result.get('unknown_tokens', []):
            all_unknown_tokens[token] += 1
    
    overall_coverage = total_coverage / n_smiles if n_smiles > 0 else 0.0
    
    # Get unique missing tokens sorted by frequency
    missing_tokens_sorted = sorted(
        all_unknown_tokens.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Prepare result dict
    result_dict = {
        "n_smiles": n_smiles,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_unknown_token": n_unknown_token,
        "overall_coverage": round(overall_coverage, 2),
        "missing_tokens": list(all_unknown_tokens.keys()),
        "missing_token_counts": dict(missing_tokens_sorted),
        "summary": (
            f"{n_passed} passed, {n_unknown_token} with unknown tokens, {n_failed} failed. "
            f"({overall_coverage:.1f}% average coverage). "
            f"Found {len(all_unknown_tokens)} unique unknown tokens."
        )
    }
    
    # Add column to dataset if requested
    if add_coverage_column:
        df_copy = df.copy()
        
        # Initialize column with None, then fill with status strings
        df_copy['vocab_status'] = None
        for idx, status in coverage_status.items():
            df_copy.at[idx, 'vocab_status'] = status
        
        # Save dataset (always create new resource for traceability)
        if output_filename is None:
            raise ValueError(
                "output_filename is required when add_coverage_column=True"
            )
        saved_filename = _store_resource(
            df_copy,
            project_manifest_path,
            output_filename,
            explanation,
            'csv'
        )
        
        result_dict['output_filename'] = saved_filename
        result_dict['column_added'] = 'vocab_status'
    
    return result_dict
