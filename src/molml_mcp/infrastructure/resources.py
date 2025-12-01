"""
Internal resource management infrastructure.

This module handles the low-level operations for storing and loading resources
(datasets, models, JSON files, etc.) from the filesystem. These are internal
functions used by client-facing tools but not directly exposed to MCP clients.
"""

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from molml_mcp.infrastructure.supported_resource_types import TYPE_REGISTRY
from molml_mcp.config import DATA_ROOT


def _generate_id(type_tag: str) -> str:
    """Generate a unique resource ID with timestamp, type tag, and random ID component."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    rand = secrets.token_hex(4).upper()  # 8 hex chars

    ext = TYPE_REGISTRY[type_tag]["ext"]
    
    return f"{ts}_{type_tag}_{rand}{ext}"


def _store_resource(obj: Any, type_tag: str) -> str:
    """Store an object as a resource and return its unique resource_id."""
    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unsupported resource type: {type_tag}")
    
    rid = _generate_id(type_tag)

    path = DATA_ROOT / f"{rid}"
    save_fn: Callable[[Any, Path], None] = TYPE_REGISTRY[type_tag]['save']
    save_fn(obj, path)

    return rid


def _load_resource(resource_id: str) -> Any:
    """Load a resource from disk given its resource_id."""
    # infer type from the ID. They are always in the format: TIMESTAMP_TYPE_RANDOM.ext
    try:
        type_tag = resource_id.split("_")[-2]
    except ValueError:
        raise ValueError(f"Invalid resource_id format: {resource_id}")

    if type_tag not in TYPE_REGISTRY:
        raise ValueError(f"Unknown resource type in id: {resource_id}")

    path = DATA_ROOT / f"{resource_id}"

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    load_fn: Callable[[Path], Any] = TYPE_REGISTRY[type_tag]["load"]

    return load_fn(path)
