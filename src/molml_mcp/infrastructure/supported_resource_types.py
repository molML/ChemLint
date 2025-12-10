from pathlib import Path
from typing import Any


TYPE_REGISTRY: dict[str, dict[str, Any]] = {}
# Supported resource types and their handlers:
# csv
# model
# json
# png
# joblib
# feature_vectors (joblib format)


# csv
def _save_csv(obj, path: Path):
    # assume obj is a pandas DataFrame
    import pandas as pd
    assert hasattr(obj, "to_csv"), "csv type expects a DataFrame-like object"
    obj.to_csv(path, index=False)

def _load_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path)

TYPE_REGISTRY["csv"] = {
    "ext": ".csv",
    "save": _save_csv,
    "load": _load_csv,
}


# model
def _save_model(model, path: Path):
    import joblib
    joblib.dump(model, path)

def _load_model(path: Path):
    import joblib
    return joblib.load(path)

TYPE_REGISTRY["model"] = {
    "ext": ".pkl",
    "save": _save_model,
    "load": _load_model,
}


# json
def _save_json(obj, path: Path):
    import json
    with open(path, "w") as f:
        json.dump(obj, f)

def _load_json(path: Path):
    import json
    with open(path, "r") as f:
        return json.load(f) 

TYPE_REGISTRY["json"] = {
    "ext": ".json",
    "save": _save_json,
    "load": _load_json,
}


# png (images)
def _save_png(png_bytes: bytes, path: Path):
    """Save PNG image bytes to file."""
    assert isinstance(png_bytes, bytes), "png type expects bytes"
    with open(path, "wb") as f:
        f.write(png_bytes)

def _load_png(path: Path) -> bytes:
    """Load PNG image bytes from file."""
    with open(path, "rb") as f:
        return f.read()
    
TYPE_REGISTRY["png"] = {
    "ext": ".png",
    "save": _save_png,
    "load": _load_png,
}
    

def _save_joblib(obj, path: Path):
    """Save object using joblib."""
    import joblib
    joblib.dump(obj, path, compress=3)

def _load_joblib(path: Path):
    """Load object using joblib."""
    import joblib
    return joblib.load(path)

TYPE_REGISTRY["joblib"] = {
    "ext": ".joblib",
    "save": _save_joblib,
    "load": _load_joblib,
}


def _save_feature_vectors(obj, path: Path):
    """Save feature_vectors {id: np.ndarray} using joblib."""
    import joblib
    joblib.dump(obj, path, compress=3)

def _load_feature_vectors(path: Path):
    """Load feature_vectors {id: np.ndarray} using joblib."""
    import joblib
    return joblib.load(path)

TYPE_REGISTRY["feature_vectors"] = {
    "ext": ".joblib",
    "save": _save_feature_vectors,
    "load": _load_feature_vectors,
}
