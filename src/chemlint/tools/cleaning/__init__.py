"""
Cleaning tools for molecular machine learning datasets.

This module consolidates all cleaning functionality:
- mol_cleaning: SMILES cleaning, standardization, validation
- label_cleaning: Label conversion and processing
- deduplication: Duplicate entry detection and removal (planned)
"""

# Molecular cleaning exports
from .mol_cleaning import (
    get_SMILES_standardization_guidelines,
    default_SMILES_standardization_pipeline,
    default_SMILES_standardization_pipeline_dataset,
    canonicalize_smiles,
    canonicalize_smiles_dataset,
    remove_salts,
    remove_salts_dataset,
    remove_common_solvents,
    remove_common_solvents_dataset,
    defragment_smiles,
    defragment_smiles_dataset,
    neutralize_smiles,
    neutralize_smiles_dataset,
    standardize_stereochemistry,
    standardize_stereochemistry_dataset,
    remove_isotopes,
    remove_isotopes_dataset,
    canonicalize_tautomers,
    canonicalize_tautomers_dataset,
    normalize_functional_groups,
    normalize_functional_groups_dataset,
    reionize_smiles,
    reionize_smiles_dataset,
    disconnect_metals_smiles,
    disconnect_metals_smiles_dataset,
    validate_smiles,
    validate_smiles_dataset,
    check_smiles_for_pains,
    check_smiles_for_pains_dataset,
    get_all_cleaning_tools,
)

# Label cleaning exports
from .label_cleaning import (
    continuous_to_binary_labels_dataset,
)

# Deduplication exports
from .deduplication import (
    find_duplicates_dataset,
    deduplicate_dataset,
)

__all__ = [
    # Molecular cleaning
    "get_SMILES_standardization_guidelines",
    "default_SMILES_standardization_pipeline",
    "default_SMILES_standardization_pipeline_dataset",
    "canonicalize_smiles",
    "canonicalize_smiles_dataset",
    "remove_salts",
    "remove_salts_dataset",
    "remove_common_solvents",
    "remove_common_solvents_dataset",
    "defragment_smiles",
    "defragment_smiles_dataset",
    "neutralize_smiles",
    "neutralize_smiles_dataset",
    "standardize_stereochemistry",
    "standardize_stereochemistry_dataset",
    "remove_isotopes",
    "remove_isotopes_dataset",
    "canonicalize_tautomers",
    "canonicalize_tautomers_dataset",
    "normalize_functional_groups",
    "normalize_functional_groups_dataset",
    "reionize_smiles",
    "reionize_smiles_dataset",
    "disconnect_metals_smiles",
    "disconnect_metals_smiles_dataset",
    "validate_smiles",
    "validate_smiles_dataset",
    "check_smiles_for_pains",
    "check_smiles_for_pains_dataset",
    "get_all_cleaning_tools",
    # Label cleaning
    "continuous_to_binary_labels_dataset",
    # Deduplication
    "find_duplicates_dataset",
    "deduplicate_dataset",
]
