"""Core tools package - domain-agnostic dataset and utility operations."""

from molml_mcp.tools.core.dataset_ops import (
    store_csv_as_dataset,
    store_csv_as_dataset_from_text,
    get_dataset_head,
    get_dataset_full,
    get_dataset_summary,
    inspect_dataset_rows,
    drop_from_dataset,
    keep_from_dataset,
    get_all_dataset_tools,
)
from molml_mcp.tools.core.filtering import (
    filter_by_property_range,
    filter_by_lipinski_ro5,
    filter_by_veber_rules,
    filter_by_pains,
    filter_by_lead_likeness,
    filter_by_rule_of_three,
    filter_by_qed,
    filter_by_scaffold,
    filter_by_functional_groups
)
from molml_mcp.tools.core.dim_reduction import (
    reduce_dimensions_pca,
    reduce_dimensions_tsne,
)
from molml_mcp.tools.core.statistics import (
    get_all_statistical_test_tools,
)

__all__ = [
    'store_csv_as_dataset',
    'store_csv_as_dataset_from_text',
    'get_dataset_head',
    'get_dataset_full',
    'get_dataset_summary',
    'inspect_dataset_rows',
    'drop_from_dataset',
    'keep_from_dataset',
    'get_all_dataset_tools',
    'filter_by_property_range',
    'filter_by_lipinski_ro5',
    'filter_by_veber_rules',
    'filter_by_pains',
    'filter_by_lead_likeness',
    'filter_by_rule_of_three',
    'filter_by_qed',
    'filter_by_scaffold',
    'filter_by_functional_groups',
    'reduce_dimensions_pca',
    'reduce_dimensions_tsne',
    'get_all_statistical_test_tools',
]
