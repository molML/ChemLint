"""Core tools package - domain-agnostic dataset and utility operations."""

from chemlint.tools.core.dataset_ops import (
    import_csv_from_path,
    import_csv_from_text,
    get_dataset_head,
    get_dataset_full,
    get_dataset_summary,
    inspect_dataset_rows,
    drop_from_dataset,
    subset_dataset,
    combine_datasets_horizontal,
    merge_datasets_on_smiles,
    read_txt,
    read_json,
    get_all_dataset_tools,
)
from chemlint.tools.core.filtering import (
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
from chemlint.tools.core.dim_reduction import (
    reduce_dimensions_pca,
    reduce_dimensions_tsne,
)
from chemlint.tools.core.statistics import (
    get_all_statistical_test_tools,
)
from chemlint.tools.core.outliers import (
    get_all_outlier_detection_tools,
)

__all__ = [
    'import_csv_from_path',
    'import_csv_from_text',
    'get_dataset_head',
    'get_dataset_full',
    'get_dataset_summary',
    'inspect_dataset_rows',
    'drop_from_dataset',
    'subset_dataset',
    'combine_datasets_horizontal',
    'merge_datasets_on_smiles',
    'read_txt',
    'read_json',
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
    'get_all_outlier_detection_tools',
]
