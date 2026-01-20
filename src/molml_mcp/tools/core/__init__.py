"""Core tools package - domain-agnostic dataset and utility operations."""

from molml_mcp.tools.core.dataset_ops import (
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
from molml_mcp.tools.core.outliers import (
    get_all_outlier_detection_tools,
)
from molml_mcp.tools.core.plotting import (
    add_molecular_scatter_plot,
    add_histogram,
    add_density_plot,
    add_box_plot,
    remove_plot,
    list_active_plots,
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
    'add_molecular_scatter_plot',
    'remove_plot',
    'list_active_plots',
    'add_histogram',
    'add_density_plot',
    'add_box_plot',
]
