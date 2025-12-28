from mcp.server.fastmcp import FastMCP

# All tools we want to expose via the MCP server
from molml_mcp.infrastructure.resources import get_all_resources_tools

from molml_mcp.tools.core import (
    get_all_dataset_tools,
    filter_by_property_range,
    filter_by_lipinski_ro5,
    filter_by_veber_rules,
    filter_by_pains,
    filter_by_lead_likeness,
    filter_by_rule_of_three,
    filter_by_qed,
    filter_by_scaffold,
    filter_by_functional_groups,
    reduce_dimensions_pca,
    reduce_dimensions_tsne
)
from molml_mcp.tools.cleaning import get_all_cleaning_tools, find_duplicates_dataset, deduplicate_dataset
from molml_mcp.tools.core_mol import get_all_scaffold_tools, get_all_complexity_tools
from molml_mcp.tools.core_mol.visualize import smiles_to_acs1996_png, smiles_grid_to_acs1996_png
from molml_mcp.tools.core_mol.smiles_ops import enumerate_stereo_isomers_smiles
from molml_mcp.tools.core_mol.substructure_matching import get_all_substructure_matching_tools
from molml_mcp.tools.core_mol.data_splitting import random_split_dataset, scaffold_split_dataset, stratified_split_dataset, cluster_based_split_dataset
from molml_mcp.tools.core_mol.similarity import (
    compute_similarity_matrix,
    find_k_nearest_neighbors,
    add_similarity_statistics_dataset
)

from molml_mcp.tools.featurization.SMILES_encoding import get_all_smiles_encoding_tools
from molml_mcp.tools.featurization.simple_descriptors import list_rdkit_descriptors, calculate_simple_descriptors
from molml_mcp.tools.featurization.complex_descriptors import get_all_complex_descriptor_tools
from molml_mcp.tools.ml import get_all_ml_tools

# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add supported resource types tool
for tool_func in get_all_resources_tools():
    mcp.add_tool(tool_func)

# Add dataset management tools
for tool_func in get_all_dataset_tools():
    mcp.add_tool(tool_func)

# Add filtering tools
mcp.add_tool(filter_by_property_range)
mcp.add_tool(filter_by_lipinski_ro5)
mcp.add_tool(filter_by_veber_rules)
mcp.add_tool(filter_by_pains)
mcp.add_tool(filter_by_lead_likeness)
mcp.add_tool(filter_by_rule_of_three)
mcp.add_tool(filter_by_qed)
mcp.add_tool(filter_by_scaffold)
mcp.add_tool(filter_by_functional_groups)

# Add dimensionality reduction tools
mcp.add_tool(reduce_dimensions_pca)
mcp.add_tool(reduce_dimensions_tsne)

# Add molecular cleaning tools
for tool_func in get_all_cleaning_tools():
    mcp.add_tool(tool_func)

# Add deduplication tools
mcp.add_tool(find_duplicates_dataset)
mcp.add_tool(deduplicate_dataset)

# Add scaffold tools
for tool_func in get_all_scaffold_tools():
    mcp.add_tool(tool_func)

# Add complexity tools
for tool_func in get_all_complexity_tools():
    mcp.add_tool(tool_func)

# Add substructure matching tools
for tool_func in get_all_substructure_matching_tools():
    mcp.add_tool(tool_func)

# Add stereoisomer smiles_ops tool
mcp.add_tool(enumerate_stereo_isomers_smiles)

# Add descriptor tools
mcp.add_tool(list_rdkit_descriptors)
mcp.add_tool(calculate_simple_descriptors)
for tool_func in get_all_complex_descriptor_tools():
    mcp.add_tool(tool_func)

# Add visualization tools
mcp.add_tool(smiles_to_acs1996_png)
mcp.add_tool(smiles_grid_to_acs1996_png)

# Add data splitting tool
mcp.add_tool(random_split_dataset)
mcp.add_tool(scaffold_split_dataset)
mcp.add_tool(stratified_split_dataset)
mcp.add_tool(cluster_based_split_dataset)

# Add similarity tools
mcp.add_tool(compute_similarity_matrix)
mcp.add_tool(find_k_nearest_neighbors)
mcp.add_tool(add_similarity_statistics_dataset)

# Add clustering tools
from molml_mcp.tools.clustering.clust import get_all_clustering_tools
for tool_func in get_all_clustering_tools():
    mcp.add_tool(tool_func)

# Add plotting tools
from molml_mcp.tools.core.plotting import plot_histogram, plot_scatter
mcp.add_tool(plot_histogram)
mcp.add_tool(plot_scatter)

# Add ML metrics tools
for tool_func in get_all_ml_tools():
    mcp.add_tool(tool_func)

# add SMILES encoding tools
for tool_func in get_all_smiles_encoding_tools():
    mcp.add_tool(tool_func)
