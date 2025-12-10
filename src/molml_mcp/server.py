from mcp.server.fastmcp import FastMCP

# All tools we want to expose via the MCP server
from molml_mcp.infrastructure.resources import get_all_resources_tools

from molml_mcp.tools.core import get_all_dataset_tools
from molml_mcp.tools.cleaning import get_all_cleaning_tools
from molml_mcp.tools.core_mol import get_all_scaffold_tools
from molml_mcp.tools.core_mol.visualize import smiles_to_acs1996_png, smiles_grid_to_acs1996_png
from molml_mcp.tools.core_mol.smiles_ops import enumerate_stereo_isomers_smiles
from molml_mcp.tools.core_mol.substructure_matching import get_all_substructure_matching_tools
from molml_mcp.tools.core_mol.data_splitting import random_split_dataset

from molml_mcp.tools.featurization.simple_descriptors import list_rdkit_descriptors, calculate_simple_descriptors
from molml_mcp.tools.featurization.complex_descriptors import get_all_complex_descriptor_tools

# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add supported resource types tool
for tool_func in get_all_resources_tools():
    mcp.add_tool(tool_func)

# Add dataset management tools
for tool_func in get_all_dataset_tools():
    mcp.add_tool(tool_func)

# Add molecular cleaning tools
for tool_func in get_all_cleaning_tools():
    mcp.add_tool(tool_func)

# Add scaffold tools
for tool_func in get_all_scaffold_tools():
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

