from mcp.server.fastmcp import FastMCP

# All tools we want to expose via the MCP server
from molml_mcp.resources.logistics import get_all_resource_functions
from molml_mcp.tools import canonicalize_smiles, canonicalize_smiles_dataset

# create an MCP server 
mcp = FastMCP("molml-mcp") 

# Add resource management tools
for tool_func in get_all_resource_functions():
    mcp.add_tool(tool_func)

# Add tools
mcp.add_tool(canonicalize_smiles)
mcp.add_tool(canonicalize_smiles_dataset)

