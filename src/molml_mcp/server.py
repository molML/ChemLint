# from .tools import get_all_tools

# def create_mcp_server():
#     tools = get_all_tools()
#     # You’d wire these into your actual MCP server implementation here
#     # e.g. mcp.Server(tools=tools)
#     return tools

# # If your MCP runtime expects a global `app` or similar:
# app = create_mcp_server()

from molml_mcp.tools.demo import canonicalize_smiles
from fastmcp import FastMCP

mcp = FastMCP(name="molml-mcp")  

if __name__ == "__main__":
    mcp.run()
