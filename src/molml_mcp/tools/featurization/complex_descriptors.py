
from molml_mcp.tools.featurization.supported.ecfps import smiles_to_ecfp_dataset


def get_all_complex_descriptor_tools():
    """Return a list of all complex descriptor tools for MCP registration."""
    return [
        smiles_to_ecfp_dataset
    ]

