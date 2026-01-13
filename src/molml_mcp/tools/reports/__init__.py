"""
Report generation tools for molecular datasets.
"""

from molml_mcp.tools.reports.scaffold_analysis import scaffold_analysis
from molml_mcp.tools.reports.data_quality_analysis import data_quality_analysis
from molml_mcp.tools.reports.data_split_analysis import data_split_quality_analysis


def get_all_report_tools():
    """
    Returns a list of MCP-exposed report generation functions for server registration.
    """
    return [
        scaffold_analysis,
        data_quality_analysis,
        data_split_quality_analysis,
    ]


__all__ = [
    "scaffold_analysis",
    "data_quality_analysis",
    "data_split_quality_analysis",
    "get_all_report_tools",
]
