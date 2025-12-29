"""
Report generation tools for molecular datasets.
"""

from molml_mcp.tools.reports.scaffold_report import generate_scaffold_report
from molml_mcp.tools.reports.quality import generate_quality_report


def get_all_report_tools():
    """
    Returns a list of MCP-exposed report generation functions for server registration.
    """
    return [
        generate_scaffold_report,
        generate_quality_report,
    ]


__all__ = [
    "generate_scaffold_report",
    "generate_quality_report",
    "get_all_report_tools",
]
