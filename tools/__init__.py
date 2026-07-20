"""Register the complete MCP kernel-tool surface on import."""

from tools.registry import registry
from tools import (
    annotate_journal,
    benchmark_kernel,
    checkout_experiment,
    diff_experiment,
    edit_source,
    log_experiment,
    profile_kernel,
    read_journal,
    read_source,
    write_source,
)

__all__ = ["registry"]
