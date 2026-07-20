"""Benchmark adapters — one module per benchmark framework.

Each adapter implements the `BenchmarkAdapter` protocol (tools/_benchmark.py)
for a specific framework, owning that framework's fixtures and native types so
only neutral results cross into the harness. `get_adapter()` in tools/_benchmark.py
selects one by name; to add a benchmark, drop a sibling module here and register
its name there. Framework imports live inside these modules, never above them.
"""
