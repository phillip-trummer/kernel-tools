# tools/registry.py
from typing import Any, Callable

# Canonical MCP server name. Single source of truth for the `mcp__<server>__*`
# tool prefix, shared by the .mcp.json key and the server (mcp_server.py).
MCP_SERVER_NAME = "kernel-tools"


class LocalToolRegistry:
    """A mock MCP registry. Tools self-register via @registry.register(SCHEMA),
    and the agent loop calls dispatch(name, **input) without knowing what tools
    exist. Tools operate on the process cwd; the agent-loop entry point (and an
    MCP server) is responsible for os.chdir-ing into the workspace at startup."""

    def __init__(self):
        self.schemas: list[dict] = []
        self.handlers: dict[str, Callable] = {}

    def register(self, schema: dict):
        """Decorator that mirrors the future @mcp.tool() entry point."""
        def decorator(func: Callable) -> Callable:
            name = schema["name"]
            self.schemas.append(schema)
            self.handlers[name] = func
            return func
        return decorator

    def dispatch(self, tool_name: str, **kwargs) -> Any:
        handler = self.handlers.get(tool_name)
        if not handler:
            return f"Error: Unknown tool {tool_name}"
        return handler(**kwargs)


registry = LocalToolRegistry()


def select_schemas(schemas: list[dict], enabled: list[str] | None) -> list[dict]:
    """Filter advertised tool schemas to a config allowlist; None/empty = all.
    The harness applies this when building the agent-facing tool list, so tool
    selection is a startup choice without any per-tool logic in the registry."""
    return schemas if not enabled else [s for s in schemas if s["name"] in enabled]


def validate_enabled(schemas: list[dict], enabled: list[str] | None) -> None:
    """Reject misspelled tool names instead of silently hiding them."""
    if not enabled:
        return
    available = {schema["name"] for schema in schemas}
    unknown = sorted(set(enabled) - available)
    if unknown:
        raise ValueError(
            f"unknown tool(s) in [tools] enabled: {', '.join(unknown)}; "
            f"available: {', '.join(sorted(available))}"
        )
