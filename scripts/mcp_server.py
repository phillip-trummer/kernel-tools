#!/usr/bin/env python3
"""Serve the hand-written kernel-tool schemas over MCP using stdio transport."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from pathlib import Path

import mcp.types as types
import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Put the repo on sys.path so `import tools` works when this is invoked as
# `python scripts/mcp_server.py`, whatever the cwd.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import registry  # noqa: E402  (importing registers every tool)
from tools.registry import MCP_SERVER_NAME, select_schemas, validate_enabled  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Initialized kernel workspace (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "config.toml",
        help="Harness config containing the tool allowlist.",
    )
    return parser.parse_args()


ARGS = _parse_args()
WORKSPACE = ARGS.workspace.resolve()
CONFIG = ARGS.config.resolve()


def _debug(message: str) -> None:
    if os.environ.get("KERNEL_MCP_DEBUG"):
        logging.basicConfig(level=logging.DEBUG)
        print(f"[kernel-tools] {message}", file=sys.stderr, flush=True)


if not (WORKSPACE / ".state" / "tree.json").is_file():
    raise SystemExit(
        f"Error: {WORKSPACE} is not initialized (missing .state/tree.json). "
        "Run scripts/setup_workspace.py first."
    )
if not CONFIG.is_file():
    raise SystemExit(f"Error: config not found: {CONFIG}")

# Tools intentionally resolve the workspace through cwd. Make that explicit
# instead of relying on the MCP client's inherited working directory.
os.chdir(WORKSPACE)
_debug(f"workspace={WORKSPACE} config={CONFIG}")

# Tool exposure (an ablated variable) stays a repo-level config choice.
with CONFIG.open("rb") as f:
    _cfg = tomllib.load(f)
_ENABLED = _cfg.get("tools", {}).get("enabled")
try:
    validate_enabled(registry.schemas, _ENABLED)
except ValueError as exc:
    raise SystemExit(f"Error: {exc}") from exc

server = Server(MCP_SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=s["name"],
            description=s["description"],
            inputSchema=s["input_schema"],
        )
        for s in select_schemas(registry.schemas, _ENABLED)
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    # Mirror agent_loop's per-tool error handling: dispatch handles unknown
    # tools, and any handler exception becomes an "Error: ..." string the
    # model sees as the tool result rather than a transport failure.
    try:
        output = registry.dispatch(name, **(arguments or {}))
    except Exception as e:
        output = f"Error: {type(e).__name__}: {e}"
    return [types.TextContent(type="text", text=str(output))]


async def _main() -> None:
    _debug("opening stdio transport")
    async with stdio_server() as (read_stream, write_stream):
        _debug("serving requests")
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    anyio.run(_main, backend="trio")
