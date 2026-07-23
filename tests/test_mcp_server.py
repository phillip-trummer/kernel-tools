import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MCP_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("mcp", "trio")
)


@unittest.skipUnless(MCP_AVAILABLE, "MCP SDK with Trio transport is not installed")
class MCPServerIntegrationTests(unittest.TestCase):
    def test_stdio_handshake_and_tool_listing(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        repo = Path(__file__).resolve().parent.parent

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / ".state").mkdir()
                (workspace / ".state" / "tree.json").write_text(json.dumps({}))
                (workspace / "src").mkdir()
                (workspace / "src" / "kernel.cu").write_text("// kernel\n")
                parameters = StdioServerParameters(
                    command=sys.executable,
                    args=[
                        str(repo / "scripts" / "mcp_server.py"),
                        "--workspace",
                        str(workspace),
                        "--config",
                        str(repo / "config.toml"),
                    ],
                    env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                )
                async with stdio_client(parameters) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        response = await session.list_tools()
                        names = {tool.name for tool in response.tools}
                        self.assertIn("benchmark_kernel", names)
                        self.assertIn("log_experiment", names)
                        self.assertNotIn("bash", names)
                        result = await session.call_tool("read_source", {})
                        self.assertIn("kernel.cu", result.content[0].text)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
