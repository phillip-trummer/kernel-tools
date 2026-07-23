import json
import tempfile
import unittest
from pathlib import Path

from scripts.setup_workspace import (
    CLIENT_INSTRUCTIONS,
    _parse_args,
    _representative_workloads_from_config,
    _server_command,
    _write_claude_settings,
    _write_client_instructions,
    _write_mcp_config,
)


class ClientBootstrapTests(unittest.TestCase):
    def test_representative_workloads_are_loaded_by_name_and_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            workloads = Path(tmp) / "workloads.jsonl"
            workloads.write_text(
                "\n".join(
                    json.dumps({"workload": {"uuid": workload_uuid}})
                    for workload_uuid in ("s", "m", "l", "xl")
                )
            )
            task_cfg = {
                "representative_workloads": [
                    {"name": "xlarge", "uuid": "xl"},
                    {"name": "small", "uuid": "s"},
                    {"name": "large", "uuid": "l"},
                    {"name": "medium", "uuid": "m"},
                ]
            }

            configured = _representative_workloads_from_config(task_cfg, workloads)

            self.assertEqual(
                configured,
                {
                    "small": "s",
                    "medium": "m",
                    "large": "l",
                    "xlarge": "xl",
                },
            )

    def test_baseline_benchmark_is_enabled_by_default(self):
        self.assertFalse(_parse_args([]).skip_baseline_benchmark)

    def test_baseline_benchmark_can_be_skipped(self):
        self.assertTrue(
            _parse_args(["--skip-baseline-benchmark"]).skip_baseline_benchmark
        )

    def test_server_command_pins_workspace_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            command = _server_command(root, workspace)
            self.assertEqual(command[2:4], ["--workspace", str(workspace.resolve())])
            self.assertEqual(command[4], "--config")
            self.assertEqual(command[5], str((root / "config.toml").resolve()))

    def test_claude_mcp_config_matches_server_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_mcp_config(workspace, root)
            payload = json.loads((workspace / ".mcp.json").read_text())
            server = payload["mcpServers"]["kernel-tools"]
            command = _server_command(root, workspace)
            self.assertEqual(server["command"], command[0])
            self.assertEqual(server["args"], command[1:])

    def test_claude_settings_deny_all_direct_file_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write_claude_settings(workspace, ["read_journal", "edit_source"])
            payload = json.loads(
                (workspace / ".claude" / "settings.local.json").read_text()
            )
            self.assertEqual(
                payload["permissions"]["deny"],
                ["Read(**)", "Edit(**)", "Bash(*)"],
            )

    def test_both_clients_receive_same_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write_client_instructions(workspace)
            self.assertEqual((workspace / "AGENTS.md").read_text(), CLIENT_INSTRUCTIONS)
            self.assertEqual((workspace / "CLAUDE.md").read_text(), CLIENT_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
