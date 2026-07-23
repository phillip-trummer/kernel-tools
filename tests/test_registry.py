import unittest

from tools.registry import LocalToolRegistry, select_schemas, validate_enabled


SCHEMAS = [
    {"name": "one", "description": "", "input_schema": {"type": "object"}},
    {"name": "two", "description": "", "input_schema": {"type": "object"}},
]


class RegistryTests(unittest.TestCase):
    def test_allowlist_preserves_schema_order(self):
        selected = select_schemas(SCHEMAS, ["two"])
        self.assertEqual([item["name"] for item in selected], ["two"])

    def test_unknown_allowlist_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            validate_enabled(SCHEMAS, ["typo"])

    def test_dispatch_is_transport_neutral(self):
        registry = LocalToolRegistry()

        @registry.register(SCHEMAS[0])
        def one(value=1):
            return value + 1

        self.assertEqual(registry.dispatch("one", value=4), 5)
        self.assertIn("Unknown tool", registry.dispatch("missing"))


if __name__ == "__main__":
    unittest.main()
