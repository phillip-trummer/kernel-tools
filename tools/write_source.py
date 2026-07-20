from tools.registry import registry
from tools._workspace import SRC_DIR

SCHEMA = {
    "name": "write_source",
    "description": (
        "Replace the entire contents of a source file with new content. "
        "Use this for full-file rewrites. For surgical changes to a small "
        "region, prefer edit_source — its exact-match anchor protects "
        "unrelated content from being clobbered."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Full new contents of the file. Written verbatim.",
            },
            "filename": {
                "type": "string",
                "description": "Name of the source file to overwrite (as listed by read_source).",
            },
        },
        "required": ["content", "filename"],
    },
}


@registry.register(SCHEMA)
def write_source(content: str, filename: str) -> str:
    src_dir = SRC_DIR.resolve()
    file_path = (src_dir / filename).resolve()
    if src_dir not in file_path.parents:
        return f"Error: {filename!r} is not a valid source filename."
    if not file_path.is_file():
        available = sorted(p.name for p in src_dir.iterdir() if p.is_file()) if src_dir.is_dir() else []
        return (
            f"Error: {filename!r} not found. write_source overwrites existing "
            f"source files only. Available: {available}"
        )

    old_contents = file_path.read_text()
    if old_contents == content:
        return f"No change: {filename} already matches the given content."

    file_path.write_text(content)
    n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    return f"Wrote {filename}: {len(content)} bytes, {n_lines} lines."
