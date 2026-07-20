from tools.registry import registry
from tools._workspace import SRC_DIR

SCHEMA = {
    "name": "edit_source",
    "description": (
        "Apply a surgical edit to a source file by exact substring "
        "replacement. 'old_string' must match the file content EXACTLY "
        "(including whitespace and indentation) and must be unique in the "
        "file unless 'replace_all' is true. Do NOT include the line-number "
        "prefixes shown by read_source — only the actual source. "
        "For full-file rewrites, prefer write_source: it takes the new "
        "content directly and avoids constructing a large exact-match anchor."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "old_string": {
                "type": "string",
                "description": "Exact text to replace. Must be unique unless replace_all is true.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "filename": {
                "type": "string",
                "description": "Name of the source file to edit (as listed by read_source).",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match.",
                "default": False,
            },
        },
        "required": ["old_string", "new_string", "filename"],
    },
}


@registry.register(SCHEMA)
def edit_source(
    old_string: str,
    new_string: str,
    filename: str,
    replace_all: bool = False,
) -> str:
    # Resolve and validate the target file.
    src_dir = SRC_DIR.resolve()
    file_path = (src_dir / filename).resolve()
    if src_dir not in file_path.parents:
        return f"Error: {filename!r} is not a valid source filename."
    if not file_path.is_file():
        return f"Error: {filename!r} not found."
    if old_string == new_string:
        return "Error: old_string and new_string are identical; nothing to do."

    # Find the match.
    contents = file_path.read_text()
    count = contents.count(old_string)
    if count == 0:
        return "Error: old_string not found. Re-read the file and copy the text exactly (no line-number prefixes)."
    if count > 1 and not replace_all:
        return (
            f"Error: old_string matches {count} locations. Add surrounding "
            "context to make it unique, or pass replace_all=true."
        )

    # Apply the replacement.
    new_contents = contents.replace(old_string, new_string, -1 if replace_all else 1)
    file_path.write_text(new_contents)
    n = count if replace_all else 1
    return f"Replaced {n} occurrence(s) in {filename}."
