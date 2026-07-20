from pathlib import Path
from typing import Optional

from tools.registry import registry
from tools._workspace import SRC_DIR, resolve_experiment_dir

# Match the internal read tool: cap output unless a range is requested, and
# truncate very long lines so a single line can't blow up the context.
DEFAULT_LIMIT = 2000
MAX_LINE_LENGTH = 2000

SCHEMA = {
    "name": "read_source",
    "description": (
        "Read a source file by name — the working kernel may be one or "
        "several files. Call with no filename to list all source files. Pass "
        "experiment_id to read the same file from a previously logged "
        "experiment instead, without disturbing the working tree.\n\n"
        "Reads up to 2000 lines from the start by default. When you already "
        "know which part of the file you need, pass offset and limit to read "
        "just that range. Output is prefixed with line numbers in "
        "`<lineno>\\t<content>` form; lines longer than 2000 characters are "
        "truncated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of a source file, as listed when you call this with no filename. Omit to list all source files.",
            },
            "experiment_id": {
                "type": "string",
                "description": "Optional. Read from a previously logged experiment (e.g. 'v2_tiled') instead of the working tree.",
            },
            "offset": {
                "type": "integer",
                "description": "Optional. The line number to start reading from.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional. The number of lines to read.",
            },
        },
    },
}


@registry.register(SCHEMA)
def read_source(
    filename: Optional[str] = None,
    experiment_id: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    # Resolve the base directory: working src/ or an experiment snapshot.
    if experiment_id:
        base = resolve_experiment_dir(experiment_id)
        if not isinstance(base, Path):
            return f"Error: {base}"
    else:
        base = SRC_DIR.resolve()

    # No filename: list the source files making up the working kernel.
    if filename is None:
        available = sorted(p.name for p in base.iterdir() if p.is_file()) if base.is_dir() else []
        where = f"experiment {experiment_id!r}" if experiment_id else "working kernel"
        return f"Source files in {where}: {available}"

    # Resolve the file inside that base.
    file_path = (base / filename).resolve()
    if base not in file_path.parents:
        return f"Error: {filename!r} is not a valid source filename."
    if not file_path.is_file():
        available = sorted(p.name for p in base.iterdir() if p.is_file()) if base.is_dir() else []
        where = f"experiment {experiment_id!r}" if experiment_id else "working source"
        return f"Error: {filename!r} not found in {where}. Available source files: {available}"

    # Read and number-prefix the output.
    lines = file_path.read_text().splitlines()
    if not lines:
        where = f"experiment {experiment_id!r}" if experiment_id else "working source"
        return f"({filename!r} in {where} exists but is empty)"

    # Select the requested line range (offset is a 1-based line number).
    start = max(offset, 1) if offset is not None else 1
    count = limit if limit is not None else DEFAULT_LIMIT
    window = lines[start - 1 : start - 1 + count]

    # An offset past EOF yields an empty window; say so instead of returning "".
    if not window:
        where = f"experiment {experiment_id!r}" if experiment_id else "working source"
        return (
            f"(offset {start} is past the end of {filename!r} in {where}, "
            f"which has {len(lines)} lines)"
        )

    out = []
    for i, line in enumerate(window, start):
        if len(line) > MAX_LINE_LENGTH:
            line = line[:MAX_LINE_LENGTH] + "... [truncated]"
        out.append(f"{i:>6}\t{line}")

    # Make a capped read visible: if lines beyond the window remain, say so
    # rather than returning a silently-truncated tail.
    last = start - 1 + len(window)
    if last < len(lines):
        out.append(
            f"... [showing lines {start}-{last} of {len(lines)}; "
            f"pass offset={last + 1} to continue]"
        )
    return "\n".join(out)
