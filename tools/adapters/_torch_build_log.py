"""Shared cleanup for torch cpp_extension + ninja build logs.

Both adapters compile CUDA through torch's `cpp_extension`, so the *noise* is a
property of that toolchain, not of the benchmark: ninja's step headers, the
compiler command echoed again under `FAILED:`, glog chatter, the Python traceback
torch wraps a compiler error in, and the build directory it compiled from. Only
that vocabulary lives here.

What stays with each adapter: which stream carries the diagnostic, how a native
status maps onto the neutral taxonomy, and how the correctness gate is worded.
The gates genuinely differ — flashinfer fails a point only when it exceeds both
tolerances, SOL uses an allclose bound plus a matched-element ratio — so sharing
that sentence would be a bug, not a saving.
"""
from __future__ import annotations

import re


LOG_TAIL_LINES = 40

# ninja's own step header. Narrow on purpose: a bare "[0] kernel(...)" is a
# profiler's kernel banner, not build chatter.
BUILD_STEP_RE = re.compile(r"^\[\d+/\d+\]\s+")

# Compile logs append Python exception chains after the real compiler diagnostic.
_TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):")
_PYTHON_EXCEPTION_WRAPPER_RE = re.compile(
    r"^(BuildError:|RuntimeError: Error building extension|Traceback:)"
)
# Anchor compile-error tails on the real diagnostic.
_COMPILER_DIAG_RE = re.compile(
    r"\berror\s*:|catastrophic error|fatal error|"
    r"\.(?:cu|cuh|cpp|cc|cxx|c|h|hpp)[:(]\d+|"
    r"error detected in the compilation|error limit reached|"
    r"\bFAILED:|ninja: build stopped",
    re.IGNORECASE,
)
# absl/glog-style framework chatter.
_WORKER_LOG_RE = re.compile(r"^[WIE]\d{4} \d")
_NOISY_LOG_PREFIXES = (
    "nvcc warning : incompatible redefinition for option 'compiler-bindir'",
    "Builder TorchBuilder built ",
)

# The directory a source file was compiled from — a build cache or a staging temp
# dir. The agent knows its sources by bare filename, and both are internal paths
# it must never see.
_BUILD_PATH_RE = re.compile(
    r"(?:/[\w.+-]+)+/(?=[\w.+-]+\.(?:cu|cuh|cpp|cc|cxx|c|h|hpp|py|o|so)\b)"
)


def strip_build_paths(text: str) -> str:
    """Reduce compiled-from paths to bare filenames: a diagnostic naming
    `/home/.../cache/torch/torch_candidate_7c99.../kernel.cu(586)` becomes
    `kernel.cu(586)`. Line and column numbers are untouched."""
    return _BUILD_PATH_RE.sub("", text)


def strip_build_noise(text: str) -> str:
    """Drop build-system chatter from externally captured output (e.g. a
    profiler's), so its report dominates. Every other line passes through."""
    kept = [
        line
        for line in text.splitlines()
        if not BUILD_STEP_RE.match(line)
        and "cpp_extension.py" not in line
        and "TORCH_CUDA_ARCH_LIST" not in line
        and line.strip() != "ninja: no work to do."
        and not (line.strip().startswith("Builder ") and " built " in line)
    ]
    return "\n".join(kept).strip("\n")


def denoise(text: str, *, compile_error: bool = False, lines: int = LOG_TAIL_LINES) -> str:
    """Turn a raw build/eval log into the diagnostic tail the agent reads: drop
    the toolchain's chatter, focus a compile failure on the compiler's own
    diagnostic, keep the last `lines`, and strip build paths."""
    raw = text.strip().splitlines()
    kept = [line for line in raw if not _is_noisy(line, _build_commands(raw))]
    if compile_error:
        kept = _focus_compiler_diagnostics(kept)
    return strip_build_paths("\n".join(kept[-lines:]))


def _build_commands(lines: list[str]) -> set[str]:
    """The compiler commands ninja announces, so the copy it echoes again under
    `FAILED:` can be recognized and dropped."""
    return {
        stripped[m.end():].strip()
        for line in lines
        if (m := BUILD_STEP_RE.match(stripped := line.strip()))
    }


def _is_noisy(line: str, build_commands: set[str]) -> bool:
    stripped = line.strip()
    if BUILD_STEP_RE.match(stripped) is not None:
        return True
    if stripped in build_commands:  # compiler command re-echoed after a FAILED step
        return True
    if _WORKER_LOG_RE.match(stripped) is not None:
        return True
    return any(stripped.startswith(prefix) for prefix in _NOISY_LOG_PREFIXES)


def _focus_compiler_diagnostics(lines: list[str]) -> list[str]:
    cut = next((i for i, line in enumerate(lines) if _TRACEBACK_RE.match(line.strip())), None)
    head = lines[:cut] if cut else lines  # cut==0 (traceback at top) -> keep all
    anchor = next(
        (i for i, line in enumerate(head) if _COMPILER_DIAG_RE.search(line.strip())), None
    )
    focused = head[max(0, anchor - 2):] if anchor is not None else head
    return [line for line in focused if not _PYTHON_EXCEPTION_WRAPPER_RE.match(line.strip())]
