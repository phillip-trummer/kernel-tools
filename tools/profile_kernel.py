"""Run the kernel in src/ under Nsight Compute (ncu).

The registered (outer) function shells out to
``ncu ... python <this file> <representative_workload>``. The ``__main__`` block
at the bottom is the ncu-launched child: it asks the adapter to build the kernel
and materialize one representative workload's inputs, warms up, then runs
measured iterations inside cudaProfilerStart/Stop.
"""

import shutil
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # When ncu invokes this file as a script, sys.path[0] is the file's directory,
    # so `tools.*` is not importable. Add the repo root for the child process only.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.registry import registry
from tools._workloads import REPRESENTATIVE_WORKLOAD_LABELS
from tools._benchmark import get_adapter


_WARMUP_ITERS = 5
_PROFILE_ITERS = 1
_MAX_OUTPUT_BYTES = 50_000


SCHEMA = {
    "name": "profile_kernel",
    "description": (
        "Profile the current working kernel with Nsight Compute, or introspect ncu. "
        "Pass representative_workload to profile one concrete representative workload "
        "(small, medium, large, or xlarge) and return ncu's report verbatim. "
        "Omit representative_workload to run ncu with no target "
        "— for informational flags like `--help`, `--list-sets`, `--list-sections`, "
        "`--query-metrics`. ncu_args are forwarded as-is to ncu. "
        f"Output is capped at ~{_MAX_OUTPUT_BYTES // 1000} KB; if truncated, narrow "
        "the run with ncu flags."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "representative_workload": {
                "type": "string",
                "enum": list(REPRESENTATIVE_WORKLOAD_LABELS),
                "description": (
                    "Which representative workload to profile. "
                ),
            },
            "ncu_args": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": (
                    "Flags forwarded verbatim to ncu. "
                ),
            },
        },
    },
}


@registry.register(SCHEMA)
def profile_kernel(
    representative_workload: str | None = None,
    ncu_args: list[str] | None = None,
) -> str:
    if shutil.which("ncu") is None:
        return "Error: ncu (Nsight Compute) not found on PATH"

    adapter = None
    if representative_workload is None:
        # Introspection mode: run ncu with no target app (e.g. --help, --list-sets).
        cmd = ["ncu", *(ncu_args or [])]
    else:
        # Validate input, then re-enter this file under ncu.
        if representative_workload not in REPRESENTATIVE_WORKLOAD_LABELS:
            valid = ", ".join(REPRESENTATIVE_WORKLOAD_LABELS)
            return (
                f"Error: invalid representative_workload "
                f"{representative_workload!r}; expected one of: {valid}"
            )
        # Pre-build so the ncu child reuses the on-disk artifact instead of
        # recompiling under instrumentation. Best-effort: on failure the child
        # recompiles and reports it. The adapter is reused below to strip build
        # chatter from the report.
        adapter = get_adapter()
        try:
            adapter.prewarm()
        except Exception:
            pass
        cmd = [
            "ncu",
            "--target-processes", "all",
            "--profile-from-start", "off",
            *(ncu_args or []),
            sys.executable, __file__, representative_workload,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # On a profiled success, strip the backend's build chatter so the report
    # dominates. Otherwise keep everything verbatim — a failed run's build log is
    # the diagnostic, and introspection output has no chatter to strip.
    if result.returncode == 0 and adapter is not None:
        out = adapter.strip_build_noise(result.stdout)
        err = adapter.strip_build_noise(result.stderr)
        if err.strip():
            out += "\n--- stderr ---\n" + err
    else:
        out = result.stdout
        if result.stderr:
            out += "\n--- stderr ---\n" + result.stderr
    return _tail_cap(out, _MAX_OUTPUT_BYTES)


def _tail_cap(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    dropped = len(data) - limit
    kept = data[-limit:].decode("utf-8", errors="replace")
    return (
        f"[truncated: dropped {dropped} bytes from start of output; "
        f"narrow ncu_args (--set basic, --section, -k, --launch-count) "
        f"to fit under {limit // 1000} KB]\n" + kept
    )


def _run_under_ncu(representative_workload: str) -> None:
    import torch

    # Build the kernel + materialize the chosen representative workload's inputs.
    # Same kernel identity as benchmark_kernel, so the adapter's build cache is shared.
    runnable, inputs = get_adapter().build_profilable(representative_workload)

    with torch.no_grad():
        # Warm up outside the profiled region.
        for _ in range(_WARMUP_ITERS):
            runnable(*inputs)
        torch.cuda.synchronize()

        # Profile.
        torch.cuda.profiler.start()
        for _ in range(_PROFILE_ITERS):
            runnable(*inputs)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    _run_under_ncu(sys.argv[1])
