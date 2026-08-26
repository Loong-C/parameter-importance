"""Compatibility entry point for the explicit S2.6 GPU inventory producer."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep both direct-script and ``python -m ops.stage2...`` invocation forms
# independent of the caller's PYTHONPATH.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from produce_s206_formal_inputs import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(["gpu-inventory", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
