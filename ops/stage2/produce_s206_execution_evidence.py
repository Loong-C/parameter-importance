"""Compatibility entry point for the S2.6 execution-evidence producer."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from produce_s206_formal_inputs import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main(["execution-evidence", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
