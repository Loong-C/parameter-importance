from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .storage import StorageLayout, is_within


def runtime_cache_environment(
    layout: StorageLayout,
    *,
    run_id: str,
    attempt_id: str,
    session_id: str,
) -> dict[str, str]:
    cache = layout.path("cache")
    temporary = layout.path("tmp", "runs", run_id, attempt_id, session_id)
    values = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "XDG_CACHE_HOME": cache / "xdg",
        "TMPDIR": temporary,
        "TMP": temporary,
        "TEMP": temporary,
    }
    return {key: str(value) for key, value in values.items()}


def validate_runtime_cache_environment(
    environment: Mapping[str, str], layout: StorageLayout
) -> list[str]:
    failures: list[str] = []
    for name, value in runtime_cache_environment(
        layout, run_id="fixture", attempt_id="attempt-000001", session_id="fixture"
    ).items():
        actual = environment.get(name)
        if not actual:
            failures.append(f"missing:{name}")
        elif not is_within(Path(actual), layout.root):
            failures.append(f"outside_data_root:{name}:{actual}")
    return failures
