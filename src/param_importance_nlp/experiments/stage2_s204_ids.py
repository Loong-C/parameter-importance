"""Canonical S2.4 cell identifiers and filesystem projection.

The scientific contract uses ``model:stage``.  Only the filesystem projection
uses ``model__stage``; callers must not invent another spelling.
"""

from __future__ import annotations

EXPECTED_CELL_IDS: tuple[str, ...] = tuple(
    f"{model}:{stage}"
    for model in ("pythia-14m", "pythia-31m-deduped")
    for stage in ("initialization", "early", "mid_late")
)


def canonical_cell_id(model_id: str, training_stage: str) -> str:
    """Return the only accepted logical S2.4 cell spelling."""

    cell_id = f"{model_id}:{training_stage}"
    if cell_id not in EXPECTED_CELL_IDS:
        raise ValueError(f"S204_CELL_ID_INVALID:{cell_id}")
    return cell_id


def cell_path_component(cell_id: str) -> str:
    """Return the reversible, path-safe projection of a canonical cell ID."""

    if cell_id not in EXPECTED_CELL_IDS:
        raise ValueError(f"S204_CELL_ID_INVALID:{cell_id}")
    return cell_id.replace(":", "__")


def cell_id_from_path_component(component: str) -> str:
    """Inverse of :func:`cell_path_component`; reject ambiguous paths."""

    if not isinstance(component, str) or "__" not in component:
        raise ValueError(f"S204_CELL_PATH_COMPONENT_INVALID:{component}")
    cell_id = component.replace("__", ":")
    if cell_path_component(cell_id) != component:
        raise ValueError(f"S204_CELL_PATH_COMPONENT_INVALID:{component}")
    return cell_id


def is_cell_path_component(component: object) -> bool:
    try:
        cell_id_from_path_component(str(component))
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "EXPECTED_CELL_IDS",
    "canonical_cell_id",
    "cell_id_from_path_component",
    "cell_path_component",
    "is_cell_path_component",
]
