"""Label parsing from folder names (A001_* or 001_*) and label_map build/save for Custom10."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Folder name: A001_WALKING, 001_WALKING, A1_drink_water, A43_falling -> label_id A001, A001, A001, A043
# Accept 1–3 digits after optional A; normalize to A + 3 digits (zero-padded)
LABEL_FOLDER_PATTERN = re.compile(r"^([Aa]?\d{1,3})_(.+)$")


def parse_label_from_folder_name(folder_name: str) -> tuple[str, str] | None:
    """
    Parse folder name to (label_id, label_name).
    Accepts A001_*, 001_*, A1_*, A43_* (1–3 digits). Returns (A001, rest) with label_id zero-padded to 3 digits.
    """
    m = LABEL_FOLDER_PATTERN.match(folder_name.strip())
    if not m:
        return None
    prefix, rest = m.group(1), m.group(2)
    # Normalize to A + 3 digits
    digits = re.sub(r"^[Aa]", "", prefix)
    if not digits.isdigit():
        return None
    label_id = "A" + digits.zfill(3)
    return label_id, rest


def build_label_map_from_refs(
    refs: list[Any],
    *,
    label_id_attr: str = "label_id",
    label_name_attr: str = "label_name",
) -> dict[str, Any]:
    """
    Build label_map from list of refs (e.g. ClipRef) with label_id and label_name.
    Returns dict: label_to_id (str -> int), id_to_name (str id -> name), label_names (label_id -> name), num_classes.
    Compatible with WindowDataset (label_to_id, id_to_name, num_classes).
    """
    seen: dict[str, str] = {}  # label_id -> label_name (first occurrence)
    for r in refs:
        lid = getattr(r, label_id_attr, None)
        lname = getattr(r, label_name_attr, None)
        if lid and lid not in seen:
            seen[lid] = lname or lid
    labels_sorted = sorted(seen.keys())
    label_to_id: dict[str, int] = {lid: i for i, lid in enumerate(labels_sorted)}
    id_to_name: dict[str, str] = {str(i): seen[lid] for i, lid in enumerate(labels_sorted)}
    label_names: dict[str, str] = dict(seen)
    return {
        "label_to_id": label_to_id,
        "id_to_name": id_to_name,
        "label_names": label_names,
        "num_classes": len(labels_sorted),
    }


def save_label_map(label_map: dict[str, Any], path: str | Path) -> None:
    """Save label_map to JSON (compatible with contracts.labels format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(path: str | Path) -> dict[str, Any]:
    """Load label_map from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
