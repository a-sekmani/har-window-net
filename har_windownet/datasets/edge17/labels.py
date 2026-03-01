"""
Edge17 label extraction and mapping.

Extracts action labels from:
1. meta.action_id field in JSONL (primary)
2. Filename pattern as fallback (e.g., ...A008... -> A008)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LABEL_PATTERN = re.compile(r"A(\d{1,3})")


def extract_label_from_meta(meta: dict[str, Any]) -> str | None:
    """
    Extract action label from meta dict.

    Parameters
    ----------
    meta : dict
        Meta line from JSONL file

    Returns
    -------
    str or None
        Label like "A001", "A008", etc., or None if not found
    """
    action_id = meta.get("action_id")
    if action_id and isinstance(action_id, str):
        return normalize_label(action_id)
    return None


def extract_label_from_filename(path: str | Path) -> str | None:
    """
    Extract action label from filename.

    Looks for patterns like A001, A8, A008 in the filename.

    Parameters
    ----------
    path : str or Path
        File path

    Returns
    -------
    str or None
        Normalized label like "A001", or None if not found
    """
    path = Path(path)
    name = path.stem
    match = LABEL_PATTERN.search(name)
    if match:
        num = int(match.group(1))
        return f"A{num:03d}"
    return None


def normalize_label(label: str) -> str:
    """
    Normalize label to A### format (3 digits, zero-padded).

    Examples
    --------
    "A1" -> "A001"
    "A08" -> "A008"
    "A001" -> "A001"
    "A120" -> "A120"
    """
    match = LABEL_PATTERN.match(label)
    if match:
        num = int(match.group(1))
        return f"A{num:03d}"
    return label


def get_label(meta: dict[str, Any], path: str | Path | None = None) -> str:
    """
    Get label from meta or fallback to filename.

    Parameters
    ----------
    meta : dict
        Meta line from JSONL
    path : str or Path, optional
        File path for fallback extraction

    Returns
    -------
    str
        Label like "A001", or "UNKNOWN" if not found
    """
    label = extract_label_from_meta(meta)
    if label:
        return label
    if path:
        label = extract_label_from_filename(path)
        if label:
            return label
    return "UNKNOWN"
