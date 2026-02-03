"""Tests for label map utilities (labels.py)."""

import json
import tempfile
from pathlib import Path

import pytest

from har_windownet.contracts.labels import (
    NTU120_ACTION_NAMES,
    build_default_label_map,
    get_label_id,
    get_label_name,
    load_label_map,
    save_label_map,
)


def test_ntu120_action_names_count() -> None:
    """NTU 120 has exactly 120 action labels A001..A120."""
    assert len(NTU120_ACTION_NAMES) == 120
    for i in range(1, 121):
        key = f"A{i:03d}"
        assert key in NTU120_ACTION_NAMES


def test_build_default_label_map() -> None:
    """build_default_label_map returns id_to_name, label_to_id, num_classes."""
    lm = build_default_label_map()
    assert lm["num_classes"] == 120
    assert "id_to_name" in lm
    assert "label_to_id" in lm
    assert len(lm["id_to_name"]) == 120
    assert len(lm["label_to_id"]) == 120


def test_build_default_label_map_consistency() -> None:
    """label_to_id and id_to_name are consistent."""
    lm = build_default_label_map()
    for label, idx in lm["label_to_id"].items():
        assert lm["id_to_name"][str(idx)] == NTU120_ACTION_NAMES[label]


def test_get_label_id() -> None:
    """get_label_id returns correct index for A001..A120."""
    lm = build_default_label_map()
    assert get_label_id(lm, "A001") == 0
    assert get_label_id(lm, "A120") == 119


def test_get_label_name() -> None:
    """get_label_name returns action name for id."""
    lm = build_default_label_map()
    assert "water" in get_label_name(lm, 0).lower() or "drink" in get_label_name(lm, 0).lower()
    assert get_label_name(lm, 119) == "finger-guessing game"


def test_save_and_load_label_map() -> None:
    """save_label_map and load_label_map round-trip."""
    lm = build_default_label_map()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "label_map.json"
        save_label_map(lm, path)
        assert path.exists()
        loaded = load_label_map(path)
        assert loaded["num_classes"] == lm["num_classes"]
        assert loaded["label_to_id"] == lm["label_to_id"]
        assert loaded["id_to_name"] == lm["id_to_name"]


def test_load_label_map_invalid_path() -> None:
    """load_label_map raises on missing file."""
    with pytest.raises(FileNotFoundError):
        load_label_map(Path("/nonexistent/label_map.json"))
