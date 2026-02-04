"""Tests for Custom10 label parsing (folder name -> label_id, label_name)."""

import pytest

from har_windownet.datasets.custom10.labels import parse_label_from_folder_name


def test_custom10_label_parse_a001_style() -> None:
    """A001_WALKING -> label_id A001, label_name WALKING."""
    out = parse_label_from_folder_name("A001_WALKING")
    assert out is not None
    assert out[0] == "A001"
    assert out[1] == "WALKING"


def test_custom10_label_parse_001_style() -> None:
    """001_SITTING -> label_id A001, label_name SITTING."""
    out = parse_label_from_folder_name("001_SITTING")
    assert out is not None
    assert out[0] == "A001"
    assert out[1] == "SITTING"


def test_custom10_label_parse_with_suffix() -> None:
    """A003_TURNING_DAILY -> A003, TURNING_DAILY."""
    out = parse_label_from_folder_name("A003_TURNING_DAILY")
    assert out is not None
    assert out[0] == "A003"
    assert out[1] == "TURNING_DAILY"


def test_custom10_label_parse_short_digits() -> None:
    """A1_*, A43_* normalized to A001, A043 (1–3 digits zero-padded)."""
    out = parse_label_from_folder_name("A1_drink_water")
    assert out is not None
    assert out[0] == "A001"
    assert out[1] == "drink_water"
    out2 = parse_label_from_folder_name("A43_falling_down")
    assert out2 is not None
    assert out2[0] == "A043"
    assert out2[1] == "falling_down"


def test_custom10_label_parse_invalid() -> None:
    """Invalid folder name returns None."""
    assert parse_label_from_folder_name("invalid") is None
    assert parse_label_from_folder_name("walking") is None
    assert parse_label_from_folder_name("A_foo") is None  # no digits after A
