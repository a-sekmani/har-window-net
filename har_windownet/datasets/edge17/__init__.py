"""
Edge17 skeleton JSONL dataset adapter.

Reads .skeleton.jsonl files from Edge pose estimation pipeline (COCO-17, normalized).
"""

from har_windownet.datasets.edge17.builder import build_dataset_edge17
from har_windownet.datasets.edge17.labels import extract_label_from_filename, extract_label_from_meta, get_label
from har_windownet.datasets.edge17.reader import (
    list_edge17_files,
    read_clip,
    read_jsonl_file,
)
from har_windownet.datasets.edge17.windowing import slice_windows

__all__ = [
    "build_dataset_edge17",
    "extract_label_from_filename",
    "extract_label_from_meta",
    "get_label",
    "list_edge17_files",
    "read_clip",
    "read_jsonl_file",
    "slice_windows",
]
