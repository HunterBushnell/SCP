"""Relative imports and constructor-time relative asset lookup."""
from pathlib import Path
from .support import SyntheticCell


def build_cell(scale=1.0, use_mod=False):
    length = float(Path('length.txt').read_text()) * scale
    return SyntheticCell(length, use_mod=use_mod)


def invalid_cell():
    return object()
