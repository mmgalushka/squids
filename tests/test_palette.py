"""
Test for the `Palette` class form `squids/dataset/palette.py`.
"""

from squids.dataset.palette import Palette


def test_palette():
    """Tests the enumerator."""
    assert Palette.BINARY == "binary"
    assert Palette.COLOR == "color"
    assert Palette.RGB == "rgb"
    assert Palette.GRAY == "gray"

    assert Palette.values() == set(["binary", "color", "rgb", "gray"])

    assert Palette.default() == "color"
