"""
Test for the `Palette` class form `squids/palette.py`.
"""

from squids.palette import Palette


def test_palette():
    """Tests the enumerator."""
    assert Palette.BINARY == "binary"
    assert Palette.COLOR == "color"
    assert Palette.GRAY == "gray"

    assert Palette.values() == set(["binary", "color", "gray"])

    assert Palette.default() == "color"
