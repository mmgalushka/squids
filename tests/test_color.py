"""
Test for the `Color` class form `squids/dataset/color.py`.
"""

from squids.dataset.color import Color


def test_color_constructor():
    """Tests the constructor."""
    c = Color(32, 64, 128)
    assert c.red == 32
    assert c.green == 64
    assert c.blue == 128


def test_color_flatten():
    """Tests the `random` method."""
    c = Color.random()
    assert len(str(c)) == 7
    c = Color.random(rgb=True)
    assert len(str(c)) == 7


def test_color_str():
    """Tests the `str` method."""
    c = Color(32, 64, 128)
    assert str(c) == "#204080"
