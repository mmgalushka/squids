"""
Test for the `Point` class form `squids/dataset/point.py`.
"""

from squids.dataset.point import Point


def test_point_constructor():
    """Tests the constructor."""
    p = Point(1, 2)
    assert p.x == 1
    assert p.y == 2


def test_point_comparison():
    """Tests the `eq` and `ne` methods."""
    p = Point(1, 2)
    assert p == Point(1, 2)
    assert p != Point(0, 2)
    assert p != Point(1, 0)
    assert p != Point(0, 0)


def test_point_flatten():
    """Tests the `flatten` method."""
    p = Point(1, 2)
    assert p.flatten() == [1, 2]


def test_point_str():
    """Tests the `str` method."""
    p = Point(1, 2)
    assert str(p) == "(1, 2)"
