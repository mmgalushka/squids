"""
Test for the `Polygon` class form `squids/dataset/polygon.py`.
"""

from squids.dataset.point import Point
from squids.dataset.polygon import Polygon


def test_polygon_constructor():
    """Test the constructor."""
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert len(p) == 2
    assert p[0] == Point(1, 2)
    assert p[1] == Point(3, 4)


def test_polygon_flatten():
    """Tests the `flatten` method."""
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert p.flatten() == [1, 2, 3, 4]


def test_polygon_str():
    """Tests the `str` method."""
    v1 = Point(1, 2)
    v2 = Point(3, 4)
    p = Polygon([v1, v2])
    assert str(p) == "[(1, 2), (3, 4)]"
