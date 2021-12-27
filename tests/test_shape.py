"""
Test for shape classes form `squids/shape.py`.
"""

from squids.color import Color
from squids.point import Point
from squids.bbox import BBox
from squids.shape import Rectangle, Triangle

BBOX = BBox(Point(10, 10), 10, 10)
"""Default bounding box."""
COLOR = Color(32, 64, 128)
"""Default color."""


def test_rectangle():
    """Tests the `Rectangle` class."""
    s = Rectangle(BBOX, COLOR)

    assert len(s.polygon.flatten()) == 8
    assert str(s.bbox) == str(BBOX)
    assert str(s.color) == str(COLOR)
    assert s.category_id == 0

    assert "bbox=" in str(s) and "bbox=" in str(s)


def test_triangle():
    """Tests the `Triangle` class."""
    s = Triangle(BBOX, COLOR)

    assert len(s.polygon.flatten()) == 6
    assert str(s.bbox) == str(BBOX)
    assert str(s.color) == str(COLOR)
    assert s.category_id == 1

    assert "bbox=" in str(s) and "polygon=" in str(s)
