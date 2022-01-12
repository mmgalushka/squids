"""
Test for shape classes form `squids/dataset/shape.py`.
"""

from squids.dataset.color import Color
from squids.dataset.point import Point
from squids.dataset.bbox import BBox
from squids.dataset.shape import Rectangle, Triangle

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
    assert s.category_name == "rectangle"
    assert s.category_id == 1

    assert all(
        [
            "Rectangle" in str(s),
            "bbox=" in str(s),
            "polygon=" in str(s),
            "color=" in str(s),
        ]
    )


def test_triangle():
    """Tests the `Triangle` class."""
    s = Triangle(BBOX, COLOR)

    assert len(s.polygon.flatten()) == 6
    assert str(s.bbox) == str(BBOX)
    assert str(s.color) == str(COLOR)
    assert s.category_name == "triangle"
    assert s.category_id == 2

    assert all(
        [
            "Triangle" in str(s),
            "bbox=" in str(s),
            "polygon=" in str(s),
            "color=" in str(s),
        ]
    )
