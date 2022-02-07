"""
Test for shape classes form `squids/dataset/shape.py`.
"""
import pytest

from squids.dataset.color import Color
from squids.dataset.point import Point
from squids.dataset.bbox import BBox
from squids.dataset.shape import Shape, Ellipse, Triangle, Rectangle

BBOX = BBox(Point(10, 10), 10, 10)
"""Default bounding box."""
COLOR = Color(32, 64, 128)
"""Default color."""


def test_shape():
    """Tests the `Shape` class."""
    with pytest.raises(NotImplementedError):

        class Xyz(Shape):
            def get_area(self) -> float:
                return super().get_area()

        s = Xyz(None, None, None)
        s.get_area()


def test_ellipse():
    """Tests the `Ellipse` class."""
    s = Ellipse(BBOX, COLOR)

    assert len(s.polygon.flatten()) == 2 * 50
    assert str(s.bbox) == str(BBOX)
    assert str(s.color) == str(COLOR)
    assert s.category_name == "ellipse"
    assert s.category_id == 1

    assert all(
        [
            "Ellipse" in str(s),
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


def test_rectangle():
    """Tests the `Rectangle` class."""
    s = Rectangle(BBOX, COLOR)

    assert len(s.polygon.flatten()) == 8
    assert str(s.bbox) == str(BBOX)
    assert str(s.color) == str(COLOR)
    assert s.category_name == "rectangle"
    assert s.category_id == 3

    assert all(
        [
            "Rectangle" in str(s),
            "bbox=" in str(s),
            "polygon=" in str(s),
            "color=" in str(s),
        ]
    )
