"""
Test for the `BBox` class form `squids/bbox.py`.
"""

from squids.image import Point, BBox


def test_bbox_constructor():
    """Tests the constructor."""
    b = BBox(Point(1, 2), 3, 4)
    assert b.anchor == Point(1, 2)
    assert b.width == 3
    assert b.height == 4


def test_bbox_flatten():
    """Tests the `flatten` method."""
    b = BBox(Point(1, 2), 3, 4)
    assert b.flatten() == [1, 2, 3, 4]


def test_bbox_str():
    """Tests the `str` method."""
    b = BBox(Point(1, 2), 3, 4)
    assert str(b) == "[(1, 2), 3, 4]"
