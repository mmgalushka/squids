"""
A module for handling a shape on an image.
"""

import numpy as np

from .point import Point
from .polygon import Polygon
from .bbox import BBox
from .color import Color


class Shape:
    """A base shape to define geometrical figure.

    Args:
        polygon (Polygon): The collection of vertices for drawing a
            geometrical figure.
        bbox (BBox): The binding box for encapsulating a geometric
            figure.

    Attributes:
        polygon (Polygon): The collection of vertices for drawing a
            geometrical figure.
        bbox (BBox): The binding box for encapsulating a geometric
            figure.
    """

    def __init__(
        self, polygon: Polygon, bbox: BBox, color: Color, category: str
    ):
        self.polygon = polygon
        self.bbox = bbox
        self.color = color
        self.category = category

    def __str__(self):
        return f"(BBox{str(self.bbox)}, Polygon{str(self.polygon)})"


class Rectangle(Shape):
    """A rectangle shape."""

    def __init__(self, bbox: BBox, color: Color):
        polygon = Polygon(
            [
                bbox.anchor,
                Point(bbox.anchor.x + bbox.width, bbox.anchor.y),
                Point(bbox.anchor.x + bbox.width, bbox.anchor.y + bbox.height),
                Point(bbox.anchor.x, bbox.anchor.y + bbox.height),
            ]
        )
        super().__init__(polygon, bbox, color, "rectangle")


class Triangle(Shape):
    """A triangle shape."""

    def __init__(self, bbox: BBox, color: Color):
        shift = np.random.randint(1, bbox.width)
        polygon = Polygon(
            [
                Point(bbox.anchor.x + shift, bbox.anchor.y),
                Point(bbox.anchor.x + bbox.width, bbox.anchor.y + bbox.height),
                Point(bbox.anchor.x, bbox.anchor.y + bbox.height),
            ]
        )
        super().__init__(polygon, bbox, color, "triangle")
