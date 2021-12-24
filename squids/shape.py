"""
A module for handling a shape on an image.
"""

from abc import ABC, abstractmethod
from random import randint

from .point import Point
from .polygon import Polygon
from .bbox import BBox
from .color import Color


class Shape(ABC):
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

    @abstractmethod
    def get_area(self) -> float:
        return


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

    def get_area(self) -> float:
        return self.bbox.width * self.bbox.height


class Triangle(Shape):
    """A triangle shape."""

    def __init__(self, bbox: BBox, color: Color):
        shift = randint(1, bbox.width)
        polygon = Polygon(
            [
                Point(bbox.anchor.x + shift, bbox.anchor.y),
                Point(bbox.anchor.x + bbox.width, bbox.anchor.y + bbox.height),
                Point(bbox.anchor.x, bbox.anchor.y + bbox.height),
            ]
        )
        super().__init__(polygon, bbox, color, "triangle")

    def get_area(self) -> float:
        return (self.bbox.width * self.bbox.height) / 2.0
