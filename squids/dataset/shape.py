"""A module for handling a shape on an image."""

from abc import ABC, abstractmethod

from random import randint
from math import pi, sin, cos

from .point import Point
from .polygon import Polygon
from .bbox import BBox
from .color import Color


class Shape(ABC):
    """
    A base shape to define geometrical figure.

    Args:
        polygon (Polygon):
            The collection of vertices for drawing a geometrical figure.
        bbox (BBox):
            The binding box for encapsulating a geometric figure.
        color (Color):
            The shape color.

    Attributes:
        polygon (Polygon):
            The collection of vertices for drawing a geometrical figure.
        bbox (BBox):
            The binding box for encapsulating a geometric figure.
        color (Color):
            The shape color.
    """

    def __init__(
        self,
        bbox: BBox,
        polygon: Polygon,
        color: Color,
    ):
        self.bbox = bbox
        self.polygon = polygon
        self.color = color

    @abstractmethod
    def get_area(self) -> float:
        """Computes and returns the area covered by a shape.

        Returns:
            The area covered by a shape.
        """
        raise NotImplementedError(
            "This method should be implemented in the inherited class."
        )


class Ellipse(Shape):
    """A ellipse shape."""

    def __init__(self, bbox: BBox, color: Color):
        xr = bbox.width / 2.0
        yr = bbox.height / 2.0
        xc = bbox.anchor.x + xr
        yc = bbox.anchor.y + yr

        angle = 0.0
        delta = (2.0 * pi) / 50
        coordinates = []
        while angle <= 2.0 * pi:
            coordinates.append(
                Point(int(xr * cos(angle) + xc), int(yr * sin(angle) + yc))
            )
            angle += delta

        polygon = Polygon(coordinates)
        super().__init__(bbox, polygon, color)

    category_name = "ellipse"
    """The shape category name."""
    category_id = 1
    """The shape category identifier."""

    def get_area(self) -> float:
        """Returns the ellipse area."""
        return pi * self.bbox.width * self.bbox.height

    def __str__(self):
        """Returns the string representation of this class."""
        return (
            f"Ellipse(bbox={str(self.bbox)}, "
            f"polygon={str(self.polygon)}, "
            f"color='{str(self.color)}'"
        )


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
        super().__init__(bbox, polygon, color)

    category_name = "triangle"
    """The shape category name."""
    category_id = 2
    """The shape category identifier."""

    def get_area(self) -> float:
        """Returns the rectangle area."""
        return (self.bbox.width * self.bbox.height) / 2.0

    def __str__(self):
        """Returns the string representation of this class."""
        return (
            f"Triangle(bbox={str(self.bbox)}, "
            f"polygon={str(self.polygon)}, "
            f"color='{str(self.color)}'"
        )


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
        super().__init__(bbox, polygon, color)

    category_name = "rectangle"
    """The shape category name."""
    category_id = 3
    """The shape category identifier."""

    def get_area(self) -> float:
        """Returns the rectangle area."""
        return self.bbox.width * self.bbox.height

    def __str__(self):
        """Returns the string representation of this class."""
        return (
            f"Rectangle(bbox={str(self.bbox)}, "
            f"polygon={str(self.polygon)}, "
            f"color='{str(self.color)}'"
        )
