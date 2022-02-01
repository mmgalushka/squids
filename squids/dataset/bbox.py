"""A module for handling a bounding box on an image."""

from .point import Point
from .polygon import Polygon


class BBox(Polygon):
    """
    A bounding box.

    Args:
        anchor (Point): The top-left point of the bounding box.
        width (int): The bounding box width.
        height (int): The bounding box height.

    Attributes:
        anchor (Point): The top-left point of the bounding box.
        width (int): The bounding box width.
        height (int): The bounding box height.
    """

    def __init__(self, anchor: Point, width: int, height: int):
        super().__init__([anchor, Point(anchor.x + width, anchor.y + height)])

    def __str__(self):
        return f"[{self.anchor}, {self.width}, {self.height}]"

    @property
    def anchor(self):
        return self[0]

    @property
    def width(self):
        return self[-1].x - self[0].x

    @property
    def height(self):
        return self[-1].y - self[0].y

    def flatten(self) -> list:
        """Returns a flattened representation of a bounding box.

        Returns:
            The array `[x, y, width, height]` where `x, y` the bounding box
            anchor coordinates and `width, height` its width and height.
        """
        return [*self.anchor.flatten(), self.width, self.height]
