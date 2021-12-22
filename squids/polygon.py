"""
A module for handling a polygon on an image.
"""

from typing import List

from .point import Point


class Polygon(List[Point]):
    """A polygon"""

    def __str__(self):
        return f'[{", ".join(map(str,self))}]'

    def flatten(self) -> list:
        """Returns a flattened representation of a polygon.

        Returns:
            The array `[x1, y1, x2, y2, ...]` where `x_, y_` the polygon
            vertices.
        """
        output = []
        for point in self:
            output.extend(point.flatten())
        return output
