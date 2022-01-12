"""
A module for handling a point on an image.
"""


class Point:
    """A point.

    Args:
        x (int): The `x` coordinate.
        y (int): The `y` coordinate.

    Attributes:
        x (int): The `x` coordinate.
        y (int): The `y` coordinate.
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def flatten(self) -> list:
        """Returns a flattened representation of a point.

        Returns:
            The array `[x, y]` where `x, y` the point coordinates.
        """
        return [self.x, self.y]
