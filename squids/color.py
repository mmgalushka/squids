"""
A module for handling shapes and image colors.
"""

import random


class Color:
    """A color.

    Args:
        r (int): The red color code between 0 and 255.
        g (int): The green color code between 0 and 255.
        b (int): The blue color code between 0 and 255.

    Attributes:
        red (int): The red color code.
        green int): The green color code.
        blue (int): The blue color code.
    """

    def __init__(self, r: int, g: int, b: int):
        self.red = r
        self.green = g
        self.blue = b

    __GRADIENTS = list(range(25, 255, 25))

    def __str__(self):
        return "#%02x%02x%02x" % (self.red, self.green, self.blue)

    @staticmethod
    def random() -> list:
        """Returns a random color.

        Returns:
            The generated rundom color.
        """
        return Color(
            random.choice(Color.__GRADIENTS),
            random.choice(Color.__GRADIENTS),
            random.choice(Color.__GRADIENTS),
        )


WHITE_COLOR = Color(255, 255, 255)
"""A constant for the `White` color."""

BLACK_COLOR = Color(0, 0, 0)
"""A constant for the `Black` color."""
