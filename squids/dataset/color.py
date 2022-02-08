"""A module for handling shapes and image colors."""

import random


class Color:
    """
    A color.

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
    def random(rgb: bool = False):
        """Returns a random color.

        Args:
            rgb (bool):
                The flag forcing to generate with one of three contrasting
                red/green/blue colors.

        Returns:
            The generated rundom color.
        """
        if rgb:
            return random.choice(
                [Color(204, 0, 0), Color(0, 204, 0), Color(0, 0, 204)]
            )
        else:
            return Color(
                random.choice(Color.__GRADIENTS),
                random.choice(Color.__GRADIENTS),
                random.choice(Color.__GRADIENTS),
            )

    @staticmethod
    def from_hex(code: str):
        """Returns a color created from hex-code.

        Args:
            code:
                The hex-code for the creating color.

        Returns:
            The generated rundom color.
        """
        h = code.lstrip("#")
        red, green, blue = tuple(
            int(h[i : i + 2], 16) for i in (0, 2, 4)  # noqa
        )
        return Color(red, green, blue)


WHITE_COLOR = Color.from_hex("#ffffff")
"""A constant for the `White` color."""

BLACK_COLOR = Color.from_hex("#000000")
"""A constant for the `Black` color."""
