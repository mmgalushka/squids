"""A module for handling image palette."""

from enum import Enum


class Palette(str, Enum):
    COLOR = "color"
    """Defines the palette with all possible RGB colors."""
    RGB = "rgb"
    """Defines the palette with three contrasting red/green/blue colors."""
    GRAY = "gray"
    """Defines the palette with all grades between black and white."""
    BINARY = "binary"
    """Defines the palette with just two colors black and white."""

    def __str__(self):
        """Returns the string representation of this class."""
        return str(self.value)

    @staticmethod
    def values():
        """Returns a list of palette values."""
        return set(map(str, Palette))

    @staticmethod
    def default():
        """Returns a default palette value."""
        return Palette.COLOR
