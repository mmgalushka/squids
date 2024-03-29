"""A module for handling image background."""

from enum import Enum


class Background(str, Enum):
    WHITE = "white"
    """Defines the white background."""
    BLACK = "black"
    """Defines the black background."""

    def __str__(self):
        """Returns the string representation of this class."""
        return str(self.value)

    @staticmethod
    def values():
        """Returns a list of background values."""
        return set(map(str, Background))

    @staticmethod
    def default():
        """Returns a default background value."""
        return Background.WHITE
