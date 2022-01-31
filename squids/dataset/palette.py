"""A module for handling image palette."""

from enum import Enum


class Palette(str, Enum):
    COLOR = "color"
    GRAY = "gray"
    BINARY = "binary"

    def __str__(self):
        return str(self.value)

    @staticmethod
    def values():
        """Returns a list of palette values."""
        return set(map(str, Palette))

    @staticmethod
    def default():
        """Returns a default palette value."""
        return Palette.COLOR
