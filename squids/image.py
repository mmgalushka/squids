"""
A module for manipulating images.
"""

from __future__ import annotations

import numpy as np

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps

from .color import Color, WHITE_COLOR, BLACK_COLOR
from .background import Background
from .palette import Palette
from .point import Point
from .bbox import BBox
from .shape import Rectangle, Triangle


IMAGE_WIDTH = 64
"""A default image width."""
IMAGE_HEIGHT = 64
"""A default image height."""
IMAGE_CHANNELS = 3
"""A default image height."""
IMAGE_CAPACITY = 2
"""A default number of geometrical shapes per image."""


def create_synthetic_image(
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
) -> tuple:
    """Returns a generated image with binding boxes and segments.

    Args:
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.

    Returns:
        image (PIL.Image): The generate image.
        shapes (list): The geometrical shapes the image consists of.
    """
    # Creates synthetic image od appropriate capacity
    shapes = []

    n = np.random.randint(1, image_capacity + 1)
    for _ in range(n):
        # Picks a random shape.
        shape = np.random.choice([Rectangle, Triangle])

        # Picks a random shape color.
        if image_palette in [Palette.COLOR, Palette.GRAY]:
            color = Color.random()
        else:
            if image_background == Background.WHITE:
                color = BLACK_COLOR
            else:
                color = WHITE_COLOR

        min_object_size = int(min(image_width, image_height) * 0.1)
        max_object_size = int(min(image_width, image_height) * 0.5)
        w, h = np.random.randint(min_object_size, max_object_size, size=2)

        # Generates the first coordinates of a binding box.
        x = np.random.randint(0, image_width - w)
        y = np.random.randint(0, image_height - h)

        # Creates a binding box.
        bbox = BBox(Point(x, y), w, h)

        # Collects the object shape (inside the binding box).
        shapes.append(shape(bbox, color))

    # Creates a synthetic image and projecting specified shapes.
    image = PIL.Image.new("RGB", (image_width, image_height), image_background)
    draw = PIL.ImageDraw.Draw(image)
    for shape in shapes:
        draw.polygon(shape.polygon.flatten(), fill=str(shape.color))

    return image, shapes
