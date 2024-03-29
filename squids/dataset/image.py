"""A module for manipulating images."""

from random import randint, choice

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps

from .color import Color, WHITE_COLOR, BLACK_COLOR
from .background import Background
from .palette import Palette
from .point import Point
from .bbox import BBox
from .shape import Ellipse, Triangle, Rectangle


def create_synthetic_image(
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
) -> tuple:
    """
    Returns a generated image with binding boxes and segments.

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

    n = randint(1, image_capacity)
    for _ in range(n):
        # Picks a random shape.
        shape = choice([Ellipse, Triangle, Rectangle])

        # Picks a random shape color. To maintain the data reproducibility
        # property of this library, we need to be consistent in using a
        # "random-based" operation. In case the request to use binary-color
        # data we still need to call the function for generating random
        # color and ignore it afterward.
        color = Color.random(rgb=image_palette == Palette.RGB)
        if image_palette == Palette.BINARY:
            if image_background == Background.WHITE:
                color = BLACK_COLOR
            else:
                color = WHITE_COLOR
        min_object_size = int(min(image_width, image_height) * 0.1)
        max_object_size = int(min(image_width, image_height) * 0.5)

        w = randint(min_object_size, max_object_size)
        h = randint(min_object_size, max_object_size)

        # Generates the first coordinates of a binding box.
        x = randint(0, image_width - w)
        y = randint(0, image_height - h)

        # Creates a binding box.
        bbox = BBox(Point(x, y), w, h)

        # Collects the object shape (inside the binding box).
        shapes.append(shape(bbox, color))

    # Creates a synthetic image and projecting specified shapes.
    image = PIL.Image.new("RGB", (image_width, image_height), image_background)
    draw = PIL.ImageDraw.Draw(image)
    for shape in shapes:
        draw.polygon(shape.polygon.flatten(), fill=str(shape.color))

    if image_palette == Palette.GRAY:
        image = PIL.ImageOps.grayscale(image).convert("RGB")

    return image, shapes
