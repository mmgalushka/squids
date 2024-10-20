"""A module for manipulating images."""

import numpy as np
from random import random, randint, choice

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import PIL.ImageFilter

from .color import Color, WHITE_COLOR, BLACK_COLOR
from .background import Background
from .palette import Palette
from .point import Point
from .bbox import BBox
from .shape import Ellipse, Triangle, Rectangle


def gaussian_blur_xy_old(image, radius_x, radius_y):
    """Apply Gaussian blur separately in x and y directions."""
    if radius_x > 0:
        kernel_size = int(radius_x * 4 + 1)  # Ensure odd kernel size
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[center, i] = np.exp(
                -((i - center) ** 2) / (2 * radius_x**2)
            )
        kernel = kernel / kernel.sum()  # Normalize
        img_array = np.array(image)
        for c in range(3):  # Apply to each color channel
            img_array[:, :, c] = np.clip(
                np.convolve(
                    img_array[:, :, c].flatten(), kernel.flatten(), mode="same"
                ).reshape(img_array.shape[:2]),
                0,
                255,
            )
        image = PIL.Image.fromarray(img_array.astype("uint8"))

    if radius_y > 0:
        kernel_size = int(radius_y * 4 + 1)  # Ensure odd kernel size
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[i, center] = np.exp(
                -((i - center) ** 2) / (2 * radius_y**2)
            )
        kernel = kernel / kernel.sum()  # Normalize
        img_array = np.array(image)
        for c in range(3):  # Apply to each color channel
            img_array[:, :, c] = np.clip(
                np.convolve(
                    img_array[:, :, c].flatten(), kernel.flatten(), mode="same"
                ).reshape(img_array.shape[:2]),
                0,
                255,
            )
        image = PIL.Image.fromarray(img_array.astype("uint8"))

    return image


def gaussian_blur_xy(image, radius_x, radius_y):
    """Apply Gaussian blur separately in x and y directions."""
    img_array = np.array(image)

    if radius_x > 0:
        kernel_size = int(radius_x * 4 + 1)  # Ensure odd kernel size
        kernel = np.zeros(kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[i] = np.exp(-((i - center) ** 2) / (2 * radius_x**2))
        kernel = kernel / kernel.sum()  # Normalize

        # Apply horizontal blur
        for y in range(img_array.shape[0]):
            for c in range(3):  # Each color channel
                img_array[y, :, c] = np.convolve(
                    img_array[y, :, c], kernel, mode="same"
                )

    if radius_y > 0:
        kernel_size = int(radius_y * 4 + 1)  # Ensure odd kernel size
        kernel = np.zeros(kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            kernel[i] = np.exp(-((i - center) ** 2) / (2 * radius_y**2))
        kernel = kernel / kernel.sum()  # Normalize

        # Apply vertical blur
        for x in range(img_array.shape[1]):
            for c in range(3):  # Each color channel
                img_array[:, x, c] = np.convolve(
                    img_array[:, x, c], kernel, mode="same"
                )

    return PIL.Image.fromarray(np.clip(img_array, 0, 255).astype("uint8"))


def create_synthetic_image(
    image_width: int,
    image_height: int,
    image_palette: Palette,
    image_background: Background,
    image_capacity: int,
    adding_shapes: str = "rte",
    noise_level: float = 0.0,
    blur_level_x: float = 0.0,
    blur_level_y: float = 0.0,
) -> tuple:
    """
    Returns a generated image with binding boxes and segments.

    Args:
        image_width (int): The image width in pixels.
        image_height (int): The image height in pixels.
        image_palette (Palette): The palette for generating images.
        image_background (Background): The palette for generating images.
        image_capacity (int): The number of geometrical shapes per image.
        adding_shapes (str): A string representing adding shapes, can
            have a combination the following characters: "r" - for
            representing rectangle, "t" - for representing triangle
            and "e" - for representing ellipse.
        noise_level (float): Amount of white noise to add (0.0 to 1.0).
        blur_level_x (float): Amount of blurring on X-axis to add (0.0 to 1.0).
        blur_level_y (float): Amount of blurring on Y-axis to add (0.0 to 1.0).

    Returns:
        image (PIL.Image): The generate image.
        shapes (list): The geometrical shapes the image consists of.
    """
    # Validate parameters
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("'noise_level' must be between 0.0 and 1.0")
    if not 0.0 <= blur_level_x <= 1.0:
        raise ValueError("'blur_level_x' must be between 0.0 and 1.0")
    if not 0.0 <= blur_level_y <= 1.0:
        raise ValueError("'blur_level_y' must be between 0.0 and 1.0")

    shapes_for_selection = [Ellipse, Triangle, Rectangle]
    if adding_shapes:
        shapes_collection = set()
        for adding_shape in adding_shapes:
            if adding_shape == "r":
                shapes_collection.add(Rectangle)
            elif adding_shape == "t":
                shapes_collection.add(Triangle)
            elif adding_shape == "e":
                shapes_collection.add(Ellipse)
            else:
                raise ValueError(
                    "Got unknown shape representing character "
                    f"'{adding_shape}', but supported 'r', 't' and 'e';"
                )
        shapes_for_selection = list(shapes_collection)

    # Convert blur levels to radius values
    if blur_level_x == 1:
        blur_radius_x = random() * int(image_width / 10)
    else:
        blur_radius_x = blur_level_x * int(image_width / 10)

    if blur_level_y == 1:
        blur_radius_y = random() * int(image_height / 10)
    else:
        blur_radius_y = blur_level_y * int(image_height / 10)

    # Creates synthetic image of the  appropriate capacity.
    shapes = []

    # Create base image
    image = PIL.Image.new("RGB", (image_width, image_height), image_background)

    n = randint(1, image_capacity)
    for _ in range(n):
        # Picks a random shape.
        shape = choice(shapes_for_selection)

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

        # Create and store the shape
        shape_obj = shape(bbox, color)
        shapes.append(shape_obj)

    # Creates a synthetic image and projecting specified shapes.
    image = PIL.Image.new("RGB", (image_width, image_height), image_background)
    draw = PIL.ImageDraw.Draw(image)
    for shape in shapes:
        draw.polygon(shape.polygon.flatten(), fill=str(shape.color))

    # Apply blur after all shapes are drawn if needed
    if blur_level_x > 0 or blur_level_y > 0:
        image = gaussian_blur_xy(image, blur_radius_x, blur_radius_y)

    # Add noise if noise_level > 0
    if noise_level > 0:
        # Convert image to numpy array for easier manipulation
        img_array = np.array(image)

        # Generate noise array
        if noise_level == 1:
            noise = np.random.normal(0, 255 * random(), img_array.shape)
        else:
            noise = np.random.normal(0, 255 * noise_level, img_array.shape)

        # Add noise to image
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        image = PIL.Image.fromarray(noisy_img_array)

    if image_palette == Palette.GRAY:
        image = PIL.ImageOps.grayscale(image).convert("RGB")

    return image, shapes
