"""
Test for shape classes form `squids/dataset/image.py`.
"""

from squids.dataset.image import create_synthetic_image
from squids.dataset.palette import Palette
from squids.dataset.background import Background
from squids.dataset.shape import Ellipse, Triangle, Rectangle


def test_image():
    """Tests the `Rectangle` class."""
    for palette in Palette.values():
        image, shapes = create_synthetic_image(
            64, 64, palette, Background.WHITE, 3
        )
        assert image.mode == "RGB"
        assert image.size == (64, 64)

        image, shapes = create_synthetic_image(
            64, 64, palette, Background.BLACK, 3
        )
        assert image.mode == "RGB"
        assert image.size == (64, 64)

        assert len(shapes) > 0 and len(shapes) <= 3
        for shape in shapes:
            assert shape.category_id in [
                Ellipse.category_id,
                Triangle.category_id,
                Rectangle.category_id,
            ]
