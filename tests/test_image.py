"""
Test for shape classes form `squids/image.py`.
"""

from squids.image import Palette, Background, create_synthetic_image


def test_image():
    """Tests the `Rectangle` class."""
    image, shapes = create_synthetic_image(
        64, 64, Palette.COLOR, Background.WHITE, 3
    )
    assert image.mode == "RGB"
    assert image.size == (64, 64)

    assert len(shapes) > 0 and len(shapes) <= 3
    for shape in shapes:
        assert shape.category_id in [0, 1]

    image, shapes = create_synthetic_image(
        64, 64, Palette.BINARY, Background.WHITE, 3
    )
    assert image.mode == "RGB"
    assert image.size == (64, 64)

    image, shapes = create_synthetic_image(
        64, 64, Palette.BINARY, Background.BLACK, 3
    )
    assert image.mode == "RGB"
    assert image.size == (64, 64)
