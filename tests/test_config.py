"""
Test for the `BBox` class form `squids/config.py`.
"""

from squids.config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
    IMAGE_CAPACITY,
    DATASET_DIR,
    DATASET_SIZE,
)


def test_default_constants():
    """Tests default_constants."""
    assert IMAGE_WIDTH == 64
    assert IMAGE_HEIGHT == 64
    assert IMAGE_CHANNELS == 3
    assert IMAGE_CAPACITY == 3
    assert DATASET_DIR == "dataset/synthetic"
    assert DATASET_SIZE == 1000
